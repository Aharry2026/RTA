import time
import copy
import cv2
import operator
import numpy as np
import csv
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import batched_nms

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters, aggregate_pred_patches

# Optional SAM import
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("Warning: 'segment_anything' not installed. SAM refinement will be disabled.")

REFERENCE_PROMPT = 'a photo of a {}'


class RobustPromptSampler:
    """
    Scoring module for SAM mask candidates based on Semantic Consistency + Geometric metrics.
    """
    
    def __init__(self, alpha=0.0, beta=1.0, exp=1.0, use_semantic=False):
        self.alpha = alpha
        self.beta = beta
        self.exp = exp
        self.use_semantic = use_semantic

    def compute_semantic_score(self, mask, visual_feature_map, text_embedding, device):
        if visual_feature_map is None or text_embedding is None:
            return 0.0
        
        try:
            feat_h, feat_w = visual_feature_map.shape[:2]
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (feat_w, feat_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            mask_tensor = torch.from_numpy(mask_resized).to(device)
            
            if mask_tensor.sum() == 0:
                return 0.0
            
            masked_features = visual_feature_map[mask_tensor]
            avg_visual_feat = masked_features.mean(dim=0)
            avg_visual_feat = F.normalize(avg_visual_feat, p=2, dim=-1)
            text_emb_norm = F.normalize(text_embedding, p=2, dim=-1)
            cosine_sim = torch.dot(avg_visual_feat, text_emb_norm).item()
            semantic_score = max(0.0, cosine_sim)
            return semantic_score
            
        except Exception:
            return 0.0

    def get_mask_scores(self, mask, prob_map, high_response_mask, 
                        visual_feature_map=None, text_embedding=None, device=None):
        if device is None:
            device = prob_map.device
            
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
        mask_area = mask_tensor.sum() + 1e-6

        purity = (mask_tensor * prob_map).sum() / mask_area
        intersection = (mask_tensor * high_response_mask.float()).sum()
        coverage = intersection / (high_response_mask.float().sum() + 1e-6)

        semantic_score = 0.0
        if self.use_semantic:
            if visual_feature_map is not None and text_embedding is not None:
                vis_dim = visual_feature_map.shape[-1]
                text_dim = text_embedding.shape[-1] if text_embedding.dim() > 0 else 1
                
                if vis_dim == text_dim:
                    semantic_score = self.compute_semantic_score(
                        mask, visual_feature_map, text_embedding, device
                    )

        geometric_score = purity.item() * (coverage.item() ** self.exp)
        score = self.alpha * semantic_score + self.beta * geometric_score
        
        metrics = {
            'purity': purity.item(),
            'coverage': coverage.item(),
            'semantic': semantic_score,
            'geometric': geometric_score,
            'score': score
        }
        
        return score, metrics


class RTA:
    """Remember, Trust, and Adapt (RTA) for open-vocabulary segmentation."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, vision_outputs=(-1,), 
                 alpha_cls=0.0, steps=10, prompt_dir='prompts.yaml', 
                 prompt_integration='loss', runtime_calculation=False,
                 device='cpu',
                 # Semantic Consistency parameters
                 use_sam_refinement=True,
                 sam_checkpoint='./weights/sam_vit_h_4b8939.pth',
                 sam_model_type='vit_h',
                 semantic_weight=0.3,
                 geo_weight=0.7,
                 cov_exp=1.0,
                 use_semantic=True,
                 prompt_type='point',
                 post_process='nms',
                 topk_num=1,
                 nms_iou_thresh=0.9,
                 score_thresh=0.1,
                 response_thresh_ratio=0.5,
                 min_mask_area=100,
                 num_sample_points=10,
                 patch_size=(224, 224),
                 patch_stride=112,
                 use_soft_merging=False,
                 # Local cache
                 use_local_cache=True,
                 local_cache_capacity=10,
                 local_cache_beta=5.0,
                 # Fusion
                 temperature=15.0,
                 # Herding (Common)
                 use_herding=True,
                 herding_overshoot_factor=1.0,
                 use_mode_in_herding=False,
                 # === Cache 作用域控制 ===
                 use_cache_for_prompt=False,
                 use_cache_for_final=True,
                 # === 新策略参数 ===
                 local_cache_sample_stride=3,
                 samples_per_mask=3,
                 log_statistics=False,
                 log_filename='mask_stats.csv',
                 save_dir='./save',
                 # === 双层 Cache 阈值参数 ===
                 percentile_low=40,
                 percentile_high=85,
                 min_history_size=50,
                 entropy_clamp_min=0.5,
                 entropy_clamp_max=5.0,
                 ambiguous_entropy_low=1.0,
                 ambiguous_entropy_high=2.5,
                 ):
        """Initialize the paper-aligned RTA pipeline."""

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.classes = classes if classes is not None else []
        self.vision_outputs = vision_outputs
        print(f"+++ Vision output layers: {self.vision_outputs}")

        self.alpha_cls = alpha_cls
        self.steps = steps
        self.prompt_dir = prompt_dir
        self.runtime = runtime_calculation
        self.device = device

        # Semantic Consistency parameters
        self.use_sam_refinement = use_sam_refinement
        self.semantic_weight = semantic_weight
        self.geo_weight = geo_weight
        self.cov_exp = cov_exp
        self.use_semantic = use_semantic
        self.prompt_type = prompt_type
        self.post_process = post_process
        self.topk_num = topk_num
        self.nms_iou_thresh = nms_iou_thresh
        self.score_thresh = score_thresh
        self.response_thresh_ratio = response_thresh_ratio
        self.min_mask_area = min_mask_area  # 用于 Prompt 生成
        self.num_sample_points = num_sample_points
        self.patch_size = patch_size if isinstance(patch_size, tuple) else tuple(patch_size)
        self.patch_stride = patch_stride

        # === Local cache ===
        self.use_soft_merging = use_soft_merging

        self.use_local_cache = use_local_cache
        self.local_cache_capacity = local_cache_capacity
        self.local_cache_beta = local_cache_beta
        self.local_cache_reliable = {}
        self.local_cache_ambiguous = {}
        self.local_cache_negative = {}
        
        # Fusion
        self.temperature = temperature
        
        # Herding (Common)
        self.use_herding = use_herding
        self.herding_overshoot_factor = herding_overshoot_factor
        self.use_mode_in_herding = use_mode_in_herding

        # === Cache 作用域控制 ===
        self.use_cache_for_prompt = use_cache_for_prompt
        self.use_cache_for_final = use_cache_for_final

        # === 新策略参数 ===
        self.local_cache_sample_stride = local_cache_sample_stride
        self.samples_per_mask = samples_per_mask
        
        # === 百分位动态阈值参数 (新增) ===
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.min_history_size = min_history_size
        self.entropy_clamp_min = entropy_clamp_min
        self.entropy_clamp_max = entropy_clamp_max
        
        # === 初始化动态阈值（默认值作为 fallback）===
        self.ambiguous_entropy_low = ambiguous_entropy_low
        self.ambiguous_entropy_high = ambiguous_entropy_high
        
        # The paper uses continuous reliability intervals.
        self.percentile_bands = [
            (0, percentile_low),              # Reliable band
            (percentile_low, percentile_high),       # Uncertain band
            (percentile_high, 100),            # Exclusionary band
        ]
        # 存储计算后的绝对数值区间
        self.valid_entropy_bands = []
        
        # === 熵值历史队列 (用于百分位统计) ===
        self.entropy_history = []
        
        # === 日志 ===
        self.log_statistics = log_statistics
        self.log_path = os.path.join(save_dir, log_filename)
        self.processed_images = 0
        if self.log_statistics:
            os.makedirs(save_dir, exist_ok=True)
            if not os.path.exists(self.log_path):
                with open(self.log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['img_idx', 'class_idx', 'class_name', 'score', 'entropy', 'area', 'action'])

        # Load OVSS model
        self.model, self.tokenize = load_ovss(self.ovss_type, self.ovss_backbone, device=self.device)

        if self.prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        assert prompt_integration in ['loss', 'text']
        self.prompt_integration = prompt_integration

        # Setup LayerNorm gradients
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)

        params, _ = self.collect_ln_params(self.model.visual)
        print_clip_parameters(self.model)

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        # Extract text features
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(self.classes, self.prompt_templates, average=True).squeeze()

        # Initialize SAM if enabled
        if self.use_sam_refinement:
            if not HAS_SAM:
                raise ImportError("SAM not installed")
            
            print(f"\n{'='*60}")
            print(f"+++ SAM REFINEMENT ENABLED +++")
            print(f"+++ SAM model: {sam_model_type} from {sam_checkpoint}")
            self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            self.sam_model.to(device)
            self.sam_model.eval()
            self.sam_predictor = SamPredictor(self.sam_model)
            
            self.sampler = RobustPromptSampler(
                alpha=semantic_weight,
                beta=geo_weight,
                exp=cov_exp,
                use_semantic=use_semantic
            )
            print(f"+++ Semantic Consistency: weight={semantic_weight}, use_semantic={use_semantic}")
            print(f"+++ Geometric: weight={geo_weight}, exp={cov_exp}")
            print(f"+++ Prompt type: {prompt_type}, Post-process: {post_process}")
            print(f"+++ Score threshold: {score_thresh}, Response ratio: {response_thresh_ratio}")
            print(f"+++ Patch size: {self.patch_size}, Patch stride: {self.patch_stride}")
            print(f"+++ Soft Merging: {'Enabled' if self.use_soft_merging else 'Disabled'}")
            print(f"+++ Local Cache: {'Enabled' if self.use_local_cache else 'Disabled'}")
            if self.use_local_cache:
                print(f"+++   Capacity: {self.local_cache_capacity}, Beta: {self.local_cache_beta}")
            if self.use_local_cache:
                print(f"+++ Cache Method: {'Herding' if self.use_herding else 'Entropy-based'}")
                print(f"+++ Cache for Prompt: {'Enabled' if self.use_cache_for_prompt else 'Disabled'}")
                print(f"+++ Cache for Final: {'Enabled' if self.use_cache_for_final else 'Disabled'}")
                print(f"+++ Temperature: {self.temperature}, Overshoot: {self.herding_overshoot_factor}")
            print(f"+++ Strategy: Entropy Stride {self.local_cache_sample_stride} + Herding Maintenance")
            print(f"{'='*60}\n")
            
            self.sam_stats = {
                'total_images': 0,
                'total_classes_processed': 0,
                'total_masks_generated': 0,
                'total_masks_kept': 0,
                'classes_refined': 0,
                'avg_score': [],
            }
        else:
            print(f"\n{'='*60}")
            print(f"+++ SAM REFINEMENT DISABLED +++")
            print(f"{'='*60}\n")
            self.sam_model = None
            self.sam_predictor = None
            self.sampler = None
            self.sam_stats = None

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def compute_dynamic_weight(self, source_logits, cache_logits, override_temperature=None):
        temp = override_temperature if override_temperature is not None else self.temperature
        
        if temp != 1.0:
            source_logits_scaled = source_logits / temp
            cache_logits_scaled = cache_logits / temp
        else:
            source_logits_scaled = source_logits
            cache_logits_scaled = cache_logits
        
        entropy_source = self.softmax_entropy(source_logits_scaled, dim=0)
        entropy_cache = self.softmax_entropy(cache_logits_scaled, dim=0)
        
        p = 1.0 - entropy_cache / (entropy_source + entropy_cache + 1e-8)
        return p

    def _compute_semantic_ios(self, masks, obj_sim):
        n_masks = masks.shape[0]
        if n_masks == 0:
            return torch.zeros(0, device=masks.device)

        flat_masks = masks.flatten(1).float()
        inter_num = flat_masks @ flat_masks.t()
        inter_num.fill_diagonal_(0.0)
        inter_num = torch.tril(inter_num, diagonal=0)
        pos_num = flat_masks.sum(dim=1)
        _ios = inter_num / (pos_num[:, None] + 1e-6)
        
        if obj_sim is not None:
            _ios = _ios * obj_sim
            
        ios = _ios.max(dim=-1)[0]
        return ios

    def _post_process_masks(self, candidates, cls_idx=None, spatial_feature_map=None):
        if not candidates:
            return [], []
        
        masks_list = [c['mask'] for c in candidates]
        scores = torch.tensor([c['score'] for c in candidates], device=self.device)
        boxes = torch.stack([c['box'] for c in candidates])
        masks_tensor = torch.stack(masks_list)
        
        if self.post_process == 'nms':
            idxs = torch.zeros(len(candidates), dtype=torch.long, device=self.device)
            keep = batched_nms(boxes, scores, idxs, self.nms_iou_thresh)
            if self.topk_num > 0:
                # keep = keep[:self.topk_num * 10]
                keep = keep[:self.topk_num]  #修改

        elif self.post_process == 'topk':
            k = min(self.topk_num, len(candidates))
            _, keep = torch.topk(scores, k)
            
        elif self.post_process == 'none':
            keep = torch.arange(len(candidates), device=self.device)
        else:
            raise ValueError(f"Unknown post_process strategy: {self.post_process}")

        if self.use_soft_merging and spatial_feature_map is not None and len(keep) > 0:
            scores_out = scores[keep]
            masks_out_binary = masks_tensor[keep]
            
            feat_H, feat_W, D = spatial_feature_map.shape
            
            masks_small = F.interpolate(
                masks_out_binary.unsqueeze(1).float(),
                size=(feat_H, feat_W),
                mode='nearest'
            ).squeeze(1).bool()
            
            obj_feats_out = []
            for i in range(len(keep)):
                m = masks_small[i]
                if m.sum() > 0:
                    f = spatial_feature_map[m].mean(dim=0)
                else:
                    f = torch.zeros(D, device=self.device, dtype=spatial_feature_map.dtype)
                obj_feats_out.append(f)
            
            obj_feats_out = torch.stack(obj_feats_out)
            obj_feats_out = F.normalize(obj_feats_out, p=2, dim=-1)
            
            obj_sim = obj_feats_out @ obj_feats_out.t()
            obj_sim = obj_sim.clamp(min=0.0)
            
            sorted_vals, sorted_idx = torch.sort(scores_out, descending=True)
            
            masks_out_binary_sorted = masks_out_binary[sorted_idx]
            obj_sim_sorted = obj_sim[sorted_idx][:, sorted_idx]
            
            ios = self._compute_semantic_ios(masks_out_binary_sorted, obj_sim_sorted)
            score_decay = 1.0 - ios
            scores_decayed = sorted_vals * torch.pow(score_decay, 0.5)
            
            final_masks_tensor = masks_out_binary_sorted
            final_scores_tensor = scores_decayed

        else:
            final_masks_tensor = masks_tensor[keep]
            final_scores_tensor = scores[keep]

        final_masks = [m for m in final_masks_tensor]
        final_scores = final_scores_tensor.tolist()
        
        return final_masks, final_scores

    def adapt(self, x):
        self.reset()
        loss_report = self.perform_adaptation(x)
        return loss_report

    @torch.no_grad() 
    def evaluate(self, x, original_imgs=None, meta=None):
        """评估方法：根据配置选择 SAM 细化或标准评估"""
        t1 = time.time()

        patch_grid_shape = None
        image_shapes = None
        if meta is not None:
            patch_grid_shape = meta.get('patch_grid_shape')
            image_shapes = meta.get('img_shape')

        if self.use_sam_refinement and original_imgs is not None and patch_grid_shape is not None and image_shapes is not None:
            results = self._evaluate_with_sam(x, original_imgs, patch_grid_shape, image_shapes)
        else:
            logits, _, _ = self.model(x, self.text_x[-1], True, vision_outputs=self.vision_outputs, 
                                      interpolate=True, vision_out_type="adaptive_weighted_mean", 
                                      save_weights=True)
            results = logits[0]

        t2 = time.time()
        if self.runtime:
            self.eval_times.append(t2 - t1)

        return results

    def _extract_spatial_features(self, img_np, target_shape):
        """提取 CLIP 空间特征图"""
        mean = torch.tensor([122.7709, 116.7460, 104.0937], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([68.5005, 66.6322, 70.3232], device=self.device).view(1, 3, 1, 1)
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor = F.interpolate(img_tensor, size=target_shape, mode='bilinear', align_corners=False)
        img_tensor = (img_tensor - mean) / std
        
        features = self.model.encode_image(img_tensor, output_layers=(-1,), out_type="mean")
        spatial_tokens = features[:, 1:, :]
        
        if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
            proj_matrix = self.model.visual.proj
            proj_in_dim, proj_out_dim = proj_matrix.shape
            if proj_in_dim == spatial_tokens.shape[-1]:
                spatial_tokens = spatial_tokens @ proj_matrix
        
        n_patches = spatial_tokens.shape[1]
        patch_size = self.model.visual.patch_size
        grid_h = target_shape[0] // patch_size
        grid_w = target_shape[1] // patch_size
        
        if grid_h * grid_w != n_patches:
            grid_size = int(n_patches ** 0.5)
            grid_h, grid_w = grid_size, grid_size
        
        spatial_feature_map = spatial_tokens.reshape(1, grid_h, grid_w, -1).squeeze(0)
        return spatial_feature_map

    def _compute_cache_entropy_weight(
            self,
            cache,
            gamma=1.0,
            min_w=0.1,
            max_w=0.5,
        ):
            if cache is None or len(cache) == 0:
                return None

            ent_list = []
            for cls_idx in cache:
                for item in cache[cls_idx]:
                    if len(item) >= 2:
                        ent_list.append(float(item[1]))

            if len(ent_list) == 0:
                return None

            mean_ent = sum(ent_list) / len(ent_list)
            w = math.exp(-gamma * mean_ent)
            w = max(min_w, min(max_w, w))
            return w


    def _compute_affinity_logits(self, cache_dict, spatial_feature_map, image_shape, num_classes, beta):

        if not cache_dict:
            return None
        
        feat_h, feat_w, D = spatial_feature_map.shape
        device = spatial_feature_map.device
        
        # 确保是 float32
        if spatial_feature_map.dtype == torch.float16:
            spatial_feature_map = spatial_feature_map.float()
        
        # === 1. 提取按类别分组的 cache keys ===
        cache_keys = []
        cache_spans = []
        
        for cls_idx in sorted(cache_dict.keys()):
            start = len(cache_keys)
            for item in cache_dict[cls_idx]:
                feature = item[0]
                if feature.dtype == torch.float16:
                    feature = feature.float()
                cache_keys.append(feature.unsqueeze(0))
            cache_spans.append((cls_idx, start, len(cache_keys)))
        
        if not cache_keys:
            return None
        
        # 拼接：(#cache_samples, D) -> (D, #cache_samples)
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        
        # === 2. 归一化特征 ===
        spatial_feature_map_flat = spatial_feature_map.reshape(-1, D)
        
        # L2 归一化
        spatial_feature_map_norm = F.normalize(spatial_feature_map_flat, p=2, dim=-1)
        cache_keys_norm = F.normalize(cache_keys.permute(1, 0), p=2, dim=-1)
        
        # === 3. 计算亲和度 ===
        # affinity: (feat_h*feat_w, #cache_samples)
        affinity = spatial_feature_map_norm @ cache_keys_norm.permute(1, 0)
        
        # === 4. 计算 logits ===
        # The paper takes the maximum cache affinity independently for each
        # class, rather than summing or averaging cache entries.
        cache_scores = torch.exp(beta * (affinity - 1.0))
        cache_logits = torch.zeros(
            spatial_feature_map_flat.shape[0], num_classes, device=device,
            dtype=cache_scores.dtype,
        )
        for cls_idx, start, end in cache_spans:
            cache_logits[:, cls_idx] = cache_scores[:, start:end].amax(dim=1)
        
        # === 5. 重塑为空间维度 ===
        cache_logits = cache_logits.reshape(feat_h, feat_w, num_classes)
        cache_logits = cache_logits.permute(2, 0, 1)  # (num_classes, feat_h, feat_w)
        
        # === 6. [关键修复] 上采样到原始图像尺寸 ===
        cache_logits_upsampled = F.interpolate(
            cache_logits.unsqueeze(0),
            size=image_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return cache_logits_upsampled

    def _read_local_cache_logits(self, spatial_feature_map, image_shape, num_classes):
        """Read the three paper cache bands after the current sample is stored."""
        if not self.use_local_cache or spatial_feature_map is None:
            return None

        reliable_logits = self._compute_affinity_logits(
            self.local_cache_reliable, spatial_feature_map,
            image_shape, num_classes, self.local_cache_beta
        )
        uncertain_logits = self._compute_affinity_logits(
            self.local_cache_ambiguous, spatial_feature_map,
            image_shape, num_classes, self.local_cache_beta
        )
        exclusionary_logits = self._compute_affinity_logits(
            self.local_cache_negative, spatial_feature_map,
            image_shape, num_classes, self.local_cache_beta
        )

        cache_logits = reliable_logits
        if cache_logits is not None and uncertain_logits is not None:
            uncertain_weight = self.compute_dynamic_weight(
                cache_logits, uncertain_logits
            ).unsqueeze(0)
            cache_logits = (
                uncertain_weight * uncertain_logits
                + (1.0 - uncertain_weight) * cache_logits
            )
        elif cache_logits is None:
            cache_logits = uncertain_logits

        if exclusionary_logits is not None:
            cache_logits = (
                -exclusionary_logits
                if cache_logits is None
                else cache_logits - exclusionary_logits
            )
        return cache_logits

    @torch.no_grad()
    def evaluate_from_memory(self, x, original_imgs, patch_grid_shape, image_shapes):
        """Predict after TTA using memory already updated by the current sample.

        This lets the public loop follow the paper order: Remember/Trust,
        Adapt, then entropy-fuse cache and original CLIP logits.
        """
        patch_logits, _, _ = self.model(
            x, self.text_x[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=True,
            vision_out_type="adaptive_weighted_mean"
        )
        coarse_logits_list = aggregate_pred_patches(
            patch_logits[0], patch_grid_shape, image_shapes,
            self.patch_size, self.patch_stride
        )

        results = []
        num_classes = coarse_logits_list[0].shape[0]
        for img_idx, (coarse_logits, img_np) in enumerate(
                zip(coarse_logits_list, original_imgs)):
            if img_np is None:
                results.append(coarse_logits)
                continue
            if coarse_logits.dim() == 4:
                coarse_logits = coarse_logits.squeeze(0)

            ori_H, ori_W = img_np.shape[:2]
            logit_H, logit_W = coarse_logits.shape[1:]
            coarse_logits_ori = F.interpolate(
                coarse_logits.unsqueeze(0), size=(ori_H, ori_W),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            spatial_feature_map = None
            if self.use_semantic or self.use_local_cache:
                spatial_feature_map = self._extract_spatial_features(
                    img_np, image_shapes[img_idx]
                )

            cache_logits = self._read_local_cache_logits(
                spatial_feature_map, (ori_H, ori_W), num_classes
            )
            final_logit = coarse_logits_ori
            if self.use_cache_for_final and cache_logits is not None:
                cache_weight = self.compute_dynamic_weight(
                    coarse_logits_ori, cache_logits
                ).unsqueeze(0)
                final_logit = (
                    cache_weight * cache_logits
                    + (1.0 - cache_weight) * coarse_logits_ori
                )
            results.append(F.interpolate(
                final_logit.unsqueeze(0), size=(logit_H, logit_W),
                mode='bilinear', align_corners=False
            ).squeeze(0))
        return results

    def _evaluate_with_sam(self, x, original_imgs, patch_grid_shape, image_shapes):
        """核心评估方法：SAM + 动态百分位双层 Cache"""
        patch_logits, _, _ = self.model(
            x, self.text_x[-1], True, 
            vision_outputs=self.vision_outputs,
            interpolate=True, 
            vision_out_type="adaptive_weighted_mean"
        )
        patch_logits = patch_logits[0]
        
        coarse_logits_list = aggregate_pred_patches(
            patch_logits, patch_grid_shape, image_shapes,
            self.patch_size, self.patch_stride
        )
        
        final_results = []
        num_classes = coarse_logits_list[0].shape[0]
        
        if self.sam_stats:
            self.sam_stats['total_images'] += len(original_imgs)
        
        for img_idx, (coarse_logits, img_np) in enumerate(zip(coarse_logits_list, original_imgs)):
            global_img_idx = self.processed_images + img_idx
            
            if img_np is None:
                final_results.append(coarse_logits)
                continue
            
            if coarse_logits.dim() == 4:
                coarse_logits = coarse_logits.squeeze(0)
            
            ori_H, ori_W = img_np.shape[:2]
            logit_H, logit_W = coarse_logits.shape[1], coarse_logits.shape[2]
            
            coarse_logits_ori = F.interpolate(
                coarse_logits.unsqueeze(0),
                size=(ori_H, ori_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            self.sam_predictor.set_image(img_np)
            
            spatial_feature_map = None
            if self.use_semantic or self.use_local_cache:
                target_shape = image_shapes[img_idx]
                spatial_feature_map = self._extract_spatial_features(img_np, target_shape)
            
            # Remember and Trust use the current sample's original CLIP
            # prediction.  The cache is intentionally read only after this
            # sample has been inserted, matching Algorithm 1 in the paper.
            probs = coarse_logits_ori.softmax(dim=0)
            entropy_map = self.softmax_entropy(coarse_logits_ori, dim=0)
            
            # === 第一遍循环：收集当前图片所有 Mask 的综合指标（Mask Composite Metric）===
            current_img_mask_metrics = []
            temp_candidates = []
            
            # Get text features for region entropy calculation
            text_features = self.text_x[-1]
            text_features = F.normalize(text_features, p=2, dim=-1)
            logit_scale = self.model.logit_scale.exp().item() if hasattr(self.model, 'logit_scale') else 100.0
            
            # Calculate theoretical maximum entropy for normalization: ln(C)
            num_cls = text_features.shape[0] if text_features.dim() > 1 else 1
            max_entropy = math.log(num_cls) if num_cls > 1 else 1.0

            for cls_idx in range(num_classes):
                if self.classes and len(self.classes) > cls_idx:
                    class_name = self.classes[cls_idx].lower()
                    if class_name == 'background':
                        continue
                
                prob_map = probs[cls_idx]
                max_prob = prob_map.max()
                
                if self.sam_stats:
                    self.sam_stats['total_classes_processed'] += 1
                
                if max_prob < 0.1:
                    continue
                    
                thresh = self.response_thresh_ratio * max_prob
                high_response_mask = (prob_map > thresh)
                
                if high_response_mask.sum() < self.min_mask_area:
                    continue
                
                prompts = self._generate_sam_prompts_v2(prob_map, high_response_mask, (ori_H, ori_W))
                if prompts is None:
                    continue
                
                points, point_labels, boxes = prompts
                
                try:
                    masks, scores, _ = self.sam_predictor.predict_torch(
                        point_coords=points,
                        point_labels=point_labels,
                        boxes=boxes,
                        multimask_output=True
                    )
                    
                    if self.sam_stats:
                        self.sam_stats['total_masks_generated'] += masks.shape[0] * 3
                        
                except Exception:
                    continue
                
                text_emb = self.text_x[-1, cls_idx] if self.text_x.dim() > 1 else self.text_x[cls_idx]
                
                candidates = self._score_and_select_masks_v2(
                    masks, scores, prob_map, high_response_mask, 
                    cls_idx=cls_idx,
                    spatial_feature_map=spatial_feature_map,
                    text_embedding=text_emb
                )
                
                if not candidates:
                    continue
                
                final_masks, final_scores = self._post_process_masks(
                    candidates, 
                    cls_idx=cls_idx,
                    spatial_feature_map=spatial_feature_map
                )
                
                if len(final_masks) == 0:
                    continue
                
                if self.sam_stats:
                    self.sam_stats['total_masks_kept'] += len(final_masks)
                    self.sam_stats['avg_score'].extend(final_scores)
                    self.sam_stats['classes_refined'] += 1
                
                # === 收集熵值和候选项 (更新：计算综合指标) ===
                for mask, score in zip(final_masks, final_scores):
                    area = mask.sum().item()
                    if area == 0:
                        continue
                    
                    # Original Pixel Entropy Mean
                    mask_entropy_pixel_mean = entropy_map[mask].mean().item()
                    
                    if spatial_feature_map is not None:
                        mask_resized = F.interpolate(
                            mask.float().unsqueeze(0).unsqueeze(0),
                            size=spatial_feature_map.shape[:2],
                            mode='nearest'
                        ).squeeze().bool()
                        
                        if mask_resized.sum() > 0:
                            ent_map_small = F.interpolate(
                                entropy_map.unsqueeze(0).unsqueeze(0),
                                size=spatial_feature_map.shape[:2],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                            
                            # === 计算综合指标 (Mask-Level for remember, patch-level for trust) ===
                            feats_in_mask = spatial_feature_map[mask_resized]

                            # 1) Patch Entropy: Mask 内每个 patch 的局部熵 (Vector: N)
                            feats_in_mask_norm = F.normalize(feats_in_mask, p=2, dim=-1)
                            patch_logits = logit_scale * feats_in_mask_norm @ text_features.t()
                            patch_entropies = self.softmax_entropy(patch_logits, dim=-1)  # (N,)

                            # Patch 平均熵: bar(H_patch)
                            avg_patch_ent = patch_entropies.mean().item()

                            # 2) Mask Global Entropy: H_mask
                            avg_feat = feats_in_mask.mean(dim=0).unsqueeze(0)
                            avg_feat = F.normalize(avg_feat, p=2, dim=-1)
                            sim_logits = logit_scale * avg_feat @ text_features.t()
                            mask_global_ent = self.softmax_entropy(sim_logits, dim=-1).item()
                            # Cache labels come from the region's mean CLIP
                            # similarity, not from the class used to prompt SAM.
                            region_cls_idx = int(sim_logits.argmax(dim=-1).item())

                            # 3) Mask Uncertainty: (1 - Score_mask)
                            mask_uncertainty = 1.0 - score

                            # 4) Mask Composite Metric:
                            # M_mask = avg_patch_ent + mask_global_ent + (1 - score)
                            mask_metric = avg_patch_ent + mask_global_ent + mask_uncertainty

                            # 历史阈值统计按 mask 级别更新（每个 mask 记录一个值）
                            current_img_mask_metrics.append(mask_metric)

                            temp_candidates.append({
                                'cls_idx': region_cls_idx,
                                'mask': mask,
                                'mask_resized': mask_resized,
                                'score': score,
                                'area': area,
                                'mask_metric': mask_metric,
                                'patch_entropies': patch_entropies,
                                'mask_global_ent': mask_global_ent,
                                'mask_uncertainty': mask_uncertainty,
                                'spatial_feature_map': spatial_feature_map,
                                'ent_map_small': ent_map_small
                            })
                    
            
            # === 第二步：更新历史统计并计算动态阈值 ===
            if len(current_img_mask_metrics) > 0:
                self.entropy_history.extend(current_img_mask_metrics)
            
            if len(self.entropy_history) >= self.min_history_size:
                # 使用列表存储计算出的多个绝对数值区间
                self.valid_entropy_bands = []
                
                for p_low, p_high in self.percentile_bands:
                    # Clamp percentile values to [0, 100]
                    p_low = max(0, min(p_low, 100))
                    p_high = max(0, min(p_high, 100))
                    if p_low >= p_high:
                        continue
                    
                    # 分别计算当前段的百分位对应值
                    val_low = np.percentile(self.entropy_history, p_low)
                    val_high = np.percentile(self.entropy_history, p_high)
                    
                    # 独立应用截断逻辑 (Clamp)
                    val_low = max(val_low, self.entropy_clamp_min)
                    val_high = min(val_high, self.entropy_clamp_max)
                    
                    # 异常保护：如果截断导致区间倒挂或重合
                    if val_low >= val_high:
                        val_high = val_low + 0.5
                    
                    # 存入当前计算好的物理阈值对
                    self.valid_entropy_bands.append((val_low, val_high))
                
                # 兼容旧逻辑：同时更新 ambiguous_entropy_low/high 供日志使用
                if len(self.valid_entropy_bands) >= 2:
                    self.ambiguous_entropy_low = self.valid_entropy_bands[0][1]   # Reliable 上界
                    self.ambiguous_entropy_high = self.valid_entropy_bands[1][1]  # Ambiguous 上界
            
            # === 第三步：利用动态阈值执行 Cache 更新 ===
            for cand in temp_candidates:
                cls_idx = cand['cls_idx']
                mask_resized = cand['mask_resized']
                mask_metric = cand['mask_metric']
                mask_global_ent = cand['mask_global_ent']
                spatial_feature_map = cand['spatial_feature_map']
                patch_entropies = cand['patch_entropies']
                score = cand['score']
                area = cand['area']

                # 日志记录
                if self.log_statistics:
                    with open(self.log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else str(cls_idx)
                        cache_type = 'Reliable' if mask_metric < self.ambiguous_entropy_low else \
                                    'Ambiguous' if mask_metric < self.ambiguous_entropy_high else 'Negative'
                        writer.writerow([global_img_idx, cls_idx, cls_name, f"{score:.4f}",
                                       f"{mask_metric:.4f}", area, cache_type])

                # Local Cache：先按 mask 分层，再在层内筛选代表 patch
                if self.use_local_cache:
                    self._update_local_cache(
                        cls_idx, spatial_feature_map, mask_resized,
                        mask_metric,
                        patch_entropies,
                        mask_global_ent
                    )

            # Read memory only after Remember/Trust has updated it with the
            # current sample. Fuse cache and original CLIP logits using their
            # entropy, as specified by the paper.
            cache_logits = self._read_local_cache_logits(
                spatial_feature_map, (ori_H, ori_W), num_classes
            )

            final_logit_ori = coarse_logits_ori
            if self.use_cache_for_final and cache_logits is not None:
                cache_weight = self.compute_dynamic_weight(
                    coarse_logits_ori, cache_logits
                ).unsqueeze(0)
                final_logit_ori = (
                    cache_weight * cache_logits
                    + (1.0 - cache_weight) * coarse_logits_ori
                )

            self.sam_predictor.reset_image()
            
            final_logit = F.interpolate(
                final_logit_ori.unsqueeze(0),
                size=(logit_H, logit_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            final_results.append(final_logit)
        
        self.processed_images += len(original_imgs)
        return final_results

    def _generate_sam_prompts_v2(self, prob_map, high_response_mask, img_shape):
        points, point_labels, boxes = None, None, None

        # Paper prompt construction: discard small connected components first,
        # then select the top-10 response points from the remaining region.
        mask_np = high_response_mask.detach().cpu().numpy().astype(np.uint8)
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            mask_np, connectivity=8
        )
        filtered_np = np.zeros_like(mask_np)
        for component_idx in range(1, component_count):
            if component_stats[component_idx, cv2.CC_STAT_AREA] >= self.min_mask_area:
                filtered_np[component_labels == component_idx] = 1
        high_response_mask = torch.from_numpy(filtered_np.astype(bool)).to(prob_map.device)
        ys, xs = torch.where(high_response_mask)
        
        if len(xs) == 0:
            return None
        
        if self.prompt_type == 'box':
            mask_np = filtered_np
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            box_list = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_mask_area:
                    continue
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                box = [x_c, y_c, x_c + w_c, y_c + h_c]
                box_list.append(box)
            
            if box_list:
                boxes = torch.tensor(box_list, dtype=torch.float32, device=self.device)
                boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, img_shape)
        
        elif self.prompt_type == 'point':
            vals = prob_map[ys, xs]
            k = min(len(vals), self.num_sample_points)
            _, topk_idx = torch.topk(vals, k)
            
            sampled_xs = xs[topk_idx].float()
            sampled_ys = ys[topk_idx].float()
            
            points = torch.stack([sampled_xs, sampled_ys], dim=1).unsqueeze(1)
            point_labels = torch.ones(points.shape[0], 1, dtype=torch.int, device=self.device)
            points = self.sam_predictor.transform.apply_coords_torch(points, img_shape)
        
        if boxes is None and points is None:
            return None
        
        return points, point_labels, boxes

    def _score_and_select_masks_v2(self, masks, sam_scores, prob_map, high_response_mask, 
                                    cls_idx=None, spatial_feature_map=None, text_embedding=None):
        """Score mask candidates using Semantic Consistency + Geometric metrics."""
        candidates = []
        
        mask_H, mask_W = masks.shape[2], masks.shape[3]
        prob_H, prob_W = prob_map.shape
        
        if prob_H != mask_H or prob_W != mask_W:
            prob_map_resized = F.interpolate(
                prob_map.unsqueeze(0).unsqueeze(0),
                size=(mask_H, mask_W),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            hr_mask_resized = F.interpolate(
                high_response_mask.float().unsqueeze(0).unsqueeze(0),
                size=(mask_H, mask_W),
                mode='nearest'
            ).squeeze().bool()
        else:
            prob_map_resized = prob_map
            hr_mask_resized = high_response_mask
        
        for idx in range(masks.shape[0]):
            best_score = -1
            best_mask = None
            
            for layer_idx in range(3):
                mask = masks[idx, layer_idx]
                mask_np = mask.cpu().numpy().astype(np.uint8)
                
                text_emb_for_scoring = text_embedding
                if text_embedding is not None and text_embedding.dim() > 1:
                    text_emb_for_scoring = text_embedding.squeeze()
                
                score, metrics = self.sampler.get_mask_scores(
                    mask_np, prob_map_resized, hr_mask_resized,
                    visual_feature_map=spatial_feature_map,
                    text_embedding=text_emb_for_scoring,
                    device=self.device
                )
                
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_score > self.score_thresh and best_mask is not None:
                ys_m, xs_m = torch.where(best_mask)
                if len(xs_m) > 0:
                    box = torch.tensor(
                        [xs_m.min().item(), ys_m.min().item(), 
                         xs_m.max().item(), ys_m.max().item()],
                        dtype=torch.float32, device=self.device
                    )
                    candidates.append({
                        'mask': best_mask,
                        'score': best_score,
                        'box': box
                    })
        
        return candidates

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for iter in range(self.steps):
            if self.prompt_integration == 'loss':
                logits, _, _, cls_logits = self.model(x, self.text_x[:-1], True, interpolate=False,
                                                      vision_outputs=self.vision_outputs, return_vanilla_cls=True, 
                                                      vision_out_type="mean")
                
                entropy_per_pixel = self.softmax_entropy(logits)
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                
                loss = entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()

            elif self.prompt_integration == 'text':
                logits, _, _ = self.model(x, self.text_x[-1], True, interpolate=False,
                                         vision_outputs=self.vision_outputs)
                entropy_per_pixel = self.softmax_entropy(logits)
                loss = entropy_per_pixel.mean()
            else:
                raise Exception("prompt_integration should be either on 'loss' or 'text'")
        
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        t2 = time.time()
        if self.runtime:
            self.adapt_times.append(t2-t1)

        return loss_report

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def set_ln_grads(model):
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

    def _update_cache_herding(self, cache_dict, cls_idx, new_features, new_entropies, capacity):
        """
        更新缓存，使用 Herding 算法维护容量。
        
        Args:
            cache_dict: 缓存字典
            cls_idx: 类别索引
            new_features: 新特征 [N, D] 张量
            new_entropies: 新熵值 [N] 张量或标量
            capacity: 容量上限
        """
        if cls_idx not in cache_dict:
            cache_dict[cls_idx] = []
        
        # 确保 new_entropies 是张量
        if not isinstance(new_entropies, torch.Tensor):
            new_entropies = torch.tensor([new_entropies], device=self.device)
        elif new_entropies.dim() == 0:
            new_entropies = new_entropies.unsqueeze(0)
        
        # 添加新样本
        for i in range(len(new_features)):
            ent_value = new_entropies[i].item() if i < len(new_entropies) else new_entropies[0].item()
            cache_dict[cls_idx].append([new_features[i].detach().clone(), ent_value])
            
        # 如果未超容量，直接返回
        if len(cache_dict[cls_idx]) <= capacity:
            return

        # 超容量处理
        if not self.use_herding:
            # 简单的熵值排序
            cache_dict[cls_idx] = sorted(cache_dict[cls_idx], key=lambda x: x[1])[:capacity]
        else:
            # Herding 算法
            candidates = cache_dict[cls_idx]
            features = torch.stack([x[0] for x in candidates])
            
            if self.use_mode_in_herding:
                mean_feat = torch.mean(features, dim=0)
                dists = torch.norm(features - mean_feat, dim=1)
                target = features[torch.argmin(dists)]
            else:
                target = torch.mean(features, dim=0)
            
            selected_indices = []
            temp_residual = target.clone()
            
            for _ in range(capacity):
                scores = torch.matmul(features, temp_residual)
                for idx in selected_indices:
                    scores[idx] = -float('inf')
                best_idx = torch.argmax(scores).item()
                selected_indices.append(best_idx)
                temp_residual += self.herding_overshoot_factor * (target - features[best_idx])
            
            cache_dict[cls_idx] = [candidates[i] for i in selected_indices]

    def _update_local_cache(self, cls_idx, spatial_feature_map, mask_bool, mask_metric, patch_entropies, mask_global_entropy=None):
        """
        三层 Cache 更新策略（Remember -> Trust）：
        1) Remember: 按 mask_metric 将整块 mask 分到一个子缓存层。
        2) Trust: 在目标子缓存层内部，对该 mask 的所有 patch 执行 stride + herding/top-k。
        Args:
            mask_metric (float): Mask Composite Metric scalar.
            patch_entropies (Tensor): (N,) patch-level entropy vector used for trust-stage ordering.
            mask_global_entropy (float): Mask Global Entropy (H_mask), used for cache maintenance ranking.
        """
        if not self.use_local_cache:
            return

        # 提取掩码内的特征
        features = spatial_feature_map[mask_bool]  # (N, D)

        if features.shape[0] == 0:
            return

        # patch_entropies 与 features 长度对齐保护
        if not isinstance(patch_entropies, torch.Tensor):
            patch_entropies = torch.tensor(patch_entropies, device=self.device, dtype=features.dtype)
        if patch_entropies.dim() == 0:
            patch_entropies = patch_entropies.unsqueeze(0)
        patch_entropies = patch_entropies.to(features.device)

        if patch_entropies.shape[0] != features.shape[0]:
            patch_entropies = patch_entropies[:features.shape[0]]
            if patch_entropies.shape[0] < features.shape[0]:
                pad_len = features.shape[0] - patch_entropies.shape[0]
                if patch_entropies.numel() > 0:
                    pad_val = patch_entropies.mean()
                else:
                    pad_val = torch.tensor(mask_metric, device=features.device, dtype=features.dtype)
                patch_entropies = torch.cat([patch_entropies, pad_val.repeat(pad_len)], dim=0)

        # Remember：按 mask_metric 决定目标子缓存
        target_cache = None
        cache_type = None

        if len(self.valid_entropy_bands) >= 3:
            reliable_low, reliable_high = self.valid_entropy_bands[0]
            ambiguous_low, ambiguous_high = self.valid_entropy_bands[1]
            negative_low, negative_high = self.valid_entropy_bands[2]

            if reliable_low <= mask_metric <= reliable_high:
                target_cache = self.local_cache_reliable
                cache_type = 'reliable'
            elif ambiguous_low <= mask_metric <= ambiguous_high:
                target_cache = self.local_cache_ambiguous
                cache_type = 'ambiguous'
            elif negative_low <= mask_metric <= negative_high:
                target_cache = self.local_cache_negative
                cache_type = 'negative'
        else:
            # Fallback：动态区间尚未稳定时使用连续阈值
            if mask_metric < self.ambiguous_entropy_low:
                target_cache = self.local_cache_reliable
                cache_type = 'reliable'
            elif mask_metric < self.ambiguous_entropy_high:
                target_cache = self.local_cache_ambiguous
                cache_type = 'ambiguous'
            else:
                target_cache = self.local_cache_negative
                cache_type = 'negative'

        if target_cache is None:
            return

        # Trust：在目标层内对整块 mask 的 patch 做代表性筛选
        self._process_sub_cache(
            cls_idx, features, patch_entropies,
            target_cache=target_cache,
            cache_type=cache_type,
            mask_global_entropy=mask_metric if mask_global_entropy is None else mask_global_entropy
        )

    def _process_sub_cache(self, cls_idx, features, entropies, target_cache, cache_type, mask_global_entropy):
        """
        2026/02/01：SH默认方案：通用处理流程：Sort by Entropy -> Stride Sampling -> Herding/TopK -> Store
        """
        # --- Step 1: Diversity Filter (Stride) ---
        sorted_indices = torch.argsort(entropies, descending=False)
        stride_indices = sorted_indices[::self.local_cache_sample_stride]
        
        candidates = features[stride_indices]
        
        if candidates.shape[0] == 0: 
            return
        
        # 归一化特征
        candidates = F.normalize(candidates, p=2, dim=-1)
        
        # --- Step 2: Representation Selection (Herding vs Top-K) ---
        K = min(self.samples_per_mask, candidates.shape[0])
        
        selected_features = []
        
        if K == candidates.shape[0]:
            selected_features = candidates
        elif self.use_herding:
            target = torch.mean(candidates, dim=0)
            
            # 对于 Ambiguous/Negative Cache，强烈建议使用 Mode
            use_mode = self.use_mode_in_herding or (cache_type in ['ambiguous', 'negative'])
            
            if use_mode:
                dists = torch.norm(candidates - target, dim=1)
                target = candidates[torch.argmin(dists)]

            selected_indices = []
            temp_residual = target.clone()
            
            for _ in range(K):
                scores = torch.matmul(candidates, temp_residual)
                for idx in selected_indices:
                    scores[idx] = -float('inf')
                
                best_idx = torch.argmax(scores).item()
                selected_indices.append(best_idx)
                temp_residual = temp_residual + self.herding_overshoot_factor * (target - candidates[best_idx])
            
            selected_features = candidates[selected_indices]
        else:
            selected_features = candidates[:K]
            
        # --- Step 3: Store into Target Cache ---
        if cls_idx not in target_cache:
            target_cache[cls_idx] = []
        
        for feat in selected_features:
            target_cache[cls_idx].append([feat.detach().clone(), mask_global_entropy])  # 使用掩码全局熵值
            
        # --- Step 4: Maintenance ---
        self._maintain_cache_size(target_cache, cls_idx)

    def _maintain_cache_size(self, cache_dict, cls_idx):
        """
        容量维护：保留熵值最低（最可靠/最典型）的样本。
        """
        if cls_idx not in cache_dict:
            return
            
        if len(cache_dict[cls_idx]) <= self.local_cache_capacity:
            return
        
        # 按熵值升序排序 (Low -> High)
        cache_dict[cls_idx].sort(key=lambda x: x[1])
        
        # 截断
        cache_dict[cls_idx] = cache_dict[cls_idx][:self.local_cache_capacity]
