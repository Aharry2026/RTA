import numpy as np  

import torch

import mmcv
import mmengine.fileio as fileio
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

from mmcv.transforms import BaseTransform, TRANSFORMS
from mmcv.transforms import to_tensor


from utils.imagecorruptions import corrupt
from utils.imagecorruptions import get_corruption_names


@TRANSFORMS.register_module()
class SaveOriginalImage(BaseTransform):
    """
    保存原始图像以供 SAM 后续使用。
    应在 LoadImageFromFile 之后、任何图像处理之前调用。
    如果图像有 3 个通道，则将 BGR 转换为 RGB。
    """
    def __init__(self, key='img'):
        super().__init__()
        self.key = key

    def transform(self, results: dict) -> dict:
        if self.key in results:
            img = results[self.key]
            # 确保是连续内存的副本
            if img.ndim == 3:
                # BGR 转 RGB
                img = np.ascontiguousarray(img[:, :, ::-1])
            else:
                img = np.ascontiguousarray(img)
            results['original_img'] = img
        return results


@TRANSFORMS.register_module()
class CorruptTransform(BaseTransform):
    """
    对图像应用腐蚀/噪声变换。
    用于鲁棒性测试。
    """
    def __init__(self, corruption_name: str, corruption_severity: int = 5):
        super().__init__()
        self.corruption_name = corruption_name
        self.corruption_severity = corruption_severity

        if self.corruption_name not in get_corruption_names():
            raise ValueError(f"腐蚀名称 {self.corruption_name} 无效。\n请从以下选项中选择: {get_corruption_names()}")

    def transform(self, results: dict) -> dict:
        """ 
        Args:
            results (dict): 输入数据字典。
        Returns:
            dict: 腐蚀后的数据字典。

        注意：输入图像应为 numpy 数组（BGR 或 RGB）。
        """
        img = results['img']
        img_index = results['sample_idx']  

        # 保存当前随机数生成器状态
        rng_state = np.random.get_state()
        
        # 基于索引设置种子以确保可复现性
        np.random.seed(img_index)

        # 对图像应用腐蚀
        results['img'] = corrupt(img, severity=self.corruption_severity, corruption_name=self.corruption_name)
        
        # 恢复原始随机数生成器状态
        np.random.set_state(rng_state)
        
        return results
    

@TRANSFORMS.register_module()
class ResizeAndPatchify(BaseTransform):
    """
    ResizeAndPatchify 是一个变换类，用于调整图像及其对应分割图的大小，
    然后从调整后的图像中提取 patches。
    
    属性:
        resize (tuple): 调整图像和分割图大小的目标尺寸。
        patch_size (tuple): 要提取的 patch 尺寸。
        patch_stride (int): 提取 patch 的步长。
        backend (str): 用于调整大小的后端（默认为 'cv2'）。
    """
    def __init__(self, resize: tuple = (560, 448), patch_size: tuple = (224, 224), patch_stride: int = 112, backend='cv2'):
        super().__init__()
        self.resize = resize
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.backend = backend

        if resize:
            num_patches = ((resize[0] - patch_size[0]) // patch_stride + 1) * ((resize[1] - patch_size[1]) // patch_stride + 1)
            print(f"提取的 patch 数量将为: {num_patches}")
        else:
            print("不进行 patch 提取。将使用图像的原始尺寸。")

    def transform(self, results: dict) -> dict:
        """
        Args:
            results (dict): 包含 'img' 和 'gt_seg_map' 键的输入数据字典。
        
        Returns:
            dict: 为 'img' 和 'gt_seg_map' 添加了 patches 的修改后数据字典。
        
        注意：输入图像和分割图应为 numpy 数组。
        """
        img = results['img']
        gt_seg_map = results['gt_seg_map']  # 真实分割图
        h, w = img.shape[:2]
        
        if self.resize:
            # 根据图像方向确定目标调整尺寸
            target_size = self.resize if w > h else self.resize[::-1]
            
            # 调整图像和分割图的大小
            resized_img, w_scale, h_scale = mmcv.imresize(img, 
                                                        target_size,
                                                        interpolation='bilinear',  # 图像使用双线性插值
                                                        return_scale=True,
                                                        backend=self.backend)
            
            resized_seg_map = mmcv.imresize(gt_seg_map,
                                            target_size,
                                            interpolation='nearest',  # 分割图使用最近邻插值
                                            backend=self.backend)

            # 使用调整后的图像更新结果字典
            results['img'] = resized_img
            results['img_shape'] = resized_img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['gt_seg_map'] = resized_seg_map

            # 计算每个维度的 patch 数量
            target_h, target_w = resized_img.shape[:2]
            num_patches_y = (target_h - self.patch_size[1]) // self.patch_stride + 1
            num_patches_x = (target_w - self.patch_size[0]) // self.patch_stride + 1

            # 定义图像和分割图的 patch 形状和步长
            img_shape = (num_patches_y, num_patches_x, self.patch_size[1], self.patch_size[0], resized_img.shape[2])
            seg_map_shape = (num_patches_y, num_patches_x, self.patch_size[1], self.patch_size[0])

            img_strides = (
                resized_img.strides[0] * self.patch_stride,
                resized_img.strides[1] * self.patch_stride,
                resized_img.strides[0],
                resized_img.strides[1],
                resized_img.strides[2],
            )

            seg_map_strides = (
                resized_seg_map.strides[0] * self.patch_stride,
                resized_seg_map.strides[1] * self.patch_stride,
                resized_seg_map.strides[0],
                resized_seg_map.strides[1],
            )

            # 使用步长视图提取图像和分割图的 patches
            img_patches = np.lib.stride_tricks.as_strided(resized_img, shape=img_shape, strides=img_strides)
            img_patches = img_patches.reshape(-1, self.patch_size[1], self.patch_size[0], resized_img.shape[2])

            gt_seg_map_patches = np.lib.stride_tricks.as_strided(resized_seg_map, shape=seg_map_shape, strides=seg_map_strides)
            gt_seg_map_patches = gt_seg_map_patches.reshape(-1, self.patch_size[1], self.patch_size[0])

            # 使用图像和分割图的 patches 更新结果
            results['patches'] = img_patches.copy()  # 添加 copy() 确保内存连续
            results['gt_seg_map_patches'] = gt_seg_map_patches.copy()
            results['num_patches'] = img_patches.shape[0]
            results['patch_shape'] = img_patches.shape[1:3]
            results['patch_grid_shape'] = (num_patches_y, num_patches_x)

        else:
            results['scale_factor'] = (1.0, 1.0)
            results['patches'] = img[None, ...]
            results['gt_seg_map_patches'] = gt_seg_map[None, ...]
            results['num_patches'] = 1
            results['patch_shape'] = img.shape[:2]
            results['patch_grid_shape'] = (1, 1)

        return results
    


@TRANSFORMS.register_module()
class ToTensorAndNormalize(BaseTransform):
    """
    将数据（字典）转换为张量的方法。
    
    执行以下操作：
    1. 转换为张量并转置为 (C, H, W)
    2. 将 BGR 转换为 RGB
    3. 将像素值从 [0, 255] 缩放到 [0, 1]
    4. 使用均值和标准差归一化图像
    """

    def __init__(self, mean, std,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'patch_shape', 'scale_factor',
                            'patch_grid_shape')
                            ):
        
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:  
        packed_results = dict()
        
        if 'img' in results:
            img = results['img']
            img = img.transpose(2, 0, 1)
            img = to_tensor(img).contiguous()
            # 转换为 RGB
            if img.shape[0] == 3:
                img = img[[2, 1, 0], ...]
            
            # === [删除除以 255.0] ===
            # 保持 0-255 范围，与 CLIP_MEAN 和 CLIP_STD 相匹配
            img = (img - self.mean) / self.std
            packed_results['img'] = img

        if 'patches' in results:
            patches = results['patches']
            patches = patches.transpose(0, 3, 1, 2)
            patches = to_tensor(patches).contiguous()
            # 转换为 RGB
            if patches.shape[1] == 3:
                patches = patches[:, [2, 1, 0], ...]
            
            # === [删除除以 255.0] ===
            # 保持 0-255 范围，与 CLIP_MEAN 和 CLIP_STD 相匹配
            patches = (patches - self.mean[None]) / self.std[None]
            packed_results['patches'] = patches

        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            if len(gt_seg_map.shape) == 2:
                gt_seg_map = to_tensor(gt_seg_map[None, ...].astype(np.int64)).contiguous()
                packed_results['gt_seg_map'] = gt_seg_map
            else:
                raise ValueError('请注意您的真实分割图，'
                                '通常分割图是 2D 的，但得到了 '
                                f'{gt_seg_map.shape}')
        
        if 'gt_seg_map_patches' in results:
            gt_seg_map_patches = results['gt_seg_map_patches']
            gt_seg_map_patches = to_tensor(gt_seg_map_patches[:, None].astype(np.int64)).contiguous()
            packed_results['gt_seg_map_patches'] = gt_seg_map_patches

        # 保存原始图像以供 SAM 使用
        if 'original_img' in results:
            packed_results['original_img'] = results['original_img']

        img_meta = {}   
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        packed_results['meta'] = img_meta

        return packed_results



