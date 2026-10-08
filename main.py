# Standard
import os, time, argparse, contextlib

# Third-party
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local
from adapt import get_method
from utils import segmentation_datasets
from utils.metrics import intersect_and_union, process_metrics
from utils.misc import set_global_seeds, save_configuration, aggregate_pred_patches


def _sum_profiler_flops(prof):
    """Safely sum FLOPs from torch profiler events."""
    total_flops = 0
    for evt in prof.key_averages():
        flops = getattr(evt, 'flops', 0)
        if flops is not None:
            total_flops += flops
    return float(total_flops)


@contextlib.contextmanager
def _suppress_profiler_output(enabled=True):
    """Suppress noisy torch.profiler stdout/stderr output when profiling."""
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def argparser():
    parser = argparse.ArgumentParser(
        description="Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation"
    )
    
    # ----------------------------------------
    # I/O Directories
    # ----------------------------------------
    parser.add_argument(
        '--save_dir',
        type=str,
        default='save/',
        help='Directory to save model weights and results'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.data/',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default='prompts.yaml',
        help='Path to the YAML file containing prompt templates'
    )
    
    # ----------------------------------------
    # Dataset Settings
    # ----------------------------------------
    parser.add_argument(
        '--dataset',
        type=str,
        default='COCOStuffDataset',
        choices=(
            'COCOStuffDataset', 'COCOObjectDataset', 'CityscapesDataset',
            'PascalVOC20Dataset', 'PascalVOC21Dataset',
            'PascalContext59Dataset', 'PascalContext60Dataset'
        ),
        help='Which dataset to load'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of data-loading workers'
    )
    parser.add_argument(
        '--init_resize',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Resize images before patch extraction. '
            'Order doesn’t matter (e.g., (560,448) same as (448,560)). '
            'If None, use original size (batch_size must be 1).'
        )
    )
    parser.add_argument(
        '--patch_size',
        nargs='+',
        type=int,
        default=None,
        help='Size of each image patch after resize (model input size)'
    )
    parser.add_argument(
        '--patch_stride',
        type=int,
        default=112,  # 改为：添加默认值 112
        help='Stride for extracting patches'
    )
    parser.add_argument(
        '--shuffle_data',
        action='store_true',
        dest='shuffle_data',
        help='Shuffle dataset order in DataLoader (random loading order)'
    )
    parser.add_argument(
        '--no_shuffle_data',
        action='store_false',
        dest='shuffle_data',
        help='Disable DataLoader shuffle (directory/list order)'
    )
    parser.set_defaults(shuffle_data=True)
    
    # ----------------------------------------
    # Model Settings
    # ----------------------------------------
    parser.add_argument(
        '--ovss_type',
        type=str,
        default='naclip',
        help='Open-Vocabulary Semantic Segmentation type (e.g., naclip, clip, clip, etc.)'
    )
    parser.add_argument(
        '--ovss_backbone',
        type=str,
        default='ViT-L/14',
        help='Frozen NACLIP vision backbone used by the paper configuration'
    )
    parser.add_argument(
        '--class_extensions',
        action='store_true',
        help='Enable dataset-specific class extensions if available'
    )
    
    # ----------------------------------------
    # Adaptation / Training Settings
    # ----------------------------------------
    parser.add_argument(
        '--adapt',
        action='store_true',
        help='Enable test-time adaptation'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='rta',
        choices=('rta',),
        help='The paper implementation exposes RTA only'
    )
    parser.add_argument(
        '--batch_size', '--batch-size',
        type=int,
        default=128,
        dest='batch_size',
        help='Batch size for adaptation'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for adaptation optimizer'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        help='Number of TTA iterations per batch'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=3,
        help='Number of experimental repetitions'
    )
    
    # ----------------------------------------
    # Debug / Misc
    # ----------------------------------------
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--plot_loss',
        action='store_true',
        help='Plot the loss curve (averaged over batches and seeds)'
    )
    parser.add_argument(
        '--runtime_calculation',
        action='store_true',
        help='Calculate the runtime of adaptation and evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--profile_gflops',
        action='store_true',
        help='Enable GFLOPS profiling and reporting during runtime'
    )
    parser.add_argument(
        '--gflops_profile_batches',
        type=int,
        default=1,
        help='Number of initial batches to profile for GFLOPS averaging (>=1)'
    )

    return parser

def add_method_specific_args(parser, method):
    """Add only the paper configuration knobs exposed by the public entry point."""
    if method != 'rta':
        raise ValueError("The clean release exposes only the RTA method.")

    parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
    parser.add_argument('--prompt_integration', type=str, default='loss',
                        choices=('loss', 'text'))
    parser.add_argument('--alpha_cls', type=float, default=1.0)

    parser.add_argument('--use_sam_refinement',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Use the paper SAM refinement stage.')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='./weights/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_model_type', type=str, default='vit_h',
                        choices=('vit_h',))
    parser.add_argument('--semantic_weight', type=float, default=0.3)
    parser.add_argument('--geo_weight', type=float, default=0.7)
    parser.add_argument('--cov_exp', type=float, default=1.0)
    parser.add_argument('--use_semantic', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--prompt_type', type=str, default='point',
                        choices=('point',))
    parser.add_argument('--post_process', type=str, default='nms',
                        choices=('nms',))
    parser.add_argument('--topk_num', type=int, default=1)
    parser.add_argument('--nms_iou_thresh', type=float, default=0.9)
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--response_thresh_ratio', type=float, default=0.5)
    parser.add_argument('--min_mask_area', type=int, default=100)
    parser.add_argument('--num_sample_points', type=int, default=10)

    parser.add_argument('--temperature', type=float, default=15.0)
    parser.add_argument('--use_herding', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--herding_overshoot_factor', type=float, default=1.0)
    parser.add_argument('--use_mode_in_herding', action='store_true', default=False)
    parser.add_argument('--use_local_cache', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--local_cache_capacity', type=int, default=10)
    parser.add_argument('--local_cache_beta', type=float, default=5.0)
    parser.add_argument('--use_cache_for_final', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--local_cache_sample_stride', type=int, default=3)
    parser.add_argument('--samples_per_mask', type=int, default=3)
    parser.add_argument('--percentile_low', type=float, default=40.0)
    parser.add_argument('--percentile_high', type=float, default=85.0)
    parser.add_argument('--min_history_size', type=int, default=50)
    parser.add_argument('--entropy_clamp_min', type=float, default=0.5)
    parser.add_argument('--entropy_clamp_max', type=float, default=5.0)
    parser.add_argument('--ambiguous_entropy_low', type=float, default=1.0)
    parser.add_argument('--ambiguous_entropy_high', type=float, default=2.5)
    parser.add_argument('--log_statistics', action='store_true')
    parser.add_argument('--log_filename', type=str, default='rta_mask_stats.csv')
    return parser
def main(args):

    # Save the configuration settings
    save_configuration(args)

    # Start the timer
    start_time = time.time()

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the save directory if it doesn't exist
    all_results_path = os.path.join(args.save_dir, "results.txt")
    os.makedirs(os.path.dirname(all_results_path), exist_ok=True)

    # create necessary variables
    all_results = dict()
    headers = "mIoU, mDice, mAcc"
    adapt_time_all_corr = []
    eval_time_all_corr = []

    total_profiled_batches = 0
    total_profiled_images = 0
    total_profiled_total_flops = 0.0
    total_profiled_adapt_flops = 0.0
    total_profiled_eval_flops = 0.0

    if args.profile_gflops and args.gflops_profile_batches < 1:
        raise ValueError(f"gflops_profile_batches must be >= 1, got {args.gflops_profile_batches}")
    
    # The clean release evaluates the paper's original split only. Corruption
    # sweeps remain historical/ablation records outside this directory.
    for c_idx, corruption in enumerate(("original",)):
        print(f"+++ DataLoader shuffle enabled: {args.shuffle_data}")
        data_loader, org_classes = segmentation_datasets.prepare_data(args.dataset, args.data_dir, args.init_resize,
                                                                  args.patch_size, args.patch_stride, corruption=corruption, 
                                                                  batch_size=args.batch_size, num_workers=args.workers,
                                                                  shuffle=args.shuffle_data)
        
        # Check if the extensions of classes should be used
        if args.class_extensions and data_loader.dataset.class_extensions is not None:
            ext_classes = data_loader.dataset.class_extensions
            args.classes = ext_classes
            print(f"\n+++ Using class extensions")
            print(f"+++ The number of classes [no extension]: {len(org_classes)}")
            print(f"+++ The number of classes after extension:  {len(ext_classes)}")

        else:
            args.classes = org_classes
            print(f"\n+++ The number of classes [no extension]: {len(org_classes)}")

        ignore_index = data_loader.dataset.ignore_index

        # Setting up the model and the method
        adapt_method = get_method(args, device)

        # Results path
        c_results_path = os.path.join(args.save_dir, f"{c_idx:02}_{corruption}", "results.txt")
        os.makedirs(os.path.dirname(c_results_path), exist_ok=True)

        miou_seeds = []
        dice_seeds = []
        acc_seeds = []
        loss_seed_report = []

        for t in range(args.trials):
            profiled_batches_this_trial = 0
            results = []
            loss_batch_report = []
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):

                # 移除 debug 限制，让代码运行完整数据集
                # if args.debug and batch_idx == 10: 
                #     break

                inputs = data['img_patches'] 
                labels = data['gt_patches']  
                original_gts = data['gt'] 
                original_imgs = data['original_imgs']

                patch_grid_shape = data['meta']['patch_grid_shape'] 
                image_shapes = data['meta']['img_shape']
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                should_profile_batch = (
                    args.profile_gflops
                    and profiled_batches_this_trial < args.gflops_profile_batches
                )

                if should_profile_batch:
                    profile_activities = [torch.profiler.ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)

                    # reset the model before adapting to a new batch
                    adapt_method.reset()

                    adapt_flops = 0.0
                    eval_flops = 0.0
                    memory_prepared = False
                    if (args.adapt
                            and hasattr(args, 'use_sam_refinement')
                            and args.use_sam_refinement):
                        with torch.no_grad():
                            adapt_method.evaluate(
                                inputs,
                                original_imgs=original_imgs,
                                meta=data['meta']
                            )
                        memory_prepared = True
                    if args.adapt:
                        with _suppress_profiler_output(True):
                            with torch.profiler.profile(
                                activities=profile_activities,
                                with_flops=True,
                                record_shapes=False,
                                profile_memory=False,
                            ) as prof_adapt:
                                with torch.enable_grad():
                                    loss_iter_report = adapt_method.adapt(inputs)
                                loss_batch_report.append(loss_iter_report)
                        adapt_flops = _sum_profiler_flops(prof_adapt)

                    with _suppress_profiler_output(True):
                        with torch.profiler.profile(
                            activities=profile_activities,
                            with_flops=True,
                            record_shapes=False,
                            profile_memory=False,
                        ) as prof_eval:
                            # perform evaluation with optional SAM refinement
                            with torch.no_grad():
                                if hasattr(args, 'use_sam_refinement') and args.use_sam_refinement:
                                    if memory_prepared:
                                        reconstructed_preds = adapt_method.evaluate_from_memory(
                                            inputs,
                                            original_imgs=original_imgs,
                                            patch_grid_shape=patch_grid_shape,
                                            image_shapes=image_shapes
                                        )
                                    else:
                                        reconstructed_preds = adapt_method.evaluate(
                                            inputs,
                                            original_imgs=original_imgs,
                                            meta=data['meta']
                                        )
                                else:
                                    patch_preds = adapt_method.evaluate(inputs)
                                    if args.init_resize:
                                        # 从 meta 中获取所需参数
                                        patch_grid_shape = data['meta']['patch_grid_shape']
                                        image_shapes = data['meta']['img_shape']
                                        reconstructed_preds = aggregate_pred_patches(
                                            patch_preds, patch_grid_shape, image_shapes,
                                            args.patch_size, args.patch_stride
                                        )
                                    else:
                                        reconstructed_preds = patch_preds

                    eval_flops = _sum_profiler_flops(prof_eval)
                    total_flops = adapt_flops + eval_flops

                    batch_images = max(1, len(original_gts))
                    batch_patches = max(1, int(inputs.shape[0]))

                    total_profiled_adapt_flops += adapt_flops
                    total_profiled_eval_flops += eval_flops
                    total_profiled_total_flops += total_flops
                    total_profiled_batches += 1
                    total_profiled_images += batch_images
                    profiled_batches_this_trial += 1

                    # 需要查看逐 batch GFLOPS 时，把下面这段取消注释
                    print(
                        f"[GFLOPS] batch={batch_idx + 1}, "
                        f"eval_per_image={eval_flops / (1e9 * batch_images):.4f}, "
                        f"eval_per_patch={eval_flops / (1e9 * batch_patches):.4f}, "
                        f"adapt_per_image={adapt_flops / (1e9 * batch_images):.4f}, "
                        f"total_per_image={total_flops / (1e9 * batch_images):.4f} GFLOPs"
                    )
                else:
                    # reset the model before adapting to a new batch
                    adapt_method.reset()

                    # With SAM enabled, RTA follows Algorithm 1: the current
                    # sample updates memory first, then TTA runs, then the
                    # final cache/original-logit prediction is produced.
                    with torch.no_grad():
                        if hasattr(args, 'use_sam_refinement') and args.use_sam_refinement:
                            if args.adapt:
                                adapt_method.evaluate(
                                    inputs,
                                    original_imgs=original_imgs,
                                    meta=data['meta']
                                )
                                with torch.enable_grad():
                                    loss_iter_report = adapt_method.adapt(inputs)
                                loss_batch_report.append(loss_iter_report)
                                reconstructed_preds = adapt_method.evaluate_from_memory(
                                    inputs,
                                    original_imgs=original_imgs,
                                    patch_grid_shape=patch_grid_shape,
                                    image_shapes=image_shapes
                                )
                            else:
                                reconstructed_preds = adapt_method.evaluate(
                                    inputs,
                                    original_imgs=original_imgs,
                                    meta=data['meta']
                                )
                        else:
                            if args.adapt:
                                with torch.enable_grad():
                                    loss_iter_report = adapt_method.adapt(inputs)
                                loss_batch_report.append(loss_iter_report)
                            patch_preds = adapt_method.evaluate(inputs)
                            if args.init_resize:
                                # 从 meta 中获取所需参数
                                patch_grid_shape = data['meta']['patch_grid_shape']
                                image_shapes = data['meta']['img_shape']
                                reconstructed_preds = aggregate_pred_patches(
                                    patch_preds, patch_grid_shape, image_shapes,
                                    args.patch_size, args.patch_stride
                                )
                            else:
                                reconstructed_preds = patch_preds

                # calculate the metrics
                for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):
                    pd = pd.softmax(dim=0)

                    pd = pd.argmax(dim=0)
                    pd = pd.to(gt.device)  
                    gt = gt[0]
                    results.append(intersect_and_union(pd, gt, len(org_classes), ignore_index))
                
                # 每100张图片打印一次 SAM 统计
                if hasattr(args, 'use_sam_refinement') and args.use_sam_refinement:
                    if hasattr(adapt_method, 'sam_stats') and adapt_method.sam_stats:
                        if (batch_idx + 1) % 100 == 0:
                            stats = adapt_method.sam_stats
                            print(f"\n[SAM Stats @ batch {batch_idx+1}] "
                                  f"Images: {stats['total_images']}, "
                                  f"Masks generated: {stats['total_masks_generated']}, "
                                  f"Masks kept: {stats['total_masks_kept']}, "
                                  f"Classes refined: {stats['classes_refined']}")
                            if stats['avg_score']:
                                print(f"  Avg score: {np.mean(stats['avg_score']):.4f}")
            
            # Convert the batch report to a numpy array for easier averaging
            loss_batch_report = np.array(loss_batch_report) if loss_batch_report else np.array([[0]])

            # Average loss over batches for each iteration
            avg_loss_per_iter = np.mean(loss_batch_report, axis=0)  # Shape: [10] (for 10 iterations)
            loss_seed_report.append(avg_loss_per_iter)

            metrics = process_metrics(results, org_classes)
            miou_seeds.append(metrics['mIoU'])
            dice_seeds.append(metrics['mDice'])
            acc_seeds.append(metrics['mAcc'])
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")

            # 在每个 trial 结束时打印完整的 SAM 统计
            if hasattr(args, 'use_sam_refinement') and args.use_sam_refinement:
                if hasattr(adapt_method, 'sam_stats') and adapt_method.sam_stats:
                    stats = adapt_method.sam_stats
                    print(f"\n{'='*60}")
                    print(f"[SAM Final Statistics for {corruption}, trial {t}]")
                    print(f"  - Total images processed: {stats['total_images']}")
                    print(f"  - Total classes processed: {stats['total_classes_processed']}")
                    print(f"  - Total masks generated by SAM: {stats['total_masks_generated']}")
                    print(f"  - Total masks kept after filtering: {stats['total_masks_kept']}")
                    print(f"  - Classes with SAM refinement: {stats['classes_refined']}")
                    if stats['avg_score']:
                        print(f"  - Average mask score: {np.mean(stats['avg_score']):.4f}")
                        print(f"  - Score range: [{min(stats['avg_score']):.4f}, {max(stats['avg_score']):.4f}]")
                    if stats['total_masks_generated'] > 0:
                        keep_rate = stats['total_masks_kept'] / stats['total_masks_generated'] * 100
                        print(f"  - Mask keep rate: {keep_rate:.1f}%")
                    print(f"{'='*60}\n")
                    
                    # 重置统计
                    adapt_method.sam_stats = {
                        'total_images': 0,
                        'total_classes_processed': 0,
                        'total_masks_generated': 0,
                        'total_masks_kept': 0,
                        'classes_refined': 0,
                        'avg_score': [],
                    }

            # Saving the weights if self.weights_track list is not empty
            if adapt_method.model.weights_track:
                weights_path = os.path.join(args.save_dir, "weights")
                weights = adapt_method.model.weights_track
                weights = np.hstack(weights)
                os.makedirs(weights_path, exist_ok=True)
                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))
                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()
                adapt_method.model.weights_track = []

        miou_mean, miou_std = np.array(miou_seeds).mean(), np.array(miou_seeds).std()
        dice_mean, dice_std = np.array(dice_seeds).mean(), np.array(dice_seeds).std()
        acc_mean, acc_std = np.array(acc_seeds).mean(), np.array(acc_seeds).std()

        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(c_results_path, 'w') as f:        
            f.write(headers + "\n")
            f.write(c_results_print)    

        all_results[corruption] = c_results_print

        # Convert the seed report to a numpy array and average over trials (seeds)
        loss_seed_report = np.array(loss_seed_report)
        avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)  # Shape: [10] (averaged over seeds)

        if args.plot_loss and args.adapt:
            # Plot the averaged loss for this corruption
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds)+1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            
            # Save the plot in the specified directory
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

        # if the runtime calculation is enabled, we will have access to adapt_method.adapt_times and adapt_method.eval_times (each one contains a list of times)
        if args.runtime_calculation:
            if args.adapt:
                mean_adapt_time = np.mean(adapt_method.adapt_times[20:]) if len(adapt_method.adapt_times) > 20 else 0
                std_adapt_time = np.std(adapt_method.adapt_times[20:]) if len(adapt_method.adapt_times) > 20 else 0
            else:
                mean_adapt_time = 0
                std_adapt_time = 0
            
            mean_eval_time = np.mean(adapt_method.eval_times[20:]) if len(adapt_method.eval_times) > 20 else 0
            std_eval_time = np.std(adapt_method.eval_times[20:]) if len(adapt_method.eval_times) > 20 else 0

            mean_total_time = mean_adapt_time + mean_eval_time

            run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
            print(run_time_txt)
            
            runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
            with open(runtime_save_dir, 'a+') as f:
                f.write(run_time_txt + "\n")

            adapt_time_all_corr.append(mean_adapt_time)
            eval_time_all_corr.append(mean_eval_time)

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    if args.profile_gflops and total_profiled_batches > 0:
        avg_total_per_batch = total_profiled_total_flops / total_profiled_batches
        avg_eval_per_batch = total_profiled_eval_flops / total_profiled_batches
        avg_adapt_per_batch = total_profiled_adapt_flops / total_profiled_batches
        avg_eval_per_image = total_profiled_eval_flops / max(1, total_profiled_images)

        print("\n================ GFLOPS Summary ================")
        print(f"Profiled batches: {total_profiled_batches}")
        print(f"Eval FLOPs per batch: {avg_eval_per_batch / 1e9:.4f} GFLOPs")
        print(f"Adapt FLOPs per batch: {avg_adapt_per_batch / 1e9:.4f} GFLOPs")
        print(f"Total FLOPs per batch: {avg_total_per_batch / 1e9:.4f} GFLOPs")
        print(f"Eval FLOPs per image (recommended): {avg_eval_per_image / 1e9:.4f} GFLOPs")
        print("================================================\n")

    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")




if __name__ == "__main__":
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    # Run the main function with the parsed arguments
    main(args)
