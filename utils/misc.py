import os
import yaml
import torch
import random
import subprocess

import numpy as np
from datetime import datetime


def set_global_seeds(seed_value=42):
    """设置随机种子以确保各个库的可复现性。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_git_commit_hash():
    """获取当前的 Git commit hash。"""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        return commit_hash
    except Exception as e:
        print(f"获取 Git commit hash 时出错: {e}")
        return "Unknown"

def save_configuration(args, config_file="configurations.txt", cmd_file="cmd.sh"):
    """
    保存配置参数、Git commit hash 和当前日期到文件。
    同时保存运行脚本的命令行。
    """
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取 Git commit hash
    commit_hash = get_git_commit_hash()
    
    # 获取当前日期和时间
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存配置到文件
    config_filepath = os.path.join(args.save_dir, config_file)
    print("\n配置信息 +++++++++++++++++++++++++")
    print("----------------------------------------")

    with open(config_filepath, 'w') as config_f:
        # 保存日期
        config_f.write(f"date: {current_date}\n")
        print(f"       date: {current_date}")
        
        # 保存 Git commit hash
        config_f.write(f"git_commit_hash: {commit_hash}\n")
        print(f"       git_commit_hash: {commit_hash}")
        
        # 保存其他配置
        for arg in vars(args):
            value = getattr(args, arg)
            config_f.write(f"{arg}: {value}\n")
            print(f"       {arg}: {value}")

    # 保存命令到文件
    cmd_filepath = os.path.join(args.save_dir, cmd_file)
    arg_dict = vars(args)
    cmd = "python main.py"
    for key, value in arg_dict.items():
        formatted_key = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd += f" {formatted_key}"
        elif isinstance(value, list) or isinstance(value, tuple):
            cmd += f" {formatted_key} {' '.join(map(str, value))}"
        elif value is not None:
            cmd += f" {formatted_key} {value}"

    with open(cmd_filepath, "w") as cmd_f:
        cmd_f.write(cmd + "\n")
    print(f"配置和命令已保存!")
    print("----------------------------------------")


def load_prompts_from_yaml(file_path='prompts.yaml'):
    """从 YAML 文件加载提示模板。"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['prompt_templates'] 


def save_checkpoint(state, is_best, args):
    """保存模型检查点。"""
    torch.save(state, args.save + args.dataset + '_' + args.model + '.pth')
    if is_best:
        torch.save(state, args.save + args.dataset + '_' + args.model + '_torch_best.pth')


def print_clip_parameters(model):
    """
    打印 CLIP 模型中每个模块的总参数和可学习参数 (requires_grad=True)，
    以及整体摘要。

    Args:
        model (torch.nn.Module): PyTorch 模型。
    """
    modules = {
        "model.visual": model.visual,
        "model.transformer": model.transformer,
        "model.ln_final": model.ln_final,
        "model.token_embedding": model.token_embedding
    }
    
    print("\n模型参数摘要 +++++++++++++++++++++++++")
    
    total_params = 0
    learnable_params = 0

    # 打印每个模块的参数
    for name, module in modules.items():
        module_total = sum(p.numel() for p in module.parameters())
        module_learnable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += module_total
        learnable_params += module_learnable
        print(f"{name:25}: 总计 = {module_total:,}, 可学习 = {module_learnable:,}")

    # 打印整体摘要
    print("----------------------------------------")
    print(f"总参数量      : {total_params:,}")
    print(f"可学习参数量  : {learnable_params:,}")
    print("----------------------------------------")


def print_optimizer_parameters(optimizer, model):
    """
    打印传递给优化器的总参数和可学习参数，
    按 CLIP 模型的每个模块分组。

    Args:
        optimizer (torch.optim.Optimizer): 优化器实例。
        model (torch.nn.Module): PyTorch 模型。
    """
    # 定义要分析的模块
    modules = {
        "model.visual": model.visual,
        "model.transformer": model.transformer,
        "model.ln_final": model.ln_final,
        "model.token_embedding": model.token_embedding
    }
    
    # 收集优化器中的参数
    optimizer_params = {id(p): p for group in optimizer.param_groups for p in group['params']}
    
    print("\n优化器参数（按模块）++++++++++++++++++++++")
    total_optimizer_params = 0

    # 统计每个模块的参数
    for name, module in modules.items():
        module_params = sum(
            p.numel() for p in module.parameters() if id(p) in optimizer_params
        )
        total_optimizer_params += module_params
        print(f"{name:25}: 优化器中的参数 = {module_params:,}")

    # 打印总计
    print("---------------------------------------------")
    print(f"优化器中的总参数量 : {total_optimizer_params:,}")


def get_cls_idx(path):
    """
    从文件中读取类别名称和索引映射。
    
    Args:
        path: 类别扩展文件路径
        
    Returns:
        class_names: 类别名称列表
        class_indices: 对应的类别索引列表
    """
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = list(), list()
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices


def custom_collate(data):
    """
    自定义的 DataLoader collate 函数，用于处理可变尺寸的数据。
    
    Args:
        data: DataLoader 返回的数据列表
        
    Returns:
        包含批处理数据的字典
    """
    # 从字典中提取每个键
    # 由于图像尺寸可能不同，所以不能直接堆叠
    imgs_list = [item['img'] for item in data]
    gt_list = [item['gt_seg_map'] for item in data]
    
    # 对于 patches 和 gt_seg_map_patches，在 axis=0 上堆叠
    img_patches_list = [item['patches'] for item in data]
    gt_patches_list = [item['gt_seg_map_patches'] for item in data]
    
    # 沿新的 batch 轴堆叠 patches 列表
    img_patches_batch = torch.cat(img_patches_list, axis=0)
    gt_patches_batch = torch.cat(gt_patches_list, axis=0)
    
    # 提取原始图像（用于 SAM）
    original_imgs_list = [item.get('original_img', None) for item in data]
    
    # 收集元信息
    meta = {
        'img_path': [item['meta']['img_path'] for item in data],
        'seg_map_path': [item['meta']['seg_map_path'] for item in data],
        'ori_shape': [item['meta']['ori_shape'] for item in data],
        'img_shape': [item['meta']['img_shape'] for item in data],
        'patch_shape': [item['meta']['patch_shape'] for item in data],
        'scale_factor': [item['meta']['scale_factor'] for item in data],
        'patch_grid_shape': [item['meta']['patch_grid_shape'] for item in data],
    }

    # 返回一个字典
    return {
        'img': imgs_list,
        'gt': gt_list,
        'img_patches': img_patches_batch,
        'gt_patches': gt_patches_batch,
        'original_imgs': original_imgs_list,  # 新增：SAM 需要的原始图像
        'meta': meta
    }


def aggregate_pred_patches(pred, patch_grid_shapes, img_shapes, patch_size=(224, 224), patch_stride=112):
    """
    将每个类别的分数 patches 聚合回原始图像尺寸。
    
    Args:
        pred (torch.Tensor): 预测结果，形状为 (batch_size * num_patches, num_classes, 224, 224)。
        patch_grid_shapes (list of tuple): 每张图像的网格维度列表 (num_patches_y, num_patches_x)。
        img_shapes (list of tuple): 每张图像的原始尺寸列表 (height, width)。
        patch_size (tuple): 每个 patch 的尺寸。
        patch_stride (int): patch 之间的步长（用于处理重叠区域）。
        
    Returns:
        list of torch.Tensor: 重建后的预测列表，每个形状为 (num_classes, img_height, img_width)。
    """
    batch_size = len(patch_grid_shapes)
    num_classes = pred.shape[1]  # 分割任务中的类别数
    reconstructed_preds = []
    
    patch_idx = 0  # 跟踪批次中的全局 patch 索引
    
    for b in range(batch_size):
        num_patches_y, num_patches_x = patch_grid_shapes[b]
        img_height, img_width = img_shapes[b]
        
        # 初始化重建的每类分数图像和重叠计数张量
        reconstructed_pred = torch.zeros((num_classes, img_height, img_width), device=pred.device)
        overlap_count = torch.zeros((1, img_height, img_width), device=pred.device)
        
        # 根据位置放置 patches
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # 计算 patch 放置坐标
                start_y = i * patch_stride
                start_x = j * patch_stride
                end_y = start_y + patch_size[0]
                end_x = start_x + patch_size[1]
                
                # 将 patch 的每类分数添加到重建图像中
                reconstructed_pred[:, start_y:end_y, start_x:end_x] += pred[patch_idx]
                overlap_count[:, start_y:end_y, start_x:end_x] += 1
                patch_idx += 1
        
        # 对每个类别的重叠区域取平均
        reconstructed_pred /= overlap_count
        reconstructed_preds.append(reconstructed_pred)
    
    return reconstructed_preds  # 返回每张图像的每类分数张量列表
