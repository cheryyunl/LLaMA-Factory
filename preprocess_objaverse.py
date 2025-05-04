#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp
import random
from collections import defaultdict
import numbers

def dynamic_axis_partition(points, scene_range, max_splits_per_axis=5, target_points=512):
    """
    Point cloud partitioning algorithm with strict constraints:
    - Maximum 5 splits per axis
    - Fixed 512 points per patch
    - Maximum 125 patches (5*5*5)
    """
    patches = []
    patch_coords = []

    # Z-axis partitioning - requires 512*4*4 points
    z_target = target_points * max_splits_per_axis * max_splits_per_axis  # 512*4*4
    z_coords = points[:, 2]
    z_splits, _ = calculate_splits(z_coords, scene_range[2], z_target, max_splits_per_axis)
    
    for z_idx in range(len(z_splits)-1):
        z_min, z_max = z_splits[z_idx], z_splits[z_idx+1]
        z_mask = (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        z_layer = points[z_mask]
        if len(z_layer) == 0:
            continue

        # Y-axis partitioning - requires 512*4 points
        y_target = target_points * max_splits_per_axis  # 512*4
        y_coords = z_layer[:, 1]
        y_splits, _ = calculate_splits(y_coords, scene_range[1], y_target, max_splits_per_axis)

        for y_idx in range(len(y_splits)-1):
            y_min, y_max = y_splits[y_idx], y_splits[y_idx+1]
            y_mask = (z_layer[:, 1] >= y_min) & (z_layer[:, 1] < y_max)
            y_row = z_layer[y_mask]
            if len(y_row) == 0:
                continue

            # X-axis partitioning - requires 512 points
            x_coords = y_row[:, 0]
            x_splits, _ = calculate_splits(x_coords, scene_range[0], target_points, max_splits_per_axis)

            for x_idx in range(len(x_splits)-1):
                x_min, x_max = x_splits[x_idx], x_splits[x_idx+1]
                x_mask = (y_row[:, 0] >= x_min) & (y_row[:, 0] < x_max)
                patch = y_row[x_mask]
                
                # Maintain exactly 512 points
                processed_patch = adjust_points(patch, target_points)
                patch_coords.append((z_idx, y_idx, x_idx))
                patches.append(processed_patch)

    return patches, patch_coords


def calculate_splits(axis_coords, axis_range, target_points_per_split, max_splits):
    """Calculate split points based on point distribution"""
    sorted_coords = np.sort(axis_coords)
    total_points = len(sorted_coords)
    
    if total_points == 0:
        return [axis_range[0], axis_range[1]], 0
        
    # Split based on point distribution regardless of total points
    # Each partition should have at least target_points_per_split points
    required_splits = min(max_splits, max(1, total_points // target_points_per_split))
    
    # Determine split points based on cumulative point count ratio
    splits = [axis_range[0]]
    points_per_split = total_points / required_splits
    
    for i in range(1, required_splits):
        target_index = int(i * points_per_split)
        splits.append(sorted_coords[target_index])
    
    splits.append(axis_range[1])
    splits = list(np.unique(splits))
    
    return splits, len(splits)-1

def adjust_points(patch, target_points):
    """Strictly align the number of points to 512"""
    if len(patch) == 0:
        return np.zeros((target_points, patch.shape[1]))
    elif len(patch) > target_points:
        return fps_sampling(patch, target_points)
    else:
        repeat_times = (target_points // len(patch)) + 1
        return np.tile(patch, (repeat_times, 1))[:target_points]

def fps_sampling(points, n_samples, num_candidates=10):
    """Improved fast FPS sampling (fixed sampling when remaining < num_candidates)"""
    if len(points) <= n_samples:
        return points
    
    indices = [np.random.randint(len(points))]
    
    for _ in range(1, n_samples):
        # 找出剩余的点索引
        remaining = np.where(~np.isin(np.arange(len(points)), indices))[0]
        if len(remaining) == 0:
            break
        
        # 如果剩余点少于num_candidates，则取所有剩余点；否则随机抽num_candidates
        k = min(len(remaining), num_candidates)
        candidates = np.random.choice(remaining, k, replace=False)
        
        # 计算每个候选点到已选点集中最小距离
        dists = np.min(
            np.linalg.norm(points[candidates][:, None] - points[indices], axis=2),
            axis=1
        )
        # 选取最远的一个
        next_idx = candidates[np.argmax(dists)]
        indices.append(next_idx)
    
    return points[indices]
   
# 配置
PROMPTS = [
    "<image>What is this object?",
    "<image>This is a point cloud. What is this object?",
    "<image>Identify this 3D object.",
    "<image>Looking at this point cloud, what object does it represent?",
    "<image>Please classify this 3D point cloud."
]

# 系统指令常量
SYSTEM_PROMPT = (
    "You are an AI assistant specialized in understanding 3D point cloud data "
    "with 6 dimensions: coordinates (x, y, z) and colors (r, g, b). "
)

def process_pointcloud_to_patches(xyz, rgb, normalize=None):
    """将点云处理为patches和coordinates"""
    # 组合xyz和rgb为一个6维数组
    arr = np.concatenate([xyz, rgb], axis=1)
    
    # 统计原始范围
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    r_min, r_max = rgb[:, 0].min(), rgb[:, 0].max()
    g_min, g_max = rgb[:, 1].min(), rgb[:, 1].max()
    b_min, b_max = rgb[:, 2].min(), rgb[:, 2].max()
    
    # 归一化处理
    if normalize == 'xyz' or normalize == 'all':
        # 归一化空间坐标到[-1, 1]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        x_scale = max(abs(x_max - x_center), abs(x_min - x_center)) * 1.05  # 留一点余量
        y_scale = max(abs(y_max - y_center), abs(y_min - y_center)) * 1.05
        z_scale = max(abs(z_max - z_center), abs(z_min - z_center)) * 1.05
        
        max_scale = max(x_scale, y_scale, z_scale)
        
        xyz[:, 0] = (xyz[:, 0] - x_center) / max_scale
        xyz[:, 1] = (xyz[:, 1] - y_center) / max_scale
        xyz[:, 2] = (xyz[:, 2] - z_center) / max_scale
        
        # 更新arr
        arr[:, :3] = xyz
        
    if normalize == 'rgb' or normalize == 'all':
        # 归一化颜色到[0, 1]
        if r_max > 1.0 or g_max > 1.0 or b_max > 1.0:  # 推测是0-255范围
            rgb = rgb / 255.0
            arr[:, 3:] = rgb
    
    # 返回坐标范围信息
    ranges = {
        'x': (x_min, x_max),
        'y': (y_min, y_max),
        'z': (z_min, z_max),
        'r': (r_min, r_max),
        'g': (g_min, g_max),
        'b': (b_min, b_max)
    }
    
    # 计算场景范围
    scene_range = [
        (xyz[:, 0].min(), xyz[:, 0].max()),
        (xyz[:, 1].min(), xyz[:, 1].max()),
        (xyz[:, 2].min(), xyz[:, 2].max())
    ]
    
    # 分割点云
    patches, patch_coords = dynamic_axis_partition(arr, scene_range)
    return patches, patch_coords, ranges

def to_python_type(obj):
    """将numpy类型转换为Python标准类型"""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, numbers.Number):
        return obj.item() if hasattr(obj, 'item') else obj
    else:
        return obj

def process_one_npy_file(args):
    """处理单个npy文件"""
    npy_path, output_dir, use_diverse_prompts, normalize = args
    
    # 生成相对路径，保持目录结构
    relpath = os.path.relpath(npy_path, start=os.path.dirname(output_dir))
    base_name = os.path.basename(npy_path).replace('.npy', '')
    
    # 创建npz保存目录 (保持原目录结构)
    npz_dir = os.path.join(output_dir, 'patches', os.path.dirname(relpath))
    os.makedirs(npz_dir, exist_ok=True)
    
    # 生成npz文件路径
    npz_path = os.path.join(npz_dir, f"{base_name}.npz")
    
    # 检查npz是否已存在，如果存在则跳过处理
    if os.path.exists(npz_path):
        return {
            'status': 'skipped',
            'npy_path': npy_path,
            'npz_path': npz_path
        }
    
    try:
        # 加载npy文件
        data = np.load(npy_path, allow_pickle=True).item()
        
        # 提取xyz, rgb和text数据
        xyz = data['xyz']
        rgb = data['rgb']
        
        # 获取label (转为小写)
        if isinstance(data['text'], list) and len(data['text']) > 0:
            label = data['text'][0].lower()
        else:
            label = str(data['text']).lower()
        
        # 处理点云为patches和patch_coords
        patches, patch_coords, ranges = process_pointcloud_to_patches(xyz, rgb, normalize=normalize)
        
        # 检查patches是否为空
        if len(patches) == 0:
            return {
                'status': 'failed',
                'npy_path': npy_path,
                'reason': 'no_patches'
            }
        
        # 保存为npz
        np.savez(npz_path, patches=patches, patch_coords=patch_coords)
        
        return {
            'status': 'success',
            'npy_path': npy_path,
            'npz_path': npz_path,
            'label': label,
            'num_patches': len(patches),
            'ranges': ranges
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'npy_path': npy_path,
            'error': str(e)
        }

def find_all_npy_files(root_dir):
    """递归查找所有npy文件"""
    npy_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

def process_objaverse_lvis(root_dir, output_dir, n_proc=None, use_diverse_prompts=True, normalize=None):
    """处理objaverse_lvis目录下的所有npy文件"""
    if n_proc is None:
        n_proc = max(1, mp.cpu_count() - 1)  # 默认使用n-1个CPU
    
    # 找到所有的npy文件
    print(f"Scanning {root_dir} for npy files...")
    all_npy_files = find_all_npy_files(root_dir)
    
    if not all_npy_files:
        print(f"No npy files found in {root_dir}")
        return 0
        
    print(f"Found {len(all_npy_files)} npy files to process")
    
    # 准备任务参数
    tasks = [(npy_file, output_dir, use_diverse_prompts, normalize) for npy_file in all_npy_files]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "objaverse_lvis.jsonl")
    
    # 多进程处理
    print(f"Processing with {n_proc} workers...")
    with mp.Pool(n_proc) as pool:
        results = list(tqdm(pool.imap(process_one_npy_file, tasks), total=len(tasks), desc="Processing npy files"))
    
    # 统计信息
    stats = {
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'error': 0,
        'total_patches': 0
    }
    
    # 范围统计
    ranges_stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
    
    # 写入jsonl文件
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for result in results:
            status = result['status']
            stats[status] += 1
            
            if status == 'success':
                # 更新统计信息
                stats['total_patches'] += result['num_patches']
                
                # 更新范围统计
                for key, (min_val, max_val) in result['ranges'].items():
                    ranges_stats[key]['min'] = min(ranges_stats[key]['min'], min_val)
                    ranges_stats[key]['max'] = max(ranges_stats[key]['max'], max_val)
                
                # 选择prompt
                prompt = random.choice(PROMPTS) if use_diverse_prompts else PROMPTS[1]
                
                # 获取npz文件的相对路径
                rel_npz_path = os.path.relpath(result['npz_path'], output_dir)
                
                # 创建LLaMA Factory格式的对话数据
                data = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": result['label']}
                    ],
                    "images": [rel_npz_path]
                }
                
                # 写入jsonl
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # 打印处理统计
    print(f"\n处理完成! 结果保存在 {jsonl_path}")
    print(f"总文件数: {len(all_npy_files)}")
    print(f"成功处理: {stats['success']}")
    print(f"已存在跳过: {stats['skipped']}")
    print(f"处理失败: {stats['failed']}")
    print(f"发生错误: {stats['error']}")
    print(f"生成patch总数: {stats['total_patches']}")
    
    if stats['success'] > 0:
        print(f"平均每个点云的patch数: {stats['total_patches'] / stats['success']:.2f}")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, "objaverse_lvis_stats.json")
    with open(stats_file, "w") as f:
        json.dump({
            "stats": stats,
            "ranges": to_python_type(dict(ranges_stats))
        }, f, indent=2)
    
    print(f"统计信息已保存到 {stats_file}")
    
    return stats['success']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Objaverse-LVIS point cloud data for LLaMA-Factory training')
    parser.add_argument('--root_dir', type=str, default="uni3d_data/objaverse_lvis", help='Root directory containing npy files')
    parser.add_argument('--output_dir', type=str, default="processed_objaverse", help='Output directory for processed data')
    parser.add_argument('--n_proc', type=int, default=None, help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--diverse_prompts', action='store_true', help='Use diverse prompts for better generalization')
    parser.add_argument('--normalize', choices=['xyz', 'rgb', 'all', None], default='all', 
                      help='Normalize coordinates and/or colors (xyz: normalize coordinates, rgb: normalize colors, all: normalize both)')
    args = parser.parse_args()
    
    # 处理数据
    process_objaverse_lvis(
        args.root_dir,
        args.output_dir,
        n_proc=args.n_proc,
        use_diverse_prompts=args.diverse_prompts,
        normalize=args.normalize
    )
    
    print("\nDone! 🎉")
    print(f"To use with LLaMA-Factory, set --dataset_dir to {args.output_dir}")