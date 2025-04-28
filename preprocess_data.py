#!/usr/bin/env python3
import os
import lz4.frame
import json
import base64
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import argparse
# from pointcloud_process_utils import dynamic_axis_partition
import random
from collections import defaultdict

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

def process_pointcloud_to_patches(arr, normalize=None):
    """将点云处理为patches和coordinates"""
    # 把从 buffer 来的 arr 拷贝一份，否则不可写
    arr = arr.copy()
    xyz = arr[:, :3]
    rgb = arr[:, 3:]
    
    # 统计原始范围并打印
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    r_min, r_max = rgb[:, 0].min(), rgb[:, 0].max()
    g_min, g_max = rgb[:, 1].min(), rgb[:, 1].max()
    b_min, b_max = rgb[:, 2].min(), rgb[:, 2].max()
    
    # 归一化处理
    if normalize == 'xyz':
        # 只归一化空间坐标到[-1, 1]
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
        
    elif normalize == 'rgb':
        # 归一化颜色到[0, 1]
        if r_max > 1.0 or g_max > 1.0 or b_max > 1.0:  # 推测是0-255范围
            rgb = rgb / 255.0
            arr[:, 3:] = rgb
            
    elif normalize == 'all':
        # 归一化空间坐标和颜色
        # 空间坐标到[-1, 1]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        x_scale = max(abs(x_max - x_center), abs(x_min - x_center)) * 1.05
        y_scale = max(abs(y_max - y_center), abs(y_min - y_center)) * 1.05
        z_scale = max(abs(z_max - z_center), abs(z_min - z_center)) * 1.05
        
        max_scale = max(x_scale, y_scale, z_scale)
        
        xyz[:, 0] = (xyz[:, 0] - x_center) / max_scale
        xyz[:, 1] = (xyz[:, 1] - y_center) / max_scale
        xyz[:, 2] = (xyz[:, 2] - z_center) / max_scale
        
        # 颜色到[0, 1]
        if r_max > 1.0 or g_max > 1.0 or b_max > 1.0:  # 推测是0-255范围
            rgb = rgb / 255.0
            
        arr[:, :3] = xyz
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

def extract_text_label(obj):
    """从原始对象中提取标签文本"""
    for item in obj["content"]:
        if item.get("type") == "text":
            return item.get("text")
    return None

def analyze_pointcloud_ranges(shard_path, max_samples=100):
    """分析点云数据范围，仅用于检查"""
    all_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
    samples_processed = 0
    
    print(f"\n===== 分析点云范围: {os.path.basename(shard_path)} =====")
    with lz4.frame.open(shard_path, "rt") as fin:
        for line in tqdm(fin, desc="Analyzing ranges"):
            if samples_processed >= max_samples:
                break
                
            try:
                # 解析JSON对象
                obj = json.loads(line)
                
                # 提取点云数据
                b64 = obj["content"][0]["image_url"]["url"]
                arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(-1, 6)
                
                # 提取范围
                xyz = arr[:, :3]
                rgb = arr[:, 3:]
                
                # 更新最小/最大值
                all_ranges['x']['min'] = min(all_ranges['x']['min'], xyz[:, 0].min())
                all_ranges['x']['max'] = max(all_ranges['x']['max'], xyz[:, 0].max())
                
                all_ranges['y']['min'] = min(all_ranges['y']['min'], xyz[:, 1].min())
                all_ranges['y']['max'] = max(all_ranges['y']['max'], xyz[:, 1].max())
                
                all_ranges['z']['min'] = min(all_ranges['z']['min'], xyz[:, 2].min())
                all_ranges['z']['max'] = max(all_ranges['z']['max'], xyz[:, 2].max())
                
                all_ranges['r']['min'] = min(all_ranges['r']['min'], rgb[:, 0].min())
                all_ranges['r']['max'] = max(all_ranges['r']['max'], rgb[:, 0].max())
                
                all_ranges['g']['min'] = min(all_ranges['g']['min'], rgb[:, 1].min())
                all_ranges['g']['max'] = max(all_ranges['g']['max'], rgb[:, 1].max())
                
                all_ranges['b']['min'] = min(all_ranges['b']['min'], rgb[:, 2].min())
                all_ranges['b']['max'] = max(all_ranges['b']['max'], rgb[:, 2].max())
                
                samples_processed += 1
                
            except Exception as e:
                print(f"Error analyzing sample: {e}")
                continue
    
    # 打印范围统计
    print("\n点云坐标和颜色范围统计:")
    print(f"{'坐标/颜色':<10} {'最小值':<15} {'最大值':<15} {'范围':<15}")
    print("-" * 55)
    
    for key in ['x', 'y', 'z', 'r', 'g', 'b']:
        min_val = all_ranges[key]['min']
        max_val = all_ranges[key]['max']
        range_val = max_val - min_val
        print(f"{key:<10} {min_val:<15.6f} {max_val:<15.6f} {range_val:<15.6f}")
    
    # 建议
    print("\n归一化建议:")
    if all_ranges['r']['max'] > 1.0 or all_ranges['g']['max'] > 1.0 or all_ranges['b']['max'] > 1.0:
        print("- RGB值超过1.0，可能是0-255范围，建议归一化到[0,1]")
    else:
        print("- RGB值已在[0,1]范围内，无需归一化")
        
    if abs(all_ranges['x']['max']) > 10 or abs(all_ranges['y']['max']) > 10 or abs(all_ranges['z']['max']) > 10:
        print("- XYZ坐标值较大，建议归一化到[-1,1]范围")
    else:
        print("- XYZ坐标值范围适中，可选择是否归一化")
    
    return all_ranges

def process_one_shard(args):
    """处理单个shard文件"""
    shard_path, output_dir, use_diverse_prompts, json_index, normalize = args
    
    # 获取数据集名称和shard名称
    dataset_name = os.path.basename(os.path.dirname(shard_path))
    shard_name = os.path.basename(shard_path).replace('.jsonl.lz4', '')
    
    # 创建npz保存目录
    npz_dir = os.path.join(output_dir, dataset_name, f"patches_{shard_name}")
    os.makedirs(npz_dir, exist_ok=True)
    
    # 创建jsonl输出路径 - 根据数据集名称和JSON索引命名
    out_jsonl = os.path.join(output_dir, f"{dataset_name}_{json_index:03d}.jsonl")
    
    # 统计变量
    stats = {
        'pointclouds_processed': 0,
        'total_patches': 0,
        'min_patches_per_cloud': float('inf'),
        'max_patches_per_cloud': 0,
        'failed_samples': 0
    }
    
    all_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
    
    with lz4.frame.open(shard_path, "rt") as fin, open(out_jsonl, "w") as fout:
        for line in tqdm(fin, desc=f"Processing {dataset_name}/{shard_name}"):
            try:
                # 解析JSON对象
                obj = json.loads(line)
                
                # 提取点云数据
                b64 = obj["content"][0]["image_url"]["url"]
                arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(-1, 6)
                
                # 提取标签
                label = extract_text_label(obj)
                if not label:
                    stats['failed_samples'] += 1
                    continue  # 跳过没有标签的样本
                
                # 处理点云为patches，并获取坐标范围
                patches, patch_coords, ranges = process_pointcloud_to_patches(arr, normalize=normalize)
                
                # 更新统计信息
                num_patches = len(patches)
                stats['pointclouds_processed'] += 1
                stats['total_patches'] += num_patches
                stats['min_patches_per_cloud'] = min(stats['min_patches_per_cloud'], num_patches)
                stats['max_patches_per_cloud'] = max(stats['max_patches_per_cloud'], num_patches)
                
                # 更新范围统计
                for key, (min_val, max_val) in ranges.items():
                    all_ranges[key]['min'] = min(all_ranges[key]['min'], min_val)
                    all_ranges[key]['max'] = max(all_ranges[key]['max'], max_val)
                
                # 保存为npz
                obj_id = f"{dataset_name}_{stats['pointclouds_processed']:06d}"
                npz_path = os.path.join(npz_dir, f"{obj_id}.npz")
                np.savez(npz_path, patches=patches, patch_coords=patch_coords)
                
                # 选择prompt
                prompt = random.choice(PROMPTS) if use_diverse_prompts else PROMPTS[1]
                
                # 创建LLaMA Factory格式的对话数据
                data = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": label}
                    ],
                    "images": [os.path.relpath(npz_path, os.path.dirname(out_jsonl))]  # 使用相对路径
                }
                
                # 写入jsonl
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"Error processing sample in {shard_path}: {e}")
                continue
    
    # 计算平均值
    if stats['pointclouds_processed'] > 0:
        stats['avg_patches_per_cloud'] = stats['total_patches'] / stats['pointclouds_processed']
    else:
        stats['avg_patches_per_cloud'] = 0
        stats['min_patches_per_cloud'] = 0
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, f"{dataset_name}_{shard_name}_stats.json")
    with open(stats_file, "w") as f:
        json.dump({"ranges": all_ranges, "stats": stats}, f, indent=2)
        
    # 返回处理结果、范围统计和处理统计
    result_msg = f"Processed {stats['pointclouds_processed']} pointclouds with {stats['total_patches']} patches from {shard_path}"
    return (result_msg, all_ranges, stats)

def find_all_shards(root_dir, dataset_filter=None):
    """查找所有的shard文件，可选择只返回特定数据集的shard"""
    all_shards = []
    for dataset in os.listdir(root_dir):
        # 如果指定了数据集过滤，只处理匹配的数据集
        if dataset_filter and dataset.lower() != dataset_filter.lower():
            continue
            
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        
        for fname in os.listdir(dataset_path):
            if fname.endswith(".jsonl.lz4") and fname.startswith("objects.shard_"):
                shard_path = os.path.join(dataset_path, fname)
                all_shards.append(shard_path)
    
    return all_shards

def process_all_datasets(root_dir, output_dir, n_proc=None, use_diverse_prompts=True, dataset_filter=None, analyze_only=False, normalize=None):
    """处理所有数据集的所有shard"""
    if n_proc is None:
        n_proc = max(1, mp.cpu_count() - 1)  # 默认使用n-1个CPU
    
    # 找到所有的shard文件
    print(f"Scanning {root_dir} for shard files...")
    all_shards = find_all_shards(root_dir, dataset_filter)
    
    if not all_shards:
        if dataset_filter:
            print(f"No shard files found for dataset '{dataset_filter}' in {root_dir}")
        else:
            print(f"No shard files found in {root_dir}")
        return 0
        
    print(f"Found {len(all_shards)} shard files to process")
    
    # 如果只分析范围，不处理
    if analyze_only:
        print("Running in analysis-only mode")
        for shard in all_shards:
            analyze_pointcloud_ranges(shard)
        return 0
    
    # 准备任务参数，添加json索引和归一化选项
    tasks = [(shard, output_dir, use_diverse_prompts, idx, normalize) for idx, shard in enumerate(all_shards)]
    
    # 多进程处理
    print(f"Processing with {n_proc} workers...")
    with mp.Pool(n_proc) as pool:
        results = list(tqdm(pool.imap(process_one_shard, tasks), total=len(tasks), desc="Overall Progress"))
    
    # 打印结果和汇总统计
    dataset_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
    total_stats = {
        'pointclouds_processed': 0,
        'total_patches': 0,
        'failed_samples': 0
    }
    
    # 按数据集整理统计
    dataset_stats = defaultdict(lambda: {
        'pointclouds_processed': 0,
        'total_patches': 0,
        'failed_samples': 0,
        'shards': 0
    })
    
    print("\n处理结果:")
    for result_msg, ranges, stats in results:
        print(result_msg)
        
        # 更新总体统计
        total_stats['pointclouds_processed'] += stats['pointclouds_processed']
        total_stats['total_patches'] += stats['total_patches']
        total_stats['failed_samples'] += stats['failed_samples']
        
        # 更新数据集统计
        dataset_name = result_msg.split(' ')[-1].split('/')[-2]  # 从结果消息提取数据集名称
        dataset_stats[dataset_name]['pointclouds_processed'] += stats['pointclouds_processed']
        dataset_stats[dataset_name]['total_patches'] += stats['total_patches']
        dataset_stats[dataset_name]['failed_samples'] += stats['failed_samples']
        dataset_stats[dataset_name]['shards'] += 1
        
        # 更新总体范围统计
        for key, range_info in ranges.items():
            dataset_ranges[key]['min'] = min(dataset_ranges[key]['min'], range_info['min'])
            dataset_ranges[key]['max'] = max(dataset_ranges[key]['max'], range_info['max'])
    
    # 打印数据集统计
    print("\n数据集统计:")
    print(f"{'数据集':<15} {'处理点云数':<15} {'生成patch数':<15} {'平均patch/点云':<20} {'失败样本':<15}")
    print("-" * 80)
    
    for dataset, stats in dataset_stats.items():
        avg_patches = stats['total_patches'] / stats['pointclouds_processed'] if stats['pointclouds_processed'] > 0 else 0
        print(f"{dataset:<15} {stats['pointclouds_processed']:<15} {stats['total_patches']:<15} {avg_patches:<20.2f} {stats['failed_samples']:<15}")
    
    # 打印总体统计
    print("\n总体统计:")
    total_avg_patches = total_stats['total_patches'] / total_stats['pointclouds_processed'] if total_stats['pointclouds_processed'] > 0 else 0
    print(f"总处理点云数: {total_stats['pointclouds_processed']}")
    print(f"总生成patch数: {total_stats['total_patches']}")
    print(f"平均每个点云的patch数: {total_avg_patches:.2f}")
    print(f"总失败样本数: {total_stats['failed_samples']}")
    
    # 打印总体范围统计
    # ... 之前的范围统计代码不变 ...
    
    return len(all_shards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process 3D point cloud data for LLaMA-Factory training')
    parser.add_argument('--root_dir', type=str, default="3D-data", help='Root directory containing subdirectories for each dataset')
    parser.add_argument('--output_dir', type=str, default="processed_data", help='Output directory for processed data')
    parser.add_argument('--n_proc', type=int, default=None, help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--dataset', type=str, default=None, help='Only process specific dataset (e.g., "scannet")')
    parser.add_argument('--diverse_prompts', action='store_true', help='Use diverse prompts for better generalization')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze ranges without processing')
    parser.add_argument('--normalize', choices=['xyz', 'rgb', 'all', None], default=None, 
                      help='Normalize coordinates and/or colors (xyz: normalize coordinates, rgb: normalize colors, all: normalize both)')
    args = parser.parse_args()
    
    # 创建输出目录
    if not args.analyze_only:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理数据集
    num_shards = process_all_datasets(
        args.root_dir, 
        args.output_dir, 
        n_proc=args.n_proc,
        use_diverse_prompts=args.diverse_prompts,
        dataset_filter=args.dataset,  # 添加数据集过滤
        analyze_only=args.analyze_only,
        normalize=args.normalize
    )
    
    if num_shards > 0 and not args.analyze_only:
        print("\nDone! 🎉")
        print(f"To use with LLaMA-Factory, set --dataset_dir to {args.output_dir}")

