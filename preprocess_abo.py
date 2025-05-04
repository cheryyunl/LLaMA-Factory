#!/usr/bin/env python3
import os
import json
import csv
import random
import numpy as np
import argparse
from plyfile import PlyData
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

# 系统提示模板
SYSTEM_PROMPT = (
    "You are an AI assistant specialized in understanding 3D point cloud data "
    "with 6 dimensions: coordinates (x, y, z) and colors (r, g, b). "
)

# 用户提示模板
USER_PROMPTS = [
    "<image>Describe this object shown in the point cloud.",
    "<image>What do you see in this 3D point cloud?",
    "<image>Can you describe this 3D object in detail?",
    "<image>Please provide a description of this 3D object."
]

def dynamic_axis_partition(points, scene_range, max_splits_per_axis=5, target_points=512):
    """
    点云分区算法，具有严格约束：
    - 每个轴最多5个分割
    - 每个patch固定512个点
    - 最多125个patches (5*5*5)
    """
    patches = []
    patch_coords = []

    # Z轴分区 - 需要512*5*5点
    z_target = target_points * max_splits_per_axis * max_splits_per_axis
    z_coords = points[:, 2]
    z_splits, _ = calculate_splits(z_coords, scene_range[2], z_target, max_splits_per_axis)
    
    for z_idx in range(len(z_splits)-1):
        z_min, z_max = z_splits[z_idx], z_splits[z_idx+1]
        z_mask = (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        z_layer = points[z_mask]
        if len(z_layer) == 0:
            continue

        # Y轴分区 - 需要512*5点
        y_target = target_points * max_splits_per_axis
        y_coords = z_layer[:, 1]
        y_splits, _ = calculate_splits(y_coords, scene_range[1], y_target, max_splits_per_axis)

        for y_idx in range(len(y_splits)-1):
            y_min, y_max = y_splits[y_idx], y_splits[y_idx+1]
            y_mask = (z_layer[:, 1] >= y_min) & (z_layer[:, 1] < y_max)
            y_row = z_layer[y_mask]
            if len(y_row) == 0:
                continue

            # X轴分区 - 需要512点
            x_coords = y_row[:, 0]
            x_splits, _ = calculate_splits(x_coords, scene_range[0], target_points, max_splits_per_axis)

            for x_idx in range(len(x_splits)-1):
                x_min, x_max = x_splits[x_idx], x_splits[x_idx+1]
                x_mask = (y_row[:, 0] >= x_min) & (y_row[:, 0] < x_max)
                patch = y_row[x_mask]
                
                # 确保每个patch恰好有512个点
                processed_patch = adjust_points(patch, target_points)
                patch_coords.append((z_idx, y_idx, x_idx))
                patches.append(processed_patch)

    return patches, patch_coords

def calculate_splits(axis_coords, axis_range, target_points_per_split, max_splits):
    """基于点分布计算分割点"""
    sorted_coords = np.sort(axis_coords)
    total_points = len(sorted_coords)
    
    if total_points == 0:
        return [axis_range[0], axis_range[1]], 0
        
    # 根据点分布分割，每个分区至少应有target_points_per_split个点
    required_splits = min(max_splits, max(1, total_points // target_points_per_split))
    
    # 基于累积点数比例确定分割点
    splits = [axis_range[0]]
    points_per_split = total_points / required_splits
    
    for i in range(1, required_splits):
        target_index = int(i * points_per_split)
        splits.append(sorted_coords[target_index])
    
    splits.append(axis_range[1])
    splits = list(np.unique(splits))
    
    return splits, len(splits)-1

def adjust_points(patch, target_points):
    """严格将点数调整为512个"""
    if len(patch) == 0:
        return np.zeros((target_points, patch.shape[1]))
    elif len(patch) > target_points:
        return fps_sampling(patch, target_points)
    else:
        repeat_times = (target_points // len(patch)) + 1
        return np.tile(patch, (repeat_times, 1))[:target_points]

def fps_sampling(points, n_samples, num_candidates=10):
    """改进的快速FPS采样"""
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

def normalize_pointcloud(xyz, rgb):
    """归一化点云数据"""
    # 归一化空间坐标到[-1, 1]
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    x_scale = max(abs(x_max - x_center), abs(x_min - x_center)) * 1.05  # 留5%余量
    y_scale = max(abs(y_max - y_center), abs(y_min - y_center)) * 1.05
    z_scale = max(abs(z_max - z_center), abs(z_min - z_center)) * 1.05
    
    max_scale = max(x_scale, y_scale, z_scale)
    if max_scale > 0:  # 防止除以零
        xyz[:, 0] = (xyz[:, 0] - x_center) / max_scale
        xyz[:, 1] = (xyz[:, 1] - y_center) / max_scale
        xyz[:, 2] = (xyz[:, 2] - z_center) / max_scale
    
    # 归一化颜色到[0, 1]
    r_max = rgb[:, 0].max()
    g_max = rgb[:, 1].max()
    b_max = rgb[:, 2].max()
    
    if r_max > 1.0 or g_max > 1.0 or b_max > 1.0:  # 推测是0-255范围
        rgb = rgb / 255.0
    
    return xyz, rgb

def extract_vertex_data(vertex):
    """从vertex中安全地提取坐标和颜色"""
    try:
        # 尝试直接使用索引获取坐标
        if hasattr(vertex, 'dtype') and hasattr(vertex.dtype, 'names'):
            # 使用正常的数组索引方式
            x = vertex['x']
            y = vertex['y']
            z = vertex['z']
            
            # 检查是否有颜色信息
            has_color = 'red' in vertex.dtype.names and 'green' in vertex.dtype.names and 'blue' in vertex.dtype.names
            if has_color:
                r = vertex['red']
                g = vertex['green']
                b = vertex['blue']
            else:
                # 如果没有颜色，使用默认灰色
                r = np.ones_like(x) * 128
                g = np.ones_like(x) * 128
                b = np.ones_like(x) * 128
        else:
            # 尝试从更底层的数据结构提取信息
            data = vertex.data
            x = np.array([pt[0] for pt in data])
            y = np.array([pt[1] for pt in data])
            z = np.array([pt[2] for pt in data])
            
            # 检查是否有颜色信息 (典型的PLY格式包含6-7个字段: x,y,z,r,g,b,[a])
            if len(data[0]) >= 6:
                r = np.array([pt[3] for pt in data])
                g = np.array([pt[4] for pt in data])
                b = np.array([pt[5] for pt in data])
            else:
                # 如果没有颜色，使用默认灰色
                r = np.ones_like(x) * 128
                g = np.ones_like(x) * 128
                b = np.ones_like(x) * 128
                
        # 堆叠成坐标和颜色数组
        xyz = np.vstack((x, y, z)).T
        rgb = np.vstack((r, g, b)).T
        
        return xyz, rgb, True
    except Exception as e:
        print(f"提取顶点数据时出错: {str(e)}")
        return None, None, False

def load_ply_file(ply_path, target_points=16384):
    """加载PLY文件并处理为6维点云数据"""
    try:
        # 读取PLY文件
        plydata = PlyData.read(ply_path)
        
        # 检查vertex元素
        if 'vertex' not in plydata:
            print(f"错误: PLY文件 {ply_path} 中没有vertex元素")
            return None, False
            
        vertex = plydata['vertex']
        
        # 提取坐标和颜色
        xyz, rgb, success = extract_vertex_data(vertex)
        if not success:
            return None, False
        
        # 将xyz和rgb组合为6维
        points_6d = np.concatenate([xyz, rgb], axis=1)
        
        # 如果点数少于目标点数，通过重复填充；如果多于目标点数，通过采样减少
        if len(points_6d) < target_points:
            repeat_count = (target_points // len(points_6d)) + 1
            points_6d = np.tile(points_6d, (repeat_count, 1))[:target_points]
        elif len(points_6d) > target_points:
            # 使用FPS采样减少点数
            points_6d = fps_sampling(points_6d, target_points)
        
        # 归一化点云
        xyz_norm, rgb_norm = normalize_pointcloud(points_6d[:, :3].copy(), points_6d[:, 3:].copy())
        points_6d[:, :3] = xyz_norm
        points_6d[:, 3:] = rgb_norm
        
        return points_6d, True
    except Exception as e:
        print(f"处理文件 {ply_path} 时出错: {str(e)}")
        return None, False

def process_pointcloud_to_patches(points_6d):
    """将点云处理为patches和patch_coords"""
    # 提取xyz和rgb
    xyz = points_6d[:, :3]
    
    # 计算场景范围
    scene_range = [
        (xyz[:, 0].min(), xyz[:, 0].max()),
        (xyz[:, 1].min(), xyz[:, 1].max()),
        (xyz[:, 2].min(), xyz[:, 2].max())
    ]
    
    # 分割点云
    patches, patch_coords = dynamic_axis_partition(points_6d, scene_range)
    return patches, patch_coords

def load_captions_from_csv(csv_path):
    """从CSV加载物体描述"""
    captions = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                object_id = row[0]
                caption_type = row[1]
                caption_text = row[2]
                
                # 只使用general类型的标题，或者如果没有类型信息，使用所有标题
                if caption_type == "general" or len(row) == 2:
                    captions[object_id] = caption_text.strip()
    return captions

def process_single_file(args):
    """处理单个PLY文件"""
    ply_path, npz_dir, object_id, caption, dataset_name = args
    
    # 创建输出目录
    os.makedirs(npz_dir, exist_ok=True)
    
    # 生成输出文件路径
    npz_path = os.path.join(npz_dir, f"{object_id}.npz")
    
    # 如果NPZ文件已存在，则跳过处理
    if os.path.exists(npz_path):
        return {
            "object_id": object_id,
            "status": "skipped",
            "npz_path": npz_path,
            "caption": caption,
            "dataset_name": dataset_name
        }
    
    # 加载PLY文件
    points_6d, success = load_ply_file(ply_path)
    if not success or points_6d is None:
        return {
            "object_id": object_id,
            "status": "failed",
            "error": "Failed to load PLY file"
        }
    
    # 处理点云为patches
    patches, patch_coords = process_pointcloud_to_patches(points_6d)
    
    # 检查是否成功生成patches
    if len(patches) == 0:
        return {
            "object_id": object_id,
            "status": "failed",
            "error": "No patches generated"
        }
    
    # 保存为NPZ
    np.savez(npz_path, patches=patches, patch_coords=patch_coords)
    
    return {
        "object_id": object_id,
        "status": "success",
        "npz_path": npz_path,
        "num_patches": len(patches),
        "caption": caption,
        "dataset_name": dataset_name
    }

def main():
    parser = argparse.ArgumentParser(description='处理ABO点云数据集')
    parser.add_argument('--ply_dir', type=str, required=True, help='PLY文件目录')
    parser.add_argument('--csv_file', type=str, required=True, help='包含描述的CSV文件')
    parser.add_argument('--output_dir', type=str, default='data', help='输出根目录')
    parser.add_argument('--dataset_name', type=str, default='ABO', help='数据集名称，用于路径前缀')
    parser.add_argument('--output_jsonl', type=str, default=None, help='输出JSONL文件')
    parser.add_argument('--n_workers', type=int, default=None, help='工作进程数')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    args = parser.parse_args()
    
    # 设置默认的JSONL文件路径
    if args.output_jsonl is None:
        args.output_jsonl = os.path.join(args.output_dir, f"{args.dataset_name.lower()}_dataset.jsonl")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 定义NPZ文件保存目录
    npz_dir = os.path.join(args.output_dir, args.dataset_name)
    
    # 创建输出目录
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    
    # 加载描述
    print("加载描述...")
    captions = load_captions_from_csv(args.csv_file)
    print(f"加载了 {len(captions)} 个描述")
    
    # 查找所有PLY文件
    ply_files = []
    for file in os.listdir(args.ply_dir):
        if file.endswith('.ply'):
            object_id = os.path.splitext(file)[0]
            if object_id in captions:
                ply_files.append((
                    os.path.join(args.ply_dir, file),
                    npz_dir,
                    object_id,
                    captions[object_id],
                    args.dataset_name
                ))
    
    print(f"找到 {len(ply_files)} 个PLY文件进行处理")
    
    # 设置工作进程数
    if args.n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    else:
        n_workers = args.n_workers
    
    # 多进程处理
    print(f"使用 {n_workers} 个工作进程处理数据...")
    results = []
    with mp.Pool(n_workers) as pool:
        for result in tqdm(pool.imap(process_single_file, ply_files), total=len(ply_files)):
            results.append(result)
    
    # 统计处理结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"处理完成. 成功: {success_count}, 跳过: {skipped_count}, 失败: {failed_count}")
    
    # 创建JSONL文件
    print(f"生成JSONL文件: {args.output_jsonl}")
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            if result['status'] in ('success', 'skipped'):
                # 随机选择用户提示
                user_prompt = random.choice(USER_PROMPTS)
                # 获取助手回答
                assistant_answer = result['caption']
                
                # 构建图像路径，格式为"ABO/B073P13P8T.npz"
                dataset_name = result['dataset_name']
                object_id = result['object_id']
                image_path = f"{dataset_name}/{object_id}.npz"
                
                # 创建对话数据
                dialogue = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_answer}
                    ],
                    "images": [image_path]
                }
                
                # 写入JSONL
                f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
    
    print(f"完成! JSONL文件已生成: {args.output_jsonl}")
    print(f"要使用该数据集, 设置 --dataset_dir 为 {os.path.dirname(args.output_jsonl)}")

if __name__ == "__main__":
    main()