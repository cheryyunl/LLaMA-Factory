#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp

# 系统提示模板
SYSTEM_PROMPT = (
    "You are an AI assistant specialized in understanding 3D point cloud data "
    "with 6 dimensions: coordinates (x, y, z) and colors (r, g, b). "
)

# 用户提示模板
USER_PROMPT_TEMPLATE = "<image>{}"

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

def process_pointcloud_to_patches(points_6d):
    """将点云处理为patches和patch_coords"""
    # 提取xyz和rgb
    xyz = points_6d[:, :3]
    rgb = points_6d[:, 3:]
    
    # 归一化点云
    xyz_norm, rgb_norm = normalize_pointcloud(xyz.copy(), rgb.copy())
    points_6d[:, :3] = xyz_norm
    points_6d[:, 3:] = rgb_norm
    
    # 计算场景范围
    scene_range = [
        (xyz_norm[:, 0].min(), xyz_norm[:, 0].max()),
        (xyz_norm[:, 1].min(), xyz_norm[:, 1].max()),
        (xyz_norm[:, 2].min(), xyz_norm[:, 2].max())
    ]
    
    # 分割点云
    patches, patch_coords = dynamic_axis_partition(points_6d, scene_range)
    return patches, patch_coords

def load_and_process_point_cloud(args):
    """加载和处理单个点云文件"""
    npy_path, point_id, output_dir = args
    
    # 检查输出文件是否已存在
    output_npz = os.path.join(output_dir, f"{point_id}.npz")
    if os.path.exists(output_npz):
        return {
            "point_id": point_id,
            "npz_path": output_npz,
            "status": "skipped",
            "num_patches": 0
        }
    
    try:
        # 加载点云数据
        points = np.load(npy_path)
        
        # 检查形状
        if points.shape[1] != 6:
            print(f"警告: {npy_path} 的形状是 {points.shape}，期望是 [N, 6]")
            if points.shape[1] == 3:  # 只有坐标没有颜色
                # 添加默认颜色 (灰色)
                colors = np.ones((points.shape[0], 3)) * 0.5
                points = np.hstack((points, colors))
        
        # 处理点云为patches
        patches, patch_coords = process_pointcloud_to_patches(points)
        
        # 如果没有生成任何patch，返回错误
        if len(patches) == 0:
            return {
                "point_id": point_id,
                "status": "failed",
                "error": "No patches generated",
            }
        
        # 保存为NPZ
        np.savez(output_npz, patches=patches, patch_coords=patch_coords)
        
        return {
            "point_id": point_id,
            "npz_path": output_npz,
            "status": "success",
            "num_patches": len(patches)
        }
    except Exception as e:
        print(f"处理文件 {npy_path} 时出错: {str(e)}")
        return {
            "point_id": point_id,
            "status": "failed",
            "error": str(e)
        }

def process_dataset(question_path, gt_path, points_dir, output_dir, output_jsonl, n_workers=None):
    """处理整个数据集"""
    # 创建输出目录
    npz_dir = os.path.join(output_dir, "3d_mm_vet")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    # 加载问题和答案
    print("加载问题和答案...")
    questions = {}
    with open(question_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question_id = data["question_id"]
                questions[question_id] = {
                    "point": data["point"],
                    "text": data["text"],
                    "category": data["category"]
                }
            except json.JSONDecodeError:
                print(f"警告: 跳过格式错误的问题行: {line}")
    
    answers = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question_id = data["question_id"]
                answers[question_id] = {
                    "point": data["point"],
                    "text": data["text"],
                    "category": data["category"]
                }
            except json.JSONDecodeError:
                print(f"警告: 跳过格式错误的答案行: {line}")
    
    print(f"加载了 {len(questions)} 个问题和 {len(answers)} 个答案")
    
    # 收集所有需要处理的点云文件
    point_files = set()
    for qid, q_data in questions.items():
        if qid in answers:
            point_files.add(q_data["point"])
    
    print(f"需要处理 {len(point_files)} 个点云文件")
    
    # 准备点云处理任务
    tasks = []
    for point_id in point_files:
        npy_path = os.path.join(points_dir, point_id)
        # 处理文件名，如果没有.npy后缀，添加它
        if not npy_path.endswith('.npy'):
            npy_path += '.npy'
        
        # 检查文件是否存在
        if not os.path.exists(npy_path):
            print(f"警告: 点云文件不存在: {npy_path}")
            continue
            
        tasks.append((npy_path, point_id.replace('.npy', ''), npz_dir))
    
    # 设置工作进程数
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # 多进程处理点云
    print(f"使用 {n_workers} 个工作进程处理点云...")
    results = {}
    
    with mp.Pool(n_workers) as pool:
        for result in tqdm(pool.imap(load_and_process_point_cloud, tasks), total=len(tasks), desc="处理点云"):
            results[result["point_id"]] = result
    
    # 统计处理结果
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    skipped_count = sum(1 for r in results.values() if r["status"] == "skipped")
    failed_count = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"点云处理完成. 成功: {success_count}, 跳过: {skipped_count}, 失败: {failed_count}")
    
    # 生成对话数据并写入JSONL
    print(f"生成对话数据并写入 {output_jsonl}...")
    qa_count = 0
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for qid, question in questions.items():
            # 检查是否有对应的答案
            if qid not in answers:
                continue
                
            answer = answers[qid]
            point_id = question["point"].replace('.npy', '')
            
            # 检查点云是否成功处理
            if point_id not in results or results[point_id]["status"] != "success":
                continue
            
            # 构建图像路径 - 简化格式
            image_path = f"3d_mm_vet/{point_id}.npz"
            
            # 构建对话数据 - 只包含必要字段
            dialogue = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question["text"])},
                    {"role": "assistant", "content": answer["text"]}
                ],
                "images": [image_path]
                # 移除category和question_id字段
            }
            
            # 写入JSONL
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
            qa_count += 1
    
    print(f"完成! 生成了 {qa_count} 个问答对话")
    print(f"要使用该数据集, 设置 --dataset_dir 为 {os.path.dirname(output_jsonl)}")

def main():
    parser = argparse.ArgumentParser(description='处理3D-MM-Vet数据集')
    parser.add_argument('--question_path', type=str, required=True, help='问题JSONL文件路径')
    parser.add_argument('--gt_path', type=str, required=True, help='答案JSONL文件路径')
    parser.add_argument('--points_dir', type=str, required=True, help='点云目录路径')
    parser.add_argument('--output_dir', type=str, default='data', help='输出根目录')
    parser.add_argument('--output_jsonl', type=str, default='data/3d_mm_vet.jsonl', help='输出JSONL文件路径')
    parser.add_argument('--n_workers', type=int, default=None, help='工作进程数')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 处理数据集
    process_dataset(
        args.question_path,
        args.gt_path,
        args.points_dir,
        args.output_dir,
        args.output_jsonl,
        args.n_workers
    )

if __name__ == "__main__":
    main() 