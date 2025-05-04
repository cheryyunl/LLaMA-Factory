#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp
import re
from plyfile import PlyData

# 系统提示模板
SYSTEM_PROMPT = (
    "You are an AI assistant specialized in understanding 3D point cloud data "
    "with 6 dimensions: coordinates (x, y, z) and colors (r, g, b). Objects in 3D scenes "
    "are identified as <obj_0>, <obj_1>, etc. Please answer questions based on 3D scene point cloud data and object grounding contexts. "
    "When referring to objects, directly include their IDs in your response.\n"

)

# 正则表达式，用于处理文本
P_TAG_PATTERN = re.compile(r'<p>([^<]+)</p>')
OBJ_ID_PATTERN = re.compile(r'\[<([^>]+)>\]')

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

def clean_text(text):
    """清理文本，移除<p>标签和[]"""
    # 移除<p>标签
    text = re.sub(P_TAG_PATTERN, r'\1', text)
    # 将[<object-id>]格式转换为<object-id>
    text = re.sub(OBJ_ID_PATTERN, r'<\1>', text)
    return text

def load_and_process_ply(args):
    """加载和处理单个PLY文件"""
    ply_path, scene_id, output_dir = args
    
    # 从scene_id中提取目录和文件名
    if '@' in scene_id:
        folder_id, scene_name = scene_id.split('@')
    else:
        # 如果没有@符号，直接使用文件名
        scene_name = os.path.basename(ply_path).replace('.ply', '')
    
    # 检查输出文件是否已存在
    output_npz = os.path.join(output_dir, f"{scene_name}.npz")
    if os.path.exists(output_npz):
        return {
            "scene_id": scene_id,
            "npz_path": output_npz,
            "status": "skipped"
        }
    
    try:
        # 读取PLY文件
        plydata = PlyData.read(ply_path)
        
        # 提取顶点数据
        vertex = plydata['vertex']
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        
        # 提取颜色数据（如果存在）
        if 'red' in vertex.dtype.names:
            r = vertex['red']
            g = vertex['green']
            b = vertex['blue']
        else:
            # 如果没有颜色，使用默认灰色
            r = np.ones_like(x) * 128
            g = np.ones_like(x) * 128
            b = np.ones_like(x) * 128
        
        # 组合为6D点云数据
        xyz = np.vstack((x, y, z)).T
        rgb = np.vstack((r, g, b)).T
        points_6d = np.hstack((xyz, rgb))
        
        # 处理点云为patches
        patches, patch_coords = process_pointcloud_to_patches(points_6d)
        
        # 检查是否成功生成了patches
        if len(patches) == 0:
            print(f"警告: {ply_path} 没有生成任何patches")
            return {
                "scene_id": scene_id,
                "status": "failed",
                "error": "No patches generated"
            }
        
        # 保存为NPZ
        np.savez(output_npz, patches=patches, patch_coords=patch_coords)
        
        return {
            "scene_id": scene_id,
            "npz_path": output_npz,
            "status": "success",
            "num_patches": len(patches)
        }
    except Exception as e:
        print(f"处理文件 {ply_path} 时出错: {str(e)}")
        return {
            "scene_id": scene_id,
            "status": "failed",
            "error": str(e)
        }

def process_qa_data(json_path, pcl_base_dir, output_dir, output_jsonl, n_workers=None):
    """处理QA数据和点云文件，生成JSONL"""
    # 创建输出目录
    npz_dir = os.path.join(output_dir, "3d-grand")
    os.makedirs(npz_dir, exist_ok=True)
    
    # 加载QA数据
    print(f"加载QA数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # 收集所有独特的场景ID
    scene_ids = set()
    for item in qa_data:
        scene_ids.add(item["scene_id"])
    
    print(f"找到 {len(scene_ids)} 个独特场景ID")
    
    # 确定每个场景ID对应的PLY文件路径
    ply_tasks = []
    missing_scenes = []
    
    for scene_id in scene_ids:
        if '@' in scene_id:
            folder_id, scene_name = scene_id.split('@')
            # 构建PLY文件路径
            ply_path = os.path.join(pcl_base_dir, folder_id, scene_name, f"{scene_name}.ply")
            
            if os.path.exists(ply_path):
                ply_tasks.append((ply_path, scene_id, npz_dir))
            else:
                missing_scenes.append((scene_id, ply_path))
        else:
            missing_scenes.append((scene_id, "无法解析的场景ID"))
    
    # 报告缺失场景
    if missing_scenes:
        print(f"警告: 找不到 {len(missing_scenes)} 个场景的PLY文件")
        for scene_id, path in missing_scenes[:10]:
            print(f"  - {scene_id}: {path}")
        if len(missing_scenes) > 10:
            print(f"  - ... 以及 {len(missing_scenes) - 10} 个更多")
    
    # 设置工作进程数
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # 多进程处理PLY文件
    print(f"使用 {n_workers} 个工作进程处理 {len(ply_tasks)} 个PLY文件...")
    results = {}
    
    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(pool.imap(load_and_process_ply, ply_tasks), total=len(ply_tasks), desc="处理PLY文件"):
            results[result["scene_id"]] = result
    
    # 统计处理结果
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    skipped_count = sum(1 for r in results.values() if r["status"] == "skipped")
    failed_count = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"PLY处理完成: 成功={success_count}, 跳过={skipped_count}, 失败={failed_count}")
    
    # 生成JSONL文件
    print(f"生成JSONL文件: {output_jsonl}")
    processed_count = 0
    skipped_count = 0
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in tqdm(qa_data, desc="生成JSONL"):
            scene_id = item["scene_id"]
            
            # 检查场景是否成功处理
            if scene_id not in results or results[scene_id]["status"] == "failed":
                skipped_count += 1
                continue
            
            # 获取问题和答案
            question = item["question"]
            answer = item["answer"]
            
            # 清理文本
            cleaned_question = clean_text(question)
            cleaned_answer = clean_text(answer)
            
            # 从scene_id中提取场景名
            if '@' in scene_id:
                _, scene_name = scene_id.split('@')
            else:
                scene_name = scene_id
            
            # 构建最终的对话数据
            dialogue = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>{cleaned_question}"},
                    {"role": "assistant", "content": cleaned_answer}
                ],
                "images": [f"3d-grand/{scene_name}.npz"]
            }
            
            # 写入JSONL
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
            processed_count += 1
    
    print(f"JSONL生成完成: 处理={processed_count}, 跳过={skipped_count}")
    print(f"输出文件: {output_jsonl}")

def main():
    parser = argparse.ArgumentParser(description='处理空间关系QA数据和点云文件')
    parser.add_argument('--json_path', type=str, required=True, help='QA JSON文件路径')
    parser.add_argument('--pcl_dir', type=str, required=True, help='点云基础目录')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--output_jsonl', type=str, help='输出JSONL文件路径')
    parser.add_argument('--n_workers', type=int, default=None, help='工作进程数')
    args = parser.parse_args()
    
    # 设置默认的输出JSONL路径
    if args.output_jsonl is None:
        json_name = os.path.basename(args.json_path).replace('.json', '.jsonl')
        args.output_jsonl = os.path.join(args.output_dir, json_name)
    
    process_qa_data(
        args.json_path,
        args.pcl_dir,
        args.output_dir,
        args.output_jsonl,
        args.n_workers
    )

if __name__ == "__main__":
    main()