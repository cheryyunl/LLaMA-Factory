#!/usr/bin/env python3
import os
import json
import random
import argparse
from tqdm import tqdm

def extract_subsets(input_json, output_dir, pcl_base_dir, train_count=50000, eval_count=5000, seed=42):
    """
    从原始二元空间关系QA数据中抽取训练集和评估集，
    仅包含有对应PLY文件的样本，并确保评估集场景不在训练集中出现
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载数据：{input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共加载了 {len(data)} 个问答对")
    
    # 筛选有对应PLY文件的样本
    filtered_data = []
    missing_count = 0
    
    print("检查PLY文件可用性...")
    for item in tqdm(data):
        scene_id = item["scene_id"]
        # 解析场景ID，构建PLY文件路径
        if '@' in scene_id:
            folder_id, scene_name = scene_id.split('@')
            ply_path = os.path.join(pcl_base_dir, folder_id, scene_name, f"{scene_name}.ply")
            if os.path.exists(ply_path):
                filtered_data.append(item)
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    print(f"过滤后剩余 {len(filtered_data)} 个有效样本（丢弃了 {missing_count} 个无对应PLY文件的样本）")
    
    # 随机打乱过滤后的数据
    random.shuffle(filtered_data)
    
    # 确保不超过可用数据量
    available_count = len(filtered_data)
    if train_count + eval_count > available_count:
        print(f"警告: 请求的样本数 ({train_count}+{eval_count}) 超过可用数据 ({available_count})")
        if available_count > eval_count:
            train_count = available_count - eval_count
            print(f"调整训练集大小为 {train_count}")
        else:
            train_count = int(available_count * 0.9)
            eval_count = available_count - train_count
            print(f"调整为训练集 {train_count} 样本，评估集 {eval_count} 样本")
    
    # 抽取训练集
    train_data = filtered_data[:train_count]
    
    # 计算训练集中的场景ID
    train_scenes = set(item["scene_id"] for item in train_data)
    print(f"训练集包含 {len(train_scenes)} 个唯一场景ID")
    
    # 抽取不重复的评估集（确保场景不在训练集中）
    remaining_data = filtered_data[train_count:]
    eval_data = []
    excluded_count = 0
    
    for item in remaining_data:
        if len(eval_data) >= eval_count:
            break
            
        # 确保评估集样本的场景ID不在训练集中
        if item["scene_id"] not in train_scenes:
            eval_data.append(item)
        else:
            excluded_count += 1
    
    print(f"从剩余数据中排除了 {excluded_count} 个与训练集场景重叠的样本")
    print(f"评估集包含 {len(eval_data)} 个样本")
    
    # 如果评估集数量不足，可以考虑放宽限制，允许一些场景重叠
    if len(eval_data) < eval_count * 0.8:  # 如果不到要求的80%
        print(f"警告: 不重叠评估集仅包含 {len(eval_data)} 样本，低于目标 {eval_count}。将添加部分与训练集场景重叠的样本")
        
        # 添加一些可能重叠的样本，直到达到目标数量
        additional_needed = eval_count - len(eval_data)
        random.shuffle(remaining_data)  # 再次打乱以随机选择
        
        for item in remaining_data:
            if item not in eval_data and len(eval_data) < eval_count:
                eval_data.append(item)
                if len(eval_data) >= eval_count:
                    break
    
    print(f"最终评估集包含 {len(eval_data)} 个样本")
    
    # 保存训练集
    train_output = os.path.join(output_dir, "binary_spatial_qa.json")
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存评估集
    eval_output = os.path.join(output_dir, "binary_spatial_qa_eval.json")
    with open(eval_output, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练集保存至: {train_output} ({len(train_data)} 样本)")
    print(f"评估集保存至: {eval_output} ({len(eval_data)} 样本)")
    
    # 统计一些基本信息
    question_types = {}
    for item in train_data:
        q_type = item.get("question_type", "unknown")
        question_types[q_type] = question_types.get(q_type, 0) + 1
    
    print("\n训练集问题类型分布:")
    for q_type, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {q_type}: {count} ({count/len(train_data)*100:.1f}%)")
    
    return train_output, eval_output

def main():
    parser = argparse.ArgumentParser(description='从二元空间关系QA数据中抽取训练集和评估集')
    parser.add_argument('--input_json', type=str, required=True, help='输入JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--pcl_dir', type=str, required=True, help='点云文件基础目录')
    parser.add_argument('--train_count', type=int, default=50000, help='训练集样本数')
    parser.add_argument('--eval_count', type=int, default=5000, help='评估集样本数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    extract_subsets(
        args.input_json,
        args.output_dir,
        args.pcl_dir,
        args.train_count,
        args.eval_count,
        args.seed
    )

if __name__ == "__main__":
    main()