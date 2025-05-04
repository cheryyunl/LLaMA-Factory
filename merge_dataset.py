#!/usr/bin/env python3
import os
import json
import random
import argparse
import glob
from collections import Counter
from tqdm import tqdm

def load_jsonl(file_path):
    """加载单个JSONL文件的内容"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line[:50]}...")
    return data

def extract_object_category(item):
    """从样本中提取物体种类"""
    try:
        # 找到assistant的消息
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                # 获取物体类别并去除可能的空格和标点
                category = msg["content"].strip().lower()
                # 移除句号等标点符号
                if category.endswith('.'):
                    category = category[:-1]
                return category
    except (KeyError, IndexError):
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description='合并多个scene_object数据集并创建评估基准')
    parser.add_argument('--data_dir', type=str, required=True, help='包含JSONL文件的数据目录')
    parser.add_argument('--output_file', type=str, default='benchmark_eval.jsonl', help='输出文件路径')
    parser.add_argument('--num_samples', type=int, default=30000, help='要抽取的样本数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 查找所有JSONL文件
    jsonl_files = glob.glob(os.path.join(args.data_dir, '**', '*.jsonl'), recursive=True)
    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    
    if not jsonl_files:
        print(f"错误: 在 {args.data_dir} 中未找到JSONL文件")
        return
    
    # 加载所有数据并提取物体种类
    all_data = []
    categories = Counter()
    sample_counts = {}
    
    print("正在加载并处理数据...")
    for file_path in tqdm(jsonl_files):
        file_data = load_jsonl(file_path)
        file_categories = Counter()
        
        for item in file_data:
            category = extract_object_category(item)
            if category:
                file_categories[category] += 1
            all_data.append((item, category, file_path))
        
        categories.update(file_categories)
        sample_counts[file_path] = len(file_data)
        print(f"  {os.path.basename(file_path)}: {len(file_data)} 个样本, {len(file_categories)} 种物体")
    
    print(f"\n总共加载了 {len(all_data)} 个样本, {len(categories)} 种不同的物体")
    
    # 确保我们有足够的数据
    if len(all_data) < args.num_samples:
        print(f"警告: 可用样本 ({len(all_data)}) 少于请求的样本数 ({args.num_samples})")
        num_samples = len(all_data)
    else:
        num_samples = args.num_samples
    
    # 随机抽样
    print(f"正在随机抽取 {num_samples} 个样本...")
    selected_indices = random.sample(range(len(all_data)), num_samples)
    selected_data = [all_data[i][0] for i in selected_indices]
    selected_categories = Counter([all_data[i][1] for i in selected_indices if all_data[i][1]])
    
    # 统计每个文件贡献的样本数
    file_contributions = Counter([all_data[i][2] for i in selected_indices])
    
    # 写入输出文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in selected_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"评估基准数据集已保存到: {args.output_file}")
    
    # 打印物体种类统计信息
    print("\n物体种类统计:")
    print(f"总共有 {len(selected_categories)} 种不同的物体")
    
    # 按数量排序并打印
    sorted_categories = sorted(selected_categories.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        if count > 1:  # 只显示出现多次的种类
            print(f"  {category}: {count} 个样本")
    
    # 单次出现的种类合并统计
    single_occurrence = sum(1 for _, count in selected_categories.items() if count == 1)
    if single_occurrence > 0:
        print(f"  其他 {single_occurrence} 种物体各有 1 个样本")
    
    # 打印文件贡献统计
    print("\n各文件贡献的样本数:")
    for file_path, count in file_contributions.most_common():
        file_name = os.path.basename(file_path)
        total_in_file = sample_counts[file_path]
        percentage = (count / total_in_file) * 100
        print(f"  {file_name}: {count}/{total_in_file} ({percentage:.1f}%)")
    
    print("\n完成!")

if __name__ == "__main__":
    main() 