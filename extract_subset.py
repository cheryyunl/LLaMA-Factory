#!/usr/bin/env python3
import os
import json
import random
import argparse
from tqdm import tqdm

def extract_subsets(input_json, output_dir, train_count=50000, eval_count=5000, seed=42):
    """
    从原始二元空间关系QA数据中抽取训练集和评估集
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
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 确保不超过可用数据量
    available_count = len(data)
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
    train_data = data[:train_count]
    
    # 直接取后面的数据作为评估集
    eval_data = data[train_count:train_count+eval_count]
    
    print(f"训练集包含 {len(train_data)} 个样本")
    print(f"评估集包含 {len(eval_data)} 个样本")
    
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
    parser.add_argument('--train_count', type=int, default=50000, help='训练集样本数')
    parser.add_argument('--eval_count', type=int, default=5000, help='评估集样本数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    extract_subsets(
        args.input_json,
        args.output_dir,
        args.train_count,
        args.eval_count,
        args.seed
    )

if __name__ == "__main__":
    main()