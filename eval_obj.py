import json
import random
import os

# 设置随机种子以确保结果可重现
random.seed(42)

# 定义要抽样的数据集和数量
datasets = [
    {"name": "data/objaverse_lvis.jsonl", "sample_count": 2000},
    {"name": "data/scannet.jsonl", "sample_count": 1000}
]

# 创建评估数据集
eval_samples = []

# 处理每个数据集
for dataset in datasets:
    filename = dataset["name"]
    sample_count = dataset["sample_count"]
    
    # 读取原始数据集
    original_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 确保行不为空
                original_data.append(json.loads(line))
    
    # 确保数据集大小足够
    if len(original_data) < sample_count:
        print(f"警告: {filename} 只有 {len(original_data)} 条数据，少于请求的 {sample_count} 条")
        sample_count = len(original_data)
    
    # 随机抽样数据
    sampled_indices = random.sample(range(len(original_data)), sample_count)
    sampled_indices.sort()  # 排序以便后续删除
    
    # 添加到评估数据集
    for idx in sampled_indices:
        sample = original_data[idx]
        eval_samples.append(sample)
    
    # 创建新的数据集（不含抽样的数据）
    remaining_data = [original_data[i] for i in range(len(original_data)) if i not in sampled_indices]
    
    # 保存修改后的原始数据集
    with open(f"new_{filename}", 'w', encoding='utf-8') as f:
        for item in remaining_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"从 {filename} 中抽取了 {sample_count} 条数据，剩余 {len(remaining_data)} 条")

# 保存评估数据集
with open("data/obj_eval.jsonl", 'w', encoding='utf-8') as f:
    for item in eval_samples:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"创建了评估数据集 obj_eval.jsonl，共 {len(eval_samples)} 条数据")