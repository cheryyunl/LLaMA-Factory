import json
import random
import tqdm
import os

# 输入和输出文件路径
input_file = "data/openorca.jsonl"  # 替换为您的输入文件路径
output_file = "data/openorca_0.6.jsonl"  # 替换为您想要的输出文件路径

# 要提取的样本数量
num_samples = 600000

# 计算数据集总行数
print("正在计算数据集总样本数...")
with open(input_file, "r", encoding="utf-8") as f:
    total_samples = sum(1 for _ in f)
print(f"数据集总样本数: {total_samples}")

if num_samples >= total_samples:
    print("请求的样本数大于或等于数据集大小，将复制整个数据集。")
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(infile.read())
    print(f"已复制全部 {total_samples} 个样本到 {output_file}")
else:
    # 确定要采样的行索引(内存高效方法)
    print("准备采样索引...")
    sample_indices = set(random.sample(range(total_samples), num_samples))
    
    # 执行采样
    print("提取样本中...")
    count = 0
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in tqdm.tqdm(enumerate(infile), total=total_samples, desc="采样进度"):
            if i in sample_indices:
                outfile.write(line)
                count += 1
    
    print(f"成功提取了 {count} 个样本到 {output_file}")