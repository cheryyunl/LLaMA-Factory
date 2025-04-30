import pandas as pd
import json
import os
from glob import glob
import re

# 数据集路径
parquet_dir = "/scratch/zt1/project/furongh-prj/user/cheryunl/OpenOrca"

# 获取所有parquet文件
parquet_files = glob(os.path.join(parquet_dir, "*.parquet"))

# 默认的系统提示
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# 最大序列长度（令牌数）
MAX_SEQ_LENGTH = 2048

# 简单的令牌计数估算函数
def estimate_token_count(text):
    """
    估算文本中的令牌数量。
    这是一个简化的估算，实际令牌数会因分词器而异。
    """
    if not text:
        return 0
    
    # 使用简单启发式方法估算：
    # 1. 英文单词大约是1个令牌
    # 2. 标点符号可能是单独的令牌
    # 3. 按空格分词后再乘以1.3作为经验系数
    words = re.findall(r'\w+|[^\w\s]', text)
    return int(len(words) * 1.3)

# 转换单个文件
def convert_single_file(file_path):
    print(f"正在转换 {file_path}...")
    
    # 读取parquet文件
    df = pd.read_parquet(file_path)
    
    # 检查数据结构
    print("数据列:", df.columns.tolist())
    
    # 结果列表
    json_data = []
    filtered_count = 0
    kept_count = 0
    
    # 遍历每一行数据
    for _, row in df.iterrows():
        # 尝试从数据中获取相关字段
        system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # OpenOrca数据集可能有不同的列名，需要适配
        if "system" in df.columns:
            system_prompt = row["system"]
        elif "system_prompt" in df.columns:
            system_prompt = row["system_prompt"]
            
        # 获取用户提示
        if "prompt" in df.columns:
            prompt = row["prompt"]
        elif "question" in df.columns:
            prompt = row["question"]
        elif "user_message" in df.columns:
            prompt = row["user_message"]
        else:
            # 显示实际数据结构
            print("找不到用户提示列，展示数据结构:")
            print(df.head(1).to_dict('records'))
            return None
        
        # 获取助手回答
        if "label" in df.columns:
            label = row["label"]
        elif "response" in df.columns:
            label = row["response"]
        elif "completion" in df.columns:
            label = row["completion"]
        elif "answer" in df.columns:
            label = row["answer"]
        else:
            # 显示实际数据结构
            print("找不到助手回答列，展示数据结构:")
            print(df.head(1).to_dict('records'))
            return None
        
        # 估算总令牌数
        total_tokens = estimate_token_count(system_prompt) + estimate_token_count(prompt) + estimate_token_count(label)
        
        # 过滤掉超过最大长度的对话
        if total_tokens > MAX_SEQ_LENGTH:
            filtered_count += 1
            continue
        
        # 创建您指定的格式
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label}
            ]
        }
        json_data.append(conversation)
        kept_count += 1
    
    # 保存为JSON文件
    output_file = file_path.replace(".parquet", "_messages_filtered.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！已保存到 {output_file}")
    print(f"总共处理了 {kept_count + filtered_count} 条对话")
    print(f"保留了 {kept_count} 条对话，过滤掉 {filtered_count} 条超长对话")
    print(f"过滤率: {filtered_count/(kept_count + filtered_count)*100:.2f}%")
    
    # 显示数据样例
    print("\n数据样例 (前1条记录):")
    if json_data:
        print(json.dumps(json_data[0], ensure_ascii=False, indent=2))
    
    return json_data[:1] if json_data else None

# 先检查第一个文件的结构
def check_file_structure(file_path):
    df = pd.read_parquet(file_path)
    print("文件:", os.path.basename(file_path))
    print("列名:", df.columns.tolist())
    print("数据示例:")
    print(df.head(1).to_dict('records'))
    return df.columns.tolist()

# 转换所有文件的函数
def convert_all_files():
    total_kept = 0
    total_filtered = 0
    for file in parquet_files:
        print(f"\n处理文件: {os.path.basename(file)}")
        # 读取parquet文件
        df = pd.read_parquet(file)
        
        # 结果列表
        json_data = []
        filtered_count = 0
        kept_count = 0
        
        # 遍历每一行数据
        for _, row in df.iterrows():
            # 与单文件转换函数相同的逻辑
            system_prompt = DEFAULT_SYSTEM_PROMPT
            
            if "system" in df.columns:
                system_prompt = row["system"]
            elif "system_prompt" in df.columns:
                system_prompt = row["system_prompt"]
                
            # 获取用户提示
            if "prompt" in df.columns:
                prompt = row["prompt"]
            elif "question" in df.columns:
                prompt = row["question"]
            elif "user_message" in df.columns:
                prompt = row["user_message"]
            else:
                print(f"文件 {file} 找不到用户提示列，跳过此文件")
                break
            
            # 获取助手回答
            if "label" in df.columns:
                label = row["label"]
            elif "response" in df.columns:
                label = row["response"]
            elif "completion" in df.columns:
                label = row["completion"]
            elif "answer" in df.columns:
                label = row["answer"]
            else:
                print(f"文件 {file} 找不到助手回答列，跳过此文件")
                break
            
            # 估算总令牌数
            total_tokens = estimate_token_count(system_prompt) + estimate_token_count(prompt) + estimate_token_count(label)
            
            # 过滤掉超过最大长度的对话
            if total_tokens > MAX_SEQ_LENGTH:
                filtered_count += 1
                continue
            
            # 创建您指定的格式
            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label}
                ]
            }
            json_data.append(conversation)
            kept_count += 1
        
        # 保存为JSON文件
        output_file = file.replace(".parquet", "_messages_filtered.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"文件 {os.path.basename(file)} 转换完成！")
        print(f"保留了 {kept_count} 条对话，过滤掉 {filtered_count} 条超长对话")
        
        total_kept += kept_count
        total_filtered += filtered_count
    
    print("\n全部转换完成!")
    print(f"总共保留了 {total_kept} 条对话，过滤掉 {total_filtered} 条超长对话")
    print(f"总过滤率: {total_filtered/(total_kept + total_filtered)*100:.2f}%")

# 检查第一个文件
sample_file = parquet_files[0]
columns = check_file_structure(sample_file)

# 可以运行以下命令转换单个文件:
# convert_single_file(sample_file)

# 可以运行以下命令转换所有文件:
# convert_all_files()