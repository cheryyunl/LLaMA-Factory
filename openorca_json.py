import pandas as pd
import json
import os
import time
from glob import glob
from tqdm import tqdm

# 数据集路径
parquet_dir = "/scratch/zt1/project/furongh-prj/user/cheryunl/OpenOrca"
output_path = os.path.join(parquet_dir, "/scratch/zt1/project/furongh-prj/user/cheryunl/LLaMA-Factory/data/openorca.jsonl")

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 获取所有parquet文件
parquet_files = sorted(glob(os.path.join(parquet_dir, "*.parquet")))
print(f"找到 {len(parquet_files)} 个parquet文件")

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# 估计token数量的简单函数（每个单词约1.3个token）
def estimate_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

# 转换全部文件并生成JSONL文件（每行一个JSON对象）
def convert_all_files(files, max_samples=None, batch_size=5000, max_tokens=2048):
    samples_count = 0
    filtered_count = 0
    start_time = time.time()
    
    # 添加对dataset_info.json的修改提示
    print("\n提示: 别忘了在LLaMA-Factory的data/dataset_info.json中添加以下内容:")
    print("""
    "openorca_dataset": {
        "file_name": "openorca_converted.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
    """)
    
    # 打开输出文件，使用'w'模式创建新文件
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for file_idx, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            print(f"\n处理文件 ({file_idx+1}/{len(files)}): {file_name}")
            
            try:
                # 读取parquet文件
                df = pd.read_parquet(file_path)
                
                # 打印列名以便调试
                print(f"列名: {df.columns.tolist()}")
                print(f"文件中的样本数: {len(df)}")
                
                # 批处理以节省内存
                for start_idx in tqdm(range(0, len(df), batch_size), desc="处理批次"):
                    end_idx = min(start_idx + batch_size, len(df))
                    batch = df.iloc[start_idx:end_idx]
                    
                    for _, row in batch.iterrows():
                        # 检查是否达到最大样本数
                        if max_samples and samples_count >= max_samples:
                            print(f"已达到最大样本数 {max_samples}")
                            break
                        
                        # 初始化对话字典
                        conversation = {"messages": []}
                        total_tokens = 0
                        
                        # 添加系统消息
                        if "system_prompt" in df.columns and pd.notna(row["system_prompt"]):
                            system_content = row["system_prompt"]
                        else:
                            system_content = DEFAULT_SYSTEM_PROMPT

                        conversation["messages"].append({
                            "role": "system",
                            "content": system_content
                        })
                        total_tokens += estimate_tokens(system_content)
                        
                        # 添加用户消息
                        if "question" in df.columns and pd.notna(row["question"]):
                            user_content = row["question"]
                        elif "prompt" in df.columns and pd.notna(row["prompt"]):
                            user_content = row["prompt"]
                        else:
                            continue  # 跳过没有用户消息的行
                        
                        conversation["messages"].append({
                            "role": "user",
                            "content": user_content
                        })
                        total_tokens += estimate_tokens(user_content)
                        
                        # 添加助手消息
                        if "response" in df.columns and pd.notna(row["response"]):
                            assistant_content = row["response"]
                        elif "completion" in df.columns and pd.notna(row["completion"]):
                            assistant_content = row["completion"]
                        else:
                            continue  # 跳过没有助手消息的行
                        
                        conversation["messages"].append({
                            "role": "assistant",
                            "content": assistant_content
                        })
                        total_tokens += estimate_tokens(assistant_content)
                        
                        # 跳过过长的对话
                        if max_tokens > 0 and total_tokens > max_tokens:
                            filtered_count += 1
                            continue
                        
                        # 将对话写入JSONL文件
                        out_file.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                        samples_count += 1
                        
                    # 每处理一个批次显示进度
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        speed = samples_count / elapsed_time
                        print(f"已处理 {samples_count} 条记录，过滤 {filtered_count} 条，速度：{speed:.2f} 条/秒")
                
                if max_samples and samples_count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")
    
    # 显示样例（读取第一行作为示例）
    print("\n转换完成！")
    print(f"总共转换了 {samples_count} 条记录，过滤了 {filtered_count} 条过长记录")
    print(f"保存到 {output_path}")
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                print("\n数据样例 (第一条记录):")
                sample = json.loads(first_line)
                print(json.dumps(sample, ensure_ascii=False, indent=2))
                
    except Exception as e:
        print(f"读取样例时出错: {e}")
    
    print("\n下一步:")
    print("1. 将生成的jsonl文件复制到LLaMA-Factory/data/目录下")
    print("2. 在LLaMA-Factory/data/dataset_info.json中添加上面提到的配置")
    print("3. 使用LLaMA-Factory进行训练，选择数据集为'openorca_dataset'")

# 执行转换，可以设置max_samples限制样本数量
# 如果要转换所有数据，设置max_samples=None
# max_tokens=2048表示过滤掉估计超过2048个token的对话
convert_all_files(parquet_files, max_samples=None, batch_size=5000, max_tokens=2048)

print("转换完成！")