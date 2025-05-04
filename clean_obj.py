#!/usr/bin/env python3
import json
import re
import argparse
from tqdm import tqdm

def remove_brackets(text):
    """从文本中删除各种括号及其中的内容"""
    # 移除圆括号及其内容
    text = re.sub(r'\([^)]*\)', '', text)
    # 移除方括号及其内容
    text = re.sub(r'\[[^\]]*\]', '', text)
    # 移除花括号及其内容
    text = re.sub(r'\{[^}]*\}', '', text)
    # 移除尖括号及其内容
    text = re.sub(r'<[^>]*>', '', text)
    # 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_jsonl(input_file, output_file):
    """清理JSONL文件中的括号内容"""
    count_total = 0
    count_cleaned = 0
    
    print(f"正在处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            if not line.strip():
                continue
                
            count_total += 1
            
            try:
                data = json.loads(line)
                
                # 查找assistant消息
                for msg in data["messages"]:
                    if msg["role"] == "assistant":
                        original = msg["content"]
                        cleaned = remove_brackets(original)
                        
                        if cleaned != original:
                            msg["content"] = cleaned
                            count_cleaned += 1
                        
                        break
                
                # 写入处理后的数据
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"警告: 跳过无效JSON行: {line[:50]}...")
    
    print(f"\n处理完成! 统计信息:")
    print(f"总样本数: {count_total}")
    print(f"清理的样本数: {count_cleaned}")
    print(f"清理后的文件已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='从JSONL文件中清理助手回答里的括号内容')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入的JSONL文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出的JSONL文件路径')
    args = parser.parse_args()
    
    clean_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()