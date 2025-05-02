#!/usr/bin/env python3
import json
import sys

def check_jsonl_file(file_path):
    """
    检查JSONL文件中的每一行，确保如果用户内容中包含<image>，则存在images字段且指向.npz文件
    
    Args:
        file_path (str): JSONL文件的路径
    
    Returns:
        tuple: (通过检查的行数, 不通过检查的行列表)
    """
    line_num = 0
    valid_count = 0
    invalid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 检查是否存在用户消息且包含<image>标记
                has_image_tag = False
                for message in data.get('messages', []):
                    if message.get('role') == 'user' and '<image>' in message.get('content', ''):
                        has_image_tag = True
                        break
                
                # 如果有<image>标记，则检查images字段
                if has_image_tag:
                    # 检查images字段是否存在且为非空列表
                    if 'images' not in data or not isinstance(data['images'], list) or not data['images']:
                        invalid_lines.append((line_num, f"没有images字段或字段为空: {line[:100]}..."))
                        continue
                    
                    # 检查images列表中的第一个元素是否以.npz结尾
                    image_path = data['images'][0]
                    if not isinstance(image_path, str) or not image_path.endswith('.npz'):
                        invalid_lines.append((line_num, f"images不是.npz文件: {image_path}"))
                        continue
                
                valid_count += 1
            
            except json.JSONDecodeError:
                invalid_lines.append((line_num, f"JSON解析错误: {line[:100]}..."))
    
    return valid_count, invalid_lines

def main():
    if len(sys.argv) < 2:
        print("用法: python check_jsonl.py <jsonl_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    valid_count, invalid_lines = check_jsonl_file(file_path)
    
    print(f"检查完成: {valid_count} 行有效")
    
    if invalid_lines:
        print(f"发现 {len(invalid_lines)} 行无效:")
        for line_num, reason in invalid_lines:
            print(f"  行 {line_num}: {reason}")
    else:
        print("所有行都符合要求!")

if __name__ == "__main__":
    main()