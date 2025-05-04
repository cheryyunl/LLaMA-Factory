#!/usr/bin/env python3
import json
import os
import re
import random
import argparse
from tqdm import tqdm

# 用户提示集合，用于随机采样
USER_PROMPTS = [
    "<image>What is this object?",
    "<image>This is a point cloud. What is this object?",
    "<image>Identify this 3D object.",
    "<image>Looking at this point cloud, what object does it represent?",
    "<image>Please classify this 3D point cloud."
]

# 需要过滤的修饰词列表
FILTER_MODIFIERS = [
    'low poly', 'lowpoly', 'high poly', 'highpoly', 'high-poly', 'low-poly',
    'cartoon', 'stylized', '3d', 'old', 'retro', 'used', 'vintage',
    'model', 'mesh', 'obj', 'fbx', 'blender', 'unity', 'unreal'
]

# 需要过滤的地理名词和类别
GEOGRAPHIC_TERMS = [
    'madagascar', 'africa', 'america', 'europe', 'asia', 'australia',
    'china', 'japan', 'usa', 'uk', 'france', 'germany', 'italy', 'spain',
    'russia', 'canada', 'brazil', 'india'
]

def clean_label(label):
    """清理和标准化对象标签"""
    if not label:
        return None
    
    # 转换为小写并移除前后空格
    cleaned = label.lower().strip()
    
    # 移除尺寸信息 (如 0.18x0.18x0.24)
    cleaned = re.sub(r'\d+\.?\d*\s*[x×]\s*\d+\.?\d*\s*[x×]\s*\d+\.?\d*', '', cleaned)
    
    # 移除文件扩展名和版本号
    cleaned = re.sub(r'\.(obj|fbx|stl|dae|3ds|max|blend)\b', '', cleaned)
    cleaned = re.sub(r'\b[vV]\d+\b', '', cleaned)
    cleaned = re.sub(r'\b\d+[.]?\d*\b', '', cleaned)  # 移除独立的数字
    
    # 移除常见的修饰词
    for modifier in FILTER_MODIFIERS:
        cleaned = re.sub(r'\b' + re.escape(modifier) + r'\b', '', cleaned)
    
    # 移除可能的文件名结构 (如 jar01, cup_02)
    cleaned = re.sub(r'\b[a-z]+\d+\b', '', cleaned)
    cleaned = re.sub(r'\b[a-z]+_\d+\b', '', cleaned)
    
    # 清理多余的空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else None

def is_valid_label(label):
    """检查标签是否有效"""
    if not label or label.isspace():
        return False
    
    # 清理标签
    cleaned_label = clean_label(label)
    if not cleaned_label:
        return False
    
    # 检查是否是纯数字
    if re.match(r'^[-+]?[\d.]+$', cleaned_label):
        return False
    
    # 检查是否是地址
    if re.search(r'\d+\s+\d*(st|nd|rd|th)\b', label.lower()):
        return False
    
    # 检查是否是地理名词
    if cleaned_label in GEOGRAPHIC_TERMS:
        return False
    
    # 检查长度，太短的标签可能没有意义
    if len(cleaned_label) < 3:
        return False
    
    return True

def get_clean_label_content(label):
    """获取清理后的标签内容"""
    if not is_valid_label(label):
        return None
    
    # 返回清理后的标签
    cleaned = clean_label(label)
    
    # 如果清理后标签太短或为空，返回原始标签
    if not cleaned or len(cleaned) < 3:
        return label.lower().strip()
    
    return cleaned

def clean_jsonl_file(input_file, output_file, target_dir="objaverse_lvis"):
    """清理JSONL文件，修复路径和系统提示"""
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return False
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 统计信息
    total_samples = 0
    removed_samples = 0
    fixed_path_count = 0
    fixed_prompt_count = 0
    cleaned_label_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            if not line.strip():
                continue
                
            total_samples += 1
            
            try:
                data = json.loads(line)
                
                # 1. 检查并修复对象标签
                assistant_msg = None
                for msg in data["messages"]:
                    if msg["role"] == "assistant":
                        assistant_msg = msg
                        break
                
                if assistant_msg is None:
                    removed_samples += 1
                    continue  # 跳过此样本
                
                # 获取清理后的标签
                original_label = assistant_msg["content"]
                cleaned_label = get_clean_label_content(original_label)
                
                if cleaned_label is None:
                    removed_samples += 1
                    continue  # 跳过此样本
                
                # 更新标签
                if cleaned_label != original_label:
                    assistant_msg["content"] = cleaned_label
                    cleaned_label_count += 1
                
                # 2. 修复系统提示中的空格问题
                for msg in data["messages"]:
                    if msg["role"] == "system" and "understanding3D" in msg["content"]:
                        msg["content"] = msg["content"].replace("understanding3D", "understanding 3D")
                        fixed_prompt_count += 1
                
                # 3. 随机更换用户提示
                for msg in data["messages"]:
                    if msg["role"] == "user" and "<image>" in msg["content"]:
                        msg["content"] = random.choice(USER_PROMPTS)
                
                # 4. 修复图像路径（确保路径格式为 "objaverse_lvis/000-141/xxxx.npz"）
                if "images" in data and data["images"]:
                    old_path = data["images"][0]
                    
                    # 提取路径中的重要部分
                    match = re.search(r'objaverse_lvis\/([^\/]+\/[^\/]+\.npz)', old_path)
                    if match:
                        new_path = f"{target_dir}/{match.group(1)}"
                        data["images"][0] = new_path
                        fixed_path_count += 1
                    else:
                        # 如果找不到匹配模式，保留原路径但尝试清理
                        # 移除 "patches/uni3d_data/" 前缀
                        new_path = re.sub(r'^patches\/uni3d_data\/', '', old_path)
                        # 删除路径中的反斜杠转义
                        new_path = new_path.replace('\\/', '/')
                        data["images"][0] = new_path
                        fixed_path_count += 1
                
                # 写入清理后的数据
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"警告：跳过无效的JSON行: {line[:50]}...")
    
    # 打印统计信息
    print(f"\n清理完成！统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"删除的样本数 (无效标签): {removed_samples}")
    print(f"清理的标签数: {cleaned_label_count}")
    print(f"修复的路径数: {fixed_path_count}")
    print(f"修复的系统提示数: {fixed_prompt_count}")
    print(f"保留的样本数: {total_samples - removed_samples}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='清理Objaverse LVIS数据集的JSONL文件')
    parser.add_argument('--input_file', type=str, required=True, help='输入的JSONL文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出的JSONL文件路径')
    parser.add_argument('--target_dir', type=str, default='objaverse_lvis', help='目标目录名称（默认：objaverse_lvis）')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子（默认：42）')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 清理文件
    success = clean_jsonl_file(args.input_file, args.output_file, args.target_dir)
    
    if success:
        print(f"\n清理后的数据已保存到: {args.output_file}")
    else:
        print("\n清理失败，请检查输入文件和参数！")

if __name__ == "__main__":
    main() 