#!/usr/bin/env python3
import os
import random
import shutil
import argparse
import tqdm
import multiprocessing as mp

def copy_file(args):
    """复制单个文件的工作函数"""
    src_file, dst_file = args
    try:
        shutil.copy2(src_file, dst_file)
        return True
    except Exception as e:
        print(f"复制文件 {src_file} 时出错: {str(e)}")
        return False

def sample_files(source_dir, target_dir, sample_count, n_workers=None, file_extension='.ply'):
    """从源目录随机抽样文件到目标目录"""
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有指定扩展名的文件
    all_files = [f for f in os.listdir(source_dir) if f.endswith(file_extension)]
    total_files = len(all_files)
    
    print(f"在 {source_dir} 中找到 {total_files} 个{file_extension}文件")
    
    # 如果请求的样本数大于总文件数，调整样本数
    if sample_count >= total_files:
        print(f"请求的样本数 {sample_count} 大于可用文件数 {total_files}，将复制所有文件")
        selected_files = all_files
    else:
        print(f"随机抽取 {sample_count} 个文件...")
        selected_files = random.sample(all_files, sample_count)
    
    # 准备复制任务
    copy_tasks = []
    for filename in selected_files:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        copy_tasks.append((src_path, dst_path))
    
    # 设置工作进程数
    if n_workers is None:
        n_workers = min(32, mp.cpu_count())
    
    # 使用多进程复制文件
    print(f"使用 {n_workers} 个工作进程复制文件...")
    success_count = 0
    
    with mp.Pool(n_workers) as pool:
        results = list(tqdm.tqdm(
            pool.imap(copy_file, copy_tasks),
            total=len(copy_tasks),
            desc="复制文件"
        ))
        success_count = sum(results)
    
    print(f"复制完成！成功: {success_count}/{len(selected_files)}")
    print(f"文件已保存到: {target_dir}")
    return success_count

def main():
    parser = argparse.ArgumentParser(description='从大量文件中随机抽样指定数量到新文件夹')
    parser.add_argument('--source_dir', type=str, required=True, help='源文件夹路径')
    parser.add_argument('--target_dir', type=str, required=True, help='目标文件夹路径')
    parser.add_argument('--sample_count', type=int, default=50000, help='要抽取的文件数量')
    parser.add_argument('--extension', type=str, default='.ply', help='文件扩展名，默认为.ply')
    parser.add_argument('--n_workers', type=int, default=None, help='工作进程数，默认为CPU核心数')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子，用于复现')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 执行抽样
    sample_files(
        args.source_dir,
        args.target_dir,
        args.sample_count,
        args.n_workers,
        args.extension
    )

if __name__ == "__main__":
    main() 