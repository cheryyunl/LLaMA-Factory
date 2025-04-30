import pandas as pd
import json
import os
from glob import glob

# 数据集路径
parquet_dir = "/scratch/zt1/project/furongh-prj/user/cheryunl/OpenOrca"

# 获取所有parquet文件
parquet_files = glob(os.path.join(parquet_dir, "*.parquet"))

# 转换单个文件示例（以第一个文件为例）
def convert_single_file(file_path):
    print(f"正在转换 {file_path}...")
    
    # 读取parquet文件
    df = pd.read_parquet(file_path)
    
    # 转换为JSON格式
    output_file = file_path.replace(".parquet", ".json")
    
    # 使用orient="records"将每条记录转为单独的JSON对象
    df.to_json(output_file, orient="records", lines=False, indent=2)
    
    print(f"转换完成！已保存到 {output_file}")
    
    # 显示数据样例
    print("\n数据样例 (前2条记录):")
    print(df.head(2).to_json(orient="records", indent=2))
    
    return df.head(2)

# 转换第一个文件并显示样例
sample_file = parquet_files[0]
sample_data = convert_single_file(sample_file)

# 定义转换所有文件的函数
def convert_all_files():
    for file in parquet_files:
        convert_single_file(file)

# 如需转换所有文件，请取消下行注释
convert_all_files()