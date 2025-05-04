import os
import glob
from plyfile import PlyData

def check_multiple_ply_files(directory, pattern="*.ply"):
    files = glob.glob(os.path.join(directory, pattern))
    results = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            plydata = PlyData.read(file_path)
            vertex = plydata['vertex']
            
            # 检查颜色属性
            color_props = ['red', 'green', 'blue', 'r', 'g', 'b']
            has_color = any(p in vertex.dtype.names for p in color_props)
            
            results[filename] = has_color
        except Exception as e:
            results[filename] = f"错误: {str(e)}"
    
    return results

# 示例用法
directory = "/scratch/zt1/project/furongh-prj/user/cheryunl/PointCloud_zips_ABO/ABO_pcs"
results = check_multiple_ply_files(directory)

# 输出结果
with_color = [f for f, has_color in results.items() if has_color is True]
without_color = [f for f, has_color in results.items() if has_color is False]
error_files = [f for f, has_color in results.items() if isinstance(has_color, str)]

print(f"检查了 {len(results)} 个PLY文件")
print(f"包含颜色的文件: {len(with_color)}")
print(f"不包含颜色的文件: {len(without_color)}")
print(f"读取出错的文件: {len(error_files)}")