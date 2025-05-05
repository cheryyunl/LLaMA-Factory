#!/usr/bin/env python3
import os
import json
import numpy as np
from tqdm import tqdm
from plyfile import PlyData

def normalize_coordinates(ply_path, scene_graph_objects):
    """读取PLY文件并计算用于归一化的参数，然后应用到场景图对象"""
    try:
        # 读取PLY文件
        plydata = PlyData.read(ply_path)
        vertex_data = plydata['vertex'].data
        
        # 提取坐标
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        
        # 计算归一化参数（与点云处理中相同）
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        x_scale = max(abs(x_max - x_center), abs(x_min - x_center)) * 1.05
        y_scale = max(abs(y_max - y_center), abs(y_min - y_center)) * 1.05
        z_scale = max(abs(z_max - z_center), abs(z_min - z_center)) * 1.05
        
        max_scale = max(x_scale, y_scale, z_scale)
        
        # 对场景图中的每个对象应用相同的归一化
        normalized_objects = {}
        for obj_id, obj_info in scene_graph_objects.items():
            obj_copy = obj_info.copy()
            
            # 归一化centroid
            if "centroid" in obj_copy:
                centroid = obj_copy["centroid"]
                norm_centroid = [
                    (centroid[0] - x_center) / max_scale if max_scale > 0 else 0,
                    (centroid[1] - y_center) / max_scale if max_scale > 0 else 0,
                    (centroid[2] - z_center) / max_scale if max_scale > 0 else 0
                ]
                obj_copy["original_centroid"] = centroid  # 保留原始值
                obj_copy["centroid"] = norm_centroid
            
            # 归一化extent (缩放但不平移)
            if "extent" in obj_copy:
                extent = obj_copy["extent"]
                norm_extent = [
                    extent[0] / max_scale if max_scale > 0 else 0,
                    extent[1] / max_scale if max_scale > 0 else 0,
                    extent[2] / max_scale if max_scale > 0 else 0
                ]
                obj_copy["original_extent"] = extent  # 保留原始值
                obj_copy["extent"] = norm_extent
            
            normalized_objects[obj_id] = obj_copy
        
        return normalized_objects, True
    
    except Exception as e:
        print(f"归一化失败: {ply_path}, 错误: {str(e)}")
        return scene_graph_objects, False

def add_object_contexts_with_normalization(input_jsonl, scene_graph_json, pcl_base_dir, output_jsonl):
    """修改JSONL文件，添加归一化后的对象上下文信息"""
    print(f"加载场景图数据: {scene_graph_json}")
    with open(scene_graph_json, 'r', encoding='utf-8') as f:
        scene_graph = json.load(f)
    
    print(f"处理JSONL文件: {input_jsonl}")
    input_data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            input_data.append(json.loads(line.strip()))
    
    print(f"总共 {len(input_data)} 个对话样本")
    
    updated_data = []
    skipped_count = 0
    normalized_count = 0
    
    for item in tqdm(input_data, desc="处理对话"):
        # 获取对话信息
        system_msg = item["messages"][0]["content"]
        user_msg = item["messages"][1]["content"]
        assistant_msg = item["messages"][2]["content"]
        images = item["images"]
        
        # 解析场景ID和目标对象
        scene_id = None
        for line in user_msg.split('\n'):
            if line.startswith("scene_id:"):
                scene_id = line[9:].strip()
                break
        
        if not scene_id or scene_id not in scene_graph:
            skipped_count += 1
            updated_data.append(item)  # 保持原样
            continue
        
        # 构建PLY文件路径
        if '@' in scene_id:
            folder_id, scene_name = scene_id.split('@')
            ply_path = os.path.join(pcl_base_dir, folder_id, scene_name, f"{scene_name}.ply")
        else:
            ply_path = None
        
        # 获取原始问题内容
        question_part = user_msg.split('<image>')[-1].strip()
        
        # 提取问题中提到的所有对象ID
        import re
        mentioned_objects = re.findall(r'<([^>]+)>', question_part)
        
        # 归一化场景图中的对象坐标
        normalized_objects = scene_graph[scene_id]
        if ply_path and os.path.exists(ply_path):
            normalized_objects, success = normalize_coordinates(ply_path, scene_graph[scene_id])
            if success:
                normalized_count += 1
        
        # 构建对象上下文信息
        obj_contexts = []
        for obj_id in mentioned_objects:
            full_obj_id = f"<{obj_id}>"
            if full_obj_id in normalized_objects:
                obj_info = normalized_objects[full_obj_id]
                
                # 创建简化的上下文
                context = {
                    "centroid": obj_info.get("centroid", []),
                    "extent": obj_info.get("extent", [])
                }
                
                
                obj_contexts.append(f"<{obj_id}>: {json.dumps(context, ensure_ascii=False)}")
        
        # 构建新的用户消息
        new_user_msg = f"<image>{question_part}"
        if obj_contexts:
            new_user_msg += f"\nNormalized object grounding contexts:\n{', '.join(obj_contexts)}"
        
        # 更新对话
        updated_item = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": new_user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "images": images
        }
        
        updated_data.append(updated_item)
    
    print(f"跳过 {skipped_count} 个找不到场景信息的样本")
    print(f"成功归一化 {normalized_count} 个场景中的对象坐标")
    
    # 保存更新后的JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"更新完成! 保存至: {output_jsonl}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='添加归一化后的对象上下文信息到JSONL文件')
    parser.add_argument('--input_jsonl', type=str, required=True, help='输入的JSONL文件')
    parser.add_argument('--scene_graph', type=str, required=True, help='场景图JSON文件')
    parser.add_argument('--pcl_dir', type=str, required=True, help='点云文件基础目录')
    parser.add_argument('--output_jsonl', type=str, required=True, help='输出的JSONL文件')
    args = parser.parse_args()
    
    add_object_contexts_with_normalization(args.input_jsonl, args.scene_graph, args.pcl_dir, args.output_jsonl)

if __name__ == "__main__":
    main()