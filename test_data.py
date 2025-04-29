#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import random

def validate_jsonl_file(jsonl_path):
    """验证JSONL文件格式和内容"""
    print(f"\n检查JSONL文件: {jsonl_path}")
    try:
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
        
        print(f"  包含 {len(lines)} 个样本")
        
        # 随机抽查5个样本（或全部）
        sample_count = min(5, len(lines))
        samples = random.sample(range(len(lines)), sample_count)
        
        valid_count = 0
        for i in samples:
            try:
                data = json.loads(lines[i])
                # 检查必要字段
                assert "messages" in data, "缺少messages字段"
                assert len(data["messages"]) >= 2, "messages不足2条"
                assert "images" in data, "缺少images字段"
                assert len(data["images"]) == 1, "images应该只有1个"
                assert isinstance(data["images"][0], str), "image路径应为字符串"
                
                # 检查用户和助手消息
                user_msg = data["messages"][1]
                assistant_msg = data["messages"][2]
                assert user_msg["role"] == "user", "第一条消息应为user"
                assert assistant_msg["role"] == "assistant", "第二条消息应为assistant"
                assert "<image>" in user_msg["content"], "用户消息应包含<image>标记"
                
                valid_count += 1
            except Exception as e:
                print(f"  ❌ 样本 {i} 验证失败: {e}")
        
        success_rate = valid_count / sample_count * 100
        print(f"  JSONL验证: {'✅ 通过' if success_rate == 100 else '⚠️ 部分通过'} ({success_rate:.1f}%验证成功)")
        return success_rate == 100
    except Exception as e:
        print(f"  ❌ JSONL文件验证失败: {e}")
        return False

def validate_npz_files(jsonl_path, check_normalization=True):
    """验证NPZ文件存在性和内容"""
    print(f"\n检查NPZ文件 (from {jsonl_path})")
    try:
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
        
        jsonl_dir = os.path.dirname(jsonl_path)
        
        # 随机抽查最多10个样本
        sample_count = min(10, len(lines))
        samples = random.sample(range(len(lines)), sample_count)
        
        valid_count = 0
        missing_count = 0
        stats = {
            "total_patches": 0,
            "min_patches": float('inf'),
            "max_patches": 0,
            "xyz_min": float('inf'),
            "xyz_max": float('-inf'),
            "rgb_min": float('inf'),
            "rgb_max": float('-inf')
        }
        
        for i in samples:
            try:
                data = json.loads(lines[i])
                # 获取NPZ文件路径
                npz_rel_path = data["images"][0]
                npz_path = os.path.join(jsonl_dir, npz_rel_path)
                
                if not os.path.exists(npz_path):
                    print(f"  ❌ 样本 {i} NPZ文件不存在: {npz_path}")
                    missing_count += 1
                    continue
                
                # 加载NPZ并检查内容
                try:
                    with np.load(npz_path) as npz:
                        patches = npz["patches"]
                        patch_coords = npz["patch_coords"]
                        
                        # 检查形状
                        assert len(patches) > 0, "patches为空"
                        assert len(patch_coords) == len(patches), "patches和patch_coords长度不匹配"
                        assert patches.shape[1] == 512, "每个patch应该有512个点"
                        assert patches.shape[2] == 6, "每个点应该有6个维度(xyz+rgb)"
                        
                        # 统计信息
                        stats["total_patches"] += len(patches)
                        stats["min_patches"] = min(stats["min_patches"], len(patches))
                        stats["max_patches"] = max(stats["max_patches"], len(patches))
                        
                        # 提取xyz和rgb
                        xyz = patches.reshape(-1, 6)[:, :3]
                        rgb = patches.reshape(-1, 6)[:, 3:]
                        
                        # 更新最值
                        stats["xyz_min"] = min(stats["xyz_min"], xyz.min())
                        stats["xyz_max"] = max(stats["xyz_max"], xyz.max())
                        stats["rgb_min"] = min(stats["rgb_min"], rgb.min())
                        stats["rgb_max"] = max(stats["rgb_max"], rgb.max())
                        
                        # 检查归一化
                        if check_normalization:
                            assert -1.01 <= xyz.min() <= 1.01 and -1.01 <= xyz.max() <= 1.01, f"xyz值超出归一化范围 [{xyz.min()}, {xyz.max()}]"
                            assert -0.01 <= rgb.min() <= 1.01 and -0.01 <= rgb.max() <= 1.01, f"rgb值超出归一化范围 [{rgb.min()}, {rgb.max()}]"
                        
                        valid_count += 1
                except Exception as e:
                    print(f"  ❌ 样本 {i} NPZ内容验证失败: {e}")
            except Exception as e:
                print(f"  ❌ 样本 {i} 验证失败: {e}")
        
        # 打印统计信息
        print(f"\n  NPZ文件统计:")
        print(f"  - 验证通过: {valid_count}/{sample_count} ({valid_count/sample_count*100:.1f}%)")
        print(f"  - 文件缺失: {missing_count}/{sample_count} ({missing_count/sample_count*100:.1f}%)")
        print(f"  - 平均每个点云的patch数: {stats['total_patches']/(valid_count or 1):.1f}")
        print(f"  - patch数范围: {stats['min_patches']} - {stats['max_patches']}")
        print(f"  - XYZ值范围: [{stats['xyz_min']:.3f}, {stats['xyz_max']:.3f}]")
        print(f"  - RGB值范围: [{stats['rgb_min']:.3f}, {stats['rgb_max']:.3f}]")
        
        xyz_normalized = -1.1 <= stats["xyz_min"] <= 1.1 and -1.1 <= stats["xyz_max"] <= 1.1
        rgb_normalized = -0.1 <= stats["rgb_min"] <= 1.1 and -0.1 <= stats["rgb_max"] <= 1.1
        
        print(f"  - XYZ是否已归一化到[-1,1]: {'✅ 是' if xyz_normalized else '❌ 否'}")
        print(f"  - RGB是否已归一化到[0,1]: {'✅ 是' if rgb_normalized else '❌ 否'}")
        
        success_rate = valid_count / sample_count * 100
        print(f"  NPZ验证: {'✅ 通过' if success_rate >= 90 else '⚠️ 部分通过' if success_rate >= 50 else '❌ 失败'} ({success_rate:.1f}%验证成功)")
        return success_rate >= 90
    except Exception as e:
        print(f"  ❌ NPZ文件验证失败: {e}")
        return False

def check_jsonl_npz_consistency(jsonl_path):
    """检查JSONL中的NPZ路径和实际NPZ文件是否一致"""
    print(f"\n检查JSONL和NPZ一致性")
    try:
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
        
        jsonl_dir = os.path.dirname(jsonl_path)
        
        # 统计NPZ路径
        npz_paths = []
        for line in lines:
            data = json.loads(line)
            npz_rel_path = data["images"][0]
            npz_paths.append(npz_rel_path)
        
        # 检查重复
        unique_paths = set(npz_paths)
        if len(unique_paths) != len(npz_paths):
            print(f"  ⚠️ 发现重复的NPZ路径: {len(npz_paths) - len(unique_paths)}个重复")
        
        # 检查文件是否存在
        missing = 0
        for path in unique_paths:
            full_path = os.path.join(jsonl_dir, path)
            if not os.path.exists(full_path):
                missing += 1
                if missing <= 5:  # 只显示前5个缺失
                    print(f"  ❌ NPZ文件不存在: {full_path}")
        
        if missing > 5:
            print(f"  ❌ 还有 {missing-5} 个NPZ文件不存在...")
        
        print(f"  一致性检查: {'✅ 通过' if missing == 0 else '⚠️ 部分通过' if missing < len(unique_paths)*0.1 else '❌ 失败'}")
        print(f"  - 总样本数: {len(npz_paths)}")
        print(f"  - 唯一NPZ文件: {len(unique_paths)}")
        print(f"  - 缺失NPZ文件: {missing} ({missing/len(unique_paths)*100:.1f}%)")
        
        return missing == 0
    except Exception as e:
        print(f"  ❌ 一致性检查失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='验证生成的点云数据')
    parser.add_argument('--jsonl_path', type=str, required=True, help='要验证的JSONL文件路径')
    parser.add_argument('--no_norm_check', action='store_true', help='不检查归一化')
    args = parser.parse_args()
    
    print(f"=== 开始验证数据: {args.jsonl_path} ===")
    
    results = []
    results.append(("JSONL格式验证", validate_jsonl_file(args.jsonl_path)))
    results.append(("NPZ文件验证", validate_npz_files(args.jsonl_path, not args.no_norm_check)))
    results.append(("JSONL-NPZ一致性", check_jsonl_npz_consistency(args.jsonl_path)))
    
    print("\n=== 验证结果汇总 ===")
    all_pass = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        all_pass = all_pass and result
        print(f"{name}: {status}")
    
    final_status = "✅ 所有检查通过" if all_pass else "❌ 存在问题需要修复"
    print(f"\n总体结果: {final_status}")

if __name__ == "__main__":
    main()
