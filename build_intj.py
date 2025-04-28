#!/usr/bin/env python3
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from dataclasses import dataclass
from copy import deepcopy

from modeling_intj import MultimodalQwen2Config, MultimodalQwen2ForCausalLM

# 导入LLaMA Factory相关模块
from llamafactory.model.loader import load_tokenizer
from llamafactory.hparams import ModelArguments, DataArguments
from llamafactory.data.mm_plugin import BasePlugin, register_mm_plugin, get_mm_plugin, PLUGINS
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.data.template import register_template, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IMAGE_PLACEHOLDER


# 定义一个点云数据类来方便测试
class PointCloudData:
    def __init__(self, patches, patch_coords):
        self.patches = patches
        self.patch_coords = patch_coords


class PointCloudProcessor:
    """简单的点云处理器，包含tokenizer作为属性"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 可以添加其他点云处理相关功能


def create_multimodal_qwen2_model(base_model_path, output_path):
    """创建并保存MultimodalQwen2模型"""
    # 1. 设置参数
    model_args = ModelArguments(
        model_name_or_path=base_model_path,
        add_special_tokens="<pointcloud>,<point_patch>,<layer_sep>,<row_sep>,</pointcloud>",
        resize_vocab=True
    )
    
    # 2. 加载tokenizer并添加特殊token
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 3. 创建配置 - 修改这部分
    # 先加载配置
    config = AutoConfig.from_pretrained(base_model_path)
    # 转换为字典
    config_dict = config.to_dict()
    config_dict["architectures"] = ["MultimodalQwen2ForCausalLM"]
    config_dict["point_patch_size"] = 512  # 点云patch大小
    
    # 4. 创建多模态配置和模型
    config = MultimodalQwen2Config.from_dict(config_dict)
    config.vocab_size = len(tokenizer)  # 更新词表大小
    
    # 5. 初始化模型
    model = MultimodalQwen2ForCausalLM(config)
    
    # 6. 调整embeddings大小以匹配tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # 7. 创建processor
    processor = PointCloudProcessor(tokenizer)
    
    # 8. 保存模型和tokenizer
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    config.save_pretrained(output_path)
    
    # 保存额外信息（可选）
    with open(os.path.join(output_path, "processor_info.json"), "w") as f:
        import json
        json.dump({"type": "PointCloudProcessor"}, f)
    
    return model, tokenizer, config, processor

def test_model(model_path):
    """测试模型能否处理点云输入"""
    # 1. 加载模型和tokenizer
    model = MultimodalQwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. 创建模拟输入
    prompt = f"This is a point cloud: {IMAGE_PLACEHOLDER} Describe what you see."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 3. 创建点云数据和索引
    # 创建一个随机点云 (100点, 每点6维特征)
    point_cloud = torch.rand(100, 512 * 6)  # 根据模型配置的point_patch_size
    
    # 4. 创建point_patch_indices - 修复这部分
    seq_length = inputs["input_ids"].shape[1]  # 获取实际序列长度
    patch_token_id = tokenizer.convert_tokens_to_ids("<point_patch>")
    
    # 创建与输入相同大小的全-1张量
    point_indices = torch.full_like(inputs["input_ids"], -1, dtype=torch.long)
    
    # 找出所有点云标记的位置
    patch_positions = (inputs["input_ids"][0] == patch_token_id).nonzero().squeeze(-1)
    
    # 如果找到点云标记
    if len(patch_positions) > 0:
        # 为简单起见，我们假设所有patch_positions使用相同的点云
        # 实际应用中可能需要更复杂的映射
        for idx, pos in enumerate(patch_positions):
            # 最多使用100个点云点
            point_index = idx % 100
            point_indices[0, pos] = point_index
    
    # 5. 模型前向传播
    try:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            point_patch_indices=point_indices,
            point_patches=point_cloud
        )
        print("模型前向传播成功!")
        print(f"输出logits形状: {outputs.logits.shape}")
        print(f"输入序列长度: {seq_length}")
        print(f"点云索引形状: {point_indices.shape}")
        return True
    except Exception as e:
        print(f"模型前向传播失败: {e}")
        return False

def test_with_llamafactory_collator(model_path):
    """使用LLaMA Factory的collator测试"""
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    
    # 1. 准备参数
    model_args = ModelArguments(model_name_or_path=model_path)
    data_args = DataArguments(template="qwen2_pointcloud")
    
    # 2. 加载tokenizer和模板
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 3. 创建collator
    data_collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        template=template,
        processor=processor,
        padding="longest",
        max_length=512,
        pad_to_multiple_of=8
    )
    
    # 4. 创建测试数据
    # 使用模拟的点云数据
    point_patches = np.random.randn(3, 10, 6).astype(np.float32)  # 3个patch，每个10个点
    point_coords = np.array([[0,0,0], [0,1,0], [1,0,0]]).astype(np.float32)
    pointcloud_data = PointCloudData(point_patches, point_coords)
    
    features = [
        {
            "input_ids": tokenizer(f"Describe this point cloud: {IMAGE_PLACEHOLDER}").input_ids,
            "attention_mask": [1] * len(tokenizer(f"Describe this point cloud: {IMAGE_PLACEHOLDER}").input_ids),
            "labels": tokenizer("This is a cube.").input_ids,
            "images": [pointcloud_data.patches],
            "videos": [],
            "audios": []
        }
    ]
    
    # 5. 测试collator
    try:
        batch = data_collator(features)
        print("Collator处理成功!")
        print(f"批处理键: {batch.keys()}")
        if "point_patches" in batch and "point_patch_indices" in batch:
            print(f"点云特征形状: {batch['point_patches'].shape}")
            print(f"点云索引形状: {batch['point_patch_indices'].shape}")
            return True
        else:
            print("缺少点云相关特征!")
            return False
    except Exception as e:
        print(f"Collator处理失败: {e}")
        return False


if __name__ == "__main__":
    # 配置路径
    BASE_MODEL_PATH = "/pscratch/sd/c/cheryunl/qwen2_0.5b_cache"  # 或你本地的Qwen2模型路径
    OUTPUT_PATH = "./multimodal_qwen2_model" 
    
    # 创建模型
    print("创建MultimodalQwen2模型...")
    model, tokenizer, config, processor = create_multimodal_qwen2_model(BASE_MODEL_PATH, OUTPUT_PATH)
    print(f"✅ 模型已保存到 {OUTPUT_PATH}")
    
    # 测试模型
    print("\n测试模型处理能力...")
    success = test_model(OUTPUT_PATH)
    
    # 测试LLaMA Factory集成
    if success:
        print("\n测试与LLaMA Factory集成...")
        test_with_llamafactory_collator(OUTPUT_PATH)