#!/usr/bin/env python3
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from dataclasses import dataclass

from modeling_intj import MultimodalQwen2Config, MultimodalQwen2ForCausalLM

# 导入LLaMA Factory相关模块
from llamafactory.model.loader import load_tokenizer
from llamafactory.hparams import ModelArguments
from llamafactory.data.mm_plugin import BasePlugin, register_mm_plugin
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.extras.constants import IMAGE_PLACEHOLDER

# 定义点云占位符常量
POINTCLOUD_PLACEHOLDER = "<pointcloud>"

@dataclass
class Qwen2PointCloudPlugin(BasePlugin):
    """处理点云数据的插件"""
    def __init__(self, image_token=None, video_token=None, audio_token=None, point_token="<point_patch>"):
        super().__init__(image_token, video_token, audio_token)
        self.point_token = point_token
    
    def _regularize_pointclouds(self, pointclouds, **kwargs):
        """处理点云数据为模型所需格式"""
        results = []
        for pc in pointclouds:
            # 这里简化处理，实际应用中需要实现点云预处理
            if isinstance(pc, np.ndarray):
                # 假设输入是numpy数组，形状为(n_points, 6)或其他
                processed = pc
            else:
                # 加载点云文件等
                processed = np.random.randn(100, 6)  # 用随机数据模拟
            
            # 确保数据是float32类型
            processed = processed.astype(np.float32)
            results.append(processed)
        return {"pointclouds": results}
    
    def process_messages(self, messages, images, videos, audios, processor):
        """处理包含点云标记的消息"""
        self._validate_input(processor, images, videos, audios)
        messages = super().process_messages(messages, images, videos, audios, processor)
        
        # 这里可以添加点云特定的处理
        # 但目前我们的核心功能会在get_mm_inputs实现
        return messages
    
    def get_mm_inputs(self, pointclouds, pclens, *args, **kwargs):
        """生成点云特征和索引"""
        if not pointclouds:
            return {}
            
        # 处理点云数据
        pc_data = self._regularize_pointclouds(pointclouds)["pointclouds"]
        
        # 合并所有点云patches
        all_patches = np.vstack([pc for pc in pc_data])
        
        # 假设我们需要构建point_patch_indices
        # 这部分在实际应用中需要根据tokenized文本来构建
        # 这里简化处理，只是演示
        patch_indices = [[-1] * 100 for _ in range(len(pclens))]
        
        # 转换为PyTorch张量
        point_patches = torch.tensor(all_patches, dtype=torch.float32)
        point_patch_indices = torch.tensor(patch_indices, dtype=torch.long)
        
        return {
            "point_patches": point_patches,
            "point_patch_indices": point_patch_indices
        }

# 注册点云插件
register_mm_plugin("qwen2_pointcloud", Qwen2PointCloudPlugin)

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
    
    # 3. 创建配置
    config_dict = AutoConfig.get_config_dict(base_model_path)[0]
    config_dict["architectures"] = ["MultimodalQwen2ForCausalLM"]
    config_dict["point_patch_size"] = 512  # 点云patch大小
    
    # 4. 创建多模态配置和模型
    config = MultimodalQwen2Config.from_dict(config_dict)
    config.vocab_size = len(tokenizer)  # 更新词表大小
    
    # 5. 初始化模型
    model = MultimodalQwen2ForCausalLM(config)
    
    # 6. 调整embeddings大小以匹配tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # 7. 保存模型和tokenizer
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    config.save_pretrained(output_path)
    
    return model, tokenizer, config

def test_model(model_path):
    """测试模型能否处理点云输入"""
    # 1. 加载模型和tokenizer
    model = MultimodalQwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. 创建模拟输入
    prompt = f"This is a point cloud: {POINTCLOUD_PLACEHOLDER} Describe what you see."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 3. 创建点云数据和索引
    # 创建一个随机点云 (100点, 每点6维特征)
    point_cloud = torch.rand(100, 512 * 6)  # 根据模型配置的point_patch_size
    
    # 4. 创建point_patch_indices
    # 找到<point_patch>标记的位置
    patch_token_id = tokenizer.convert_tokens_to_ids("<point_patch>")
    point_indices = torch.where(
        inputs["input_ids"][0] == patch_token_id,
        torch.arange(100),  # 点云索引
        torch.tensor(-1)  # 文本标记
    ).unsqueeze(0)
    
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
    point_data = np.random.randn(100, 6).astype(np.float32)
    
    features = [
        {
            "input_ids": tokenizer(f"Describe this point cloud: {POINTCLOUD_PLACEHOLDER}").input_ids,
            "attention_mask": [1] * len(tokenizer(f"Describe this point cloud: {POINTCLOUD_PLACEHOLDER}").input_ids),
            "labels": tokenizer("This is a cube.").input_ids,
            "pointclouds": [point_data],
            "images": [],
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
    BASE_MODEL_PATH = "Qwen/Qwen2.5-7B"  # 或你本地的Qwen2模型路径
    OUTPUT_PATH = "./multimodal_qwen2_model" 
    
    # 创建模型
    print("创建MultimodalQwen2模型...")
    model, tokenizer, config = create_multimodal_qwen2_model(BASE_MODEL_PATH, OUTPUT_PATH)
    print(f"模型已保存到 {OUTPUT_PATH}")
    
    # 测试模型
    print("\n测试模型处理能力...")
    success = test_model(OUTPUT_PATH)
    
    # 测试LLaMA Factory集成
    if success:
        print("\n测试与LLaMA Factory集成...")
        test_with_llamafactory_collator(OUTPUT_PATH)