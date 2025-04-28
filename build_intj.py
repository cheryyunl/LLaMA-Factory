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
    config = AutoConfig.from_pretrained(base_model_path)
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
    
    # 7. 保存模型和tokenizer
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    config.save_pretrained(output_path)
    
    return model, tokenizer, config


def register_qwen2_pointcloud_template():
    """注册点云模板"""
    # 确保使用我们已经定义好的plugin
    if "qwen2_pointcloud" in PLUGINS:
        class Qwen2PointCloudTemplate:
            def __init__(self, mm_plugin=None, system=None):
                self.mm_plugin = mm_plugin
                self.default_system = system
                
            def format_user(self, message):
                return f"USER: {message} ASSISTANT:"
                
            def format_assistant(self, message):
                return f" {message}"
                
            def format_system(self, message):
                return f"SYSTEM: {message}\n"
                
            def encode_multiturn(self, tokenizer, messages, system=None, tools=None):
                system = system or self.default_system
                result = ""
                if system:
                    result += self.format_system(system)
                for message in messages:
                    if message["role"] == "user":
                        result += self.format_user(message["content"])
                    else:
                        result += self.format_assistant(message["content"])
                return tokenizer.encode(result, add_special_tokens=True)
        
        # 注册模板
        plugin = get_mm_plugin(name="qwen2_pointcloud")
        return register_template(
            name="qwen2_pointcloud",
            template_class=Qwen2PointCloudTemplate,
            mm_plugin=plugin
        )
    else:
        raise ValueError("qwen2_pointcloud plugin未定义，请先定义plugin")


def test_model(model_path):
    """测试模型能否处理点云输入"""
    print("\n===== 测试模型基本功能 =====")
    # 1. 加载模型和tokenizer
    model = MultimodalQwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. 创建模拟输入
    prompt = f"This is a point cloud: {IMAGE_PLACEHOLDER} Describe what you see."
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"输入ID: {inputs['input_ids'][0].tolist()}")
    print(f"模拟提示: {prompt}")
    print(f"tokenizer长度: {len(tokenizer)}")
    
    # 3. 打印特殊token的ID
    print(f"特殊token ID:")
    special_tokens = ["<pointcloud>", "</pointcloud>", "<layer_sep>", "<row_sep>", "<point_patch>"]
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token} -> {token_id}")
    
    # 4. 创建点云数据 (100个点，每个点6维特征)
    num_points = 100
    point_features = 6
    point_cloud = torch.rand(num_points, point_features * 512)  # 512是point_patch_size
    
    # 5. 创建point_patch_indices
    seq_length = inputs["input_ids"].shape[1]
    point_patch_id = tokenizer.convert_tokens_to_ids("<point_patch>")
    
    # 创建全-1张量
    point_indices = torch.full_like(inputs["input_ids"], -1, dtype=torch.long)
    
    # 找出<point_patch>标记的位置
    input_ids = inputs["input_ids"][0].tolist()
    patch_positions = [i for i, id in enumerate(input_ids) if id == point_patch_id]
    print(f"点云token位置: {patch_positions}")
    
    # 为每个<point_patch>位置分配点云索引
    for idx, pos in enumerate(patch_positions):
        if idx < num_points:
            point_indices[0, pos] = idx
    
    # 6. 模型前向传播
    try:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            point_patch_indices=point_indices,
            point_patches=point_cloud
        )
        print("✅ 模型前向传播成功!")
        print(f"输出logits形状: {outputs.logits.shape}")
        print(f"输入序列长度: {seq_length}")
        print(f"点云索引形状: {point_indices.shape}")
        return True
    except Exception as e:
        print(f"❌ 模型前向传播失败: {e}")
        return False


def test_plugin_directly():
    """直接测试plugin功能"""
    print("\n===== 直接测试Plugin功能 =====")
    # 1. 创建测试数据
    patches = np.random.rand(3, 10, 6).astype(np.float32)  # 3个patch，每个有10个点，6个特征
    patch_coords = np.array([[0,0,0], [0,1,0], [1,0,0]])   # 3个patch的坐标
    pointcloud_data = PointCloudData(patches, patch_coords)
    
    # 2. 加载tokenizer
    model_args = ModelArguments(
        model_name_or_path=OUTPUT_PATH,
        add_special_tokens="<pointcloud>,<point_patch>,<layer_sep>,<row_sep>,</pointcloud>",
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    
    # 3. 获取plugin
    pointcloud_plugin = get_mm_plugin(name="qwen2_pointcloud")
    
    # 4. 测试process_messages
    test_messages = [
        {"role": "user", "content": f"Here is a pointcloud: {IMAGE_PLACEHOLDER} Describe it."},
        {"role": "assistant", "content": "I see points in 3D space."},
    ]
    
    processed_messages = pointcloud_plugin.process_messages(
        deepcopy(test_messages), [pointcloud_data], [], [], processor
    )
    
    print("原始消息:")
    print(test_messages[0]["content"])
    print("\n处理后消息:")
    print(processed_messages[0]["content"])
    
    # 5. 测试tokenization
    tokens = tokenizer.tokenize(processed_messages[0]["content"])
    print("\n分词结果:")
    print(tokens)
    
    # 6. 测试_regularize_images
    reg_results = pointcloud_plugin._regularize_images([pointcloud_data])
    print("\n_regularize_images结果:")
    print(f"patch数量: {len(reg_results['point_patches'][0])}")
    
    # 7. 测试get_mm_inputs
    # 模拟batch_ids
    token_ids = tokenizer.encode(processed_messages[0]["content"])
    point_patch_id = tokenizer.convert_tokens_to_ids("<point_patch>")
    
    mm_inputs = pointcloud_plugin.get_mm_inputs(
        [pointcloud_data], [], [], [1], [], [], [token_ids], processor
    )
    
    print("\nget_mm_inputs结果:")
    print(f"point_patch_indices形状: {mm_inputs['point_patch_indices'].shape}")
    if 'point_patches' in mm_inputs:
        print(f"point_patches形状: {mm_inputs['point_patches'].shape}")
    
    return True


def test_with_llamafactory_collator(model_path):
    """使用LLaMA Factory的collator测试"""
    print("\n===== 测试与LLaMA Factory集成 =====")
    
    # 0. 注册模板
    try:
        template_name = register_qwen2_pointcloud_template()
        print(f"✅ 成功注册模板: {template_name}")
    except Exception as e:
        print(f"❌ 模板注册失败: {e}")
        return False
    
    # 1. 准备参数
    model_args = ModelArguments(model_name_or_path=model_path)
    data_args = DataArguments(template=template_name)
    
    # 2. 加载tokenizer和模板
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    
    try:
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        print(f"✅ 成功加载模板")
    except Exception as e:
        print(f"❌ 模板加载失败: {e}")
        return False
    
    # 3. 创建collator
    try:
        data_collator = MultiModalDataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            template=template,
            processor=processor,
            padding="longest",
            max_length=512,
            pad_to_multiple_of=8
        )
        print(f"✅ 成功创建collator")
    except Exception as e:
        print(f"❌ collator创建失败: {e}")
        return False
    
    # 4. 创建测试数据
    # 创建点云数据
    pointcloud_patches = np.random.randn(3, 10, 6).astype(np.float32)  # 3个patch，每个有10个点
    pointcloud_coords = np.array([[0,0,0], [0,1,0], [1,0,0]]).astype(np.float32)  # 3个patch坐标
    pointcloud_data = PointCloudData(pointcloud_patches, pointcloud_coords)
    
    # 组装特征
    features = [
        {
            "input_ids": tokenizer(f"USER: Describe this point cloud: {IMAGE_PLACEHOLDER} ASSISTANT:").input_ids,
            "attention_mask": [1] * len(tokenizer(f"USER: Describe this point cloud: {IMAGE_PLACEHOLDER} ASSISTANT:").input_ids),
            "labels": tokenizer(" This is a set of points in 3D space.").input_ids,
            "images": [pointcloud_data],  # 使用images字段传递点云数据
            "videos": [],
            "audios": []
        }
    ]
    
    # 5. 测试collator
    try:
        batch = data_collator(features)
        print("✅ Collator处理成功!")
        print(f"批处理键: {batch.keys()}")
        
        if "point_patch_indices" in batch:
            print(f"点云索引形状: {batch['point_patch_indices'].shape}")
            
        if "point_patches" in batch:
            print(f"点云特征形状: {batch['point_patches'].shape}")
            return True
        else:
            print("⚠️ 警告: 缺少点云特征!")
            return False
    except Exception as e:
        print(f"❌ Collator处理失败: {e}")
        print("错误详情:", str(e))
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 配置路径
    BASE_MODEL_PATH = "/pscratch/sd/c/cheryunl/qwen2_0.5b_cache"  # 或你本地的Qwen2模型路径
    OUTPUT_PATH = "./multimodal_qwen2_model" 
    
    # 创建模型
    print("创建MultimodalQwen2模型...")
    model, tokenizer, config = create_multimodal_qwen2_model(BASE_MODEL_PATH, OUTPUT_PATH)
    print(f"✅ 模型已保存到 {OUTPUT_PATH}")
    
    # 测试模型
    print("\n测试模型处理能力...")
    test_model_success = test_model(OUTPUT_PATH)
    
    # 直接测试plugin功能
    print("\n直接测试Plugin...")
    test_plugin_success = test_plugin_directly()
    
    # 测试LLaMA Factory集成
    if test_model_success and test_plugin_success:
        print("\n测试与LLaMA Factory集成...")
        collator_success = test_with_llamafactory_collator(OUTPUT_PATH)
        
        if collator_success:
            print("\n🎉 所有测试通过! 点云模型和Plugin运行正常。")
        else:
            print("\n⚠️ LLaMA Factory集成测试失败，但基本功能正常。")
    else:
        print("\n❌ 基本测试失败，请修复基础问题。")