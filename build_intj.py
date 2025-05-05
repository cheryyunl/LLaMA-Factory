#!/usr/bin/env python3
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from copy import deepcopy
import shutil

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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


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
    
    shutil.copy("modeling_intj.py", os.path.join(output_path, "modeling_intj.py"))
    
    # 创建__init__.py文件
    init_content = """
    from .modeling_intj import MultimodalQwen2Config, MultimodalQwen2ForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    # 注册配置
    CONFIG_MAPPING.register("multimodal_qwen2", MultimodalQwen2Config)
    # 注册模型
    MODEL_FOR_CAUSAL_LM_MAPPING.register(MultimodalQwen2Config, MultimodalQwen2ForCausalLM)
    """

    with open(os.path.join(output_path, "__init__.py"), "w") as f:
        f.write(init_content)
    
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

def test_with_llamafactory_collator(model_path, custom_tokenizer, custom_processor):
    """使用LLaMA Factory的collator测试"""
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    
    # 1. 准备参数
    model_args = ModelArguments(model_name_or_path=model_path)
    data_args = DataArguments(template="qwen2_pointcloud")
    
    # 2. 加载tokenizer和模板
    tokenizer = custom_tokenizer
    processor = custom_processor
    
    # 3. 获取模板
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 4. 获取plugin
    pointcloud_plugin = get_mm_plugin(name="qwen2_pointcloud")
    
    # 5. 创建点云数据
    point_patches = np.random.randn(3, 10, 6).astype(np.float32)  # 3个patch，每个有10个点
    point_coords = np.array([[0,0,0], [0,1,0], [1,0,0]]).astype(np.float32)
    pointcloud_data = PointCloudData(point_patches, point_coords)
    
    raw_messages = [
        {"role": "user", "content": f"Describe this point cloud: {IMAGE_PLACEHOLDER}"}
    ]
    
    # 使用plugin处理消息（这会插入点云标记）
    processed_messages = pointcloud_plugin.process_messages(
        deepcopy(raw_messages), [pointcloud_data], [], [], processor
    )
    
    # 打印处理前后的消息
    print(f"原始消息: {raw_messages[0]['content']}")
    print(f"处理后消息: {processed_messages[0]['content']}")
    
    # 7. 创建collator
    data_collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        template=template,
        processor=processor,
        padding="longest",
        max_length=512,
        pad_to_multiple_of=8
    )
    
    # 8. 创建已处理过的测试数据
    processed_text = processed_messages[0]["content"]
    features = [
        {
            "input_ids": tokenizer.encode(processed_text),
            "attention_mask": [1] * len(tokenizer.encode(processed_text)),
            "labels": tokenizer("This is a set of points in 3D space.").input_ids,
            "images": [pointcloud_data],
            "videos": [],
            "audios": []
        }
    ]
    
    # 9. 测试collator
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
        import traceback
        traceback.print_exc()
        return False

def test_structured_pointcloud():
    """测试结构化点云的处理（2x2x2格式）"""
    print("\n===== 测试结构化点云处理 =====")
    
    # 1. 创建一个2x2x2结构的点云数据
    # z=0层，2x2=4个patch
    # z=1层，2x2=4个patch
    # 总共8个patch
    patches = []
    coords = []
    
    # 创建2x2x2结构的坐标
    for z in range(2):
        for y in range(2):
            for x in range(2):
                # 每个patch有5个点，3个特征
                patch = np.random.rand(5, 3).astype(np.float32)
                patches.append(patch)
                coords.append(np.array([x, y, z]))
    
    # 将列表转换为np数组
    coords = np.array(coords)
    pointcloud_data = PointCloudData(patches, coords)
    
    # 2. 加载tokenizer和插件
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH)
    processor = PointCloudProcessor(tokenizer)
    pointcloud_plugin = get_mm_plugin(name="qwen2_pointcloud")
    
    # 3. 创建测试消息
    test_message = {"role": "user", "content": f"Describe this complex point cloud: {IMAGE_PLACEHOLDER}"}
    
    # 4. 处理消息
    processed_messages = pointcloud_plugin.process_messages(
        [deepcopy(test_message)], [pointcloud_data], [], [], processor
    )
    
    # 5. 打印并分析处理后的消息
    print("原始消息:")
    print(test_message["content"])
    print("\n处理后消息:")
    print(processed_messages[0]["content"])
    
    # 6. 验证所有特殊token的出现
    processed_content = processed_messages[0]["content"]
    special_tokens = ["<pointcloud>", "</pointcloud>", "<layer_sep>", "<row_sep>", "<point_patch>"]
    token_counts = {token: processed_content.count(token) for token in special_tokens}
    
    print("\n特殊token统计:")
    for token, count in token_counts.items():
        print(f"  {token}: {count}次")
    
    # 7. 验证结构（应该有1次<layer_sep>和2次<row_sep>）
    if token_counts["<layer_sep>"] == 1:
        print("✅ 检测到正确的层分隔符数量（1个）")
    else:
        print(f"❌ 层分隔符数量错误: 期望1个，实际{token_counts['<layer_sep>']}个")
    
    # 每层应有一个row_sep(2行中间)，共2层，所以期望值为2
    if token_counts["<row_sep>"] == 2:
        print("✅ 检测到正确的行分隔符数量（2个）")
    else:
        print(f"❌ 行分隔符数量错误: 期望2个，实际{token_counts['<row_sep>']}个")
    
    # 8. 测试token顺序
    tokens = tokenizer.tokenize(processed_content)
    token_ids = tokenizer.encode(processed_content)
    
    print("\n分词结果:")
    print(tokens)
    
    # 9. 处理token和获取mm_inputs
    mm_inputs = pointcloud_plugin.get_mm_inputs(
        [pointcloud_data], [], [], [1], [], [], [token_ids], processor
    )
    
    print("\nget_mm_inputs结果:")
    if "point_patch_indices" in mm_inputs:
        print(f"point_patch_indices形状: {mm_inputs['point_patch_indices'].shape}")
        # 计算非-1索引的数量（应该是8个，与patch数量一致）
        non_negative_indices = (mm_inputs["point_patch_indices"] >= 0).sum().item()
        print(f"检测到的点云patch索引数量: {non_negative_indices}，期望值: {len(patches)}")
        
        if non_negative_indices == len(patches):
            print("✅ 点云patch索引数量正确")
        else:
            print("❌ 点云patch索引数量错误")
    
    if "point_patches" in mm_inputs:
        print(f"point_patches形状: {mm_inputs['point_patches'].shape}")
        # 检查patch数量是否正确
        if mm_inputs["point_patches"].shape[0] == len(patches):
            print("✅ 点云特征数量正确")
        else:
            print("❌ 点云特征数量错误")
    
    return True

def test_text_only_input():
    """测试纯文本输入处理"""
    print("\n===== 测试纯文本输入 =====")
    
    # 1. 加载tokenizer和插件
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH)
    processor = PointCloudProcessor(tokenizer)
    pointcloud_plugin = get_mm_plugin(name="qwen2_pointcloud")
    
    # 2. 创建纯文本消息
    text_message = {"role": "user", "content": "What is the capital of France?"}
    
    # 3. 处理消息
    processed_messages = pointcloud_plugin.process_messages(
        [deepcopy(text_message)], [], [], [], processor
    )
    
    # 4. 验证消息未被修改
    original = text_message["content"]
    processed = processed_messages[0]["content"]
    print(f"原始消息: {original}")
    print(f"处理后消息: {processed}")
    print(f"消息相同: {original == processed}")
    
    # 5. 测试get_mm_inputs
    token_ids = tokenizer.encode(processed)
    mm_inputs = pointcloud_plugin.get_mm_inputs(
        [], [], [], [0], [], [], [token_ids], processor
    )
    
    print(f"get_mm_inputs结果: {mm_inputs}")
    print(f"是否为空字典: {mm_inputs == {}}")
    
    return True

def test_plugin_with_real_data(npz_file_path):
    """测试plugin处理真实点云数据的能力"""
    import numpy as np
    from llamafactory.extras.constants import IMAGE_PLACEHOLDER
    
    print(f"\n===== 测试plugin处理真实点云数据 =====")
    print(f"加载文件: {npz_file_path}")
    
    # 1. 加载npz文件数据
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        print(f"成功加载NPZ文件。可用键: {data.files}")
        
        # 2. 提取点云数据
        if 'patches' in data and 'patch_coords' in data:
            patches = data['patches']
            patch_coords = data['patch_coords']
            print(f"成功找到patches和patch_coords键")
        else:
            # 显示所有可用键
            print(f"警告：未找到'patches'和'patch_coords'，文件中可用键: {data.files}")
            # 尝试常见的替代键名
            for p_key in ['points', 'point_patches', 'cloud_patches']:
                if p_key in data:
                    patches = data[p_key]
                    print(f"使用替代键 '{p_key}' 作为patches")
                    break
            for c_key in ['coords', 'coordinates', 'positions']:
                if c_key in data:
                    patch_coords = data[c_key]
                    print(f"使用替代键 '{c_key}' 作为patch_coords")
                    break
            if 'patches' not in locals() or 'patch_coords' not in locals():
                raise ValueError(f"无法在NPZ文件中找到点云数据")
        
        # 3. 创建PointCloudData对象
        print(f"\n点云数据：")
        print(f"- patches类型: {type(patches)}, 形状: {patches.shape if hasattr(patches, 'shape') else 'unknown'}")
        print(f"- patch_coords类型: {type(patch_coords)}, 形状: {patch_coords.shape if hasattr(patch_coords, 'shape') else 'unknown'}")
        
        pointcloud_data = PointCloudData(patches, patch_coords)
        print(f"\n创建的PointCloudData对象：")
        print(f"- 类型: {type(pointcloud_data)}")
        print(f"- hasattr(patches): {hasattr(pointcloud_data, 'patches')}")
        print(f"- hasattr(patch_coords): {hasattr(pointcloud_data, 'patch_coords')}")
        print(f"- patches数据类型: {type(pointcloud_data.patches)}")
        print(f"- patch_coords数据类型: {type(pointcloud_data.patch_coords)}")
        
        # 4. 检查plugin及其方法
        pointcloud_plugin = get_mm_plugin(name="qwen2_pointcloud")
        print(f"\nPlugin信息：")
        print(f"- Plugin类型: {type(pointcloud_plugin)}")
        print(f"- hasattr(process_messages): {hasattr(pointcloud_plugin, 'process_messages')}")
        print(f"- hasattr(_generate_structured_tokens): {hasattr(pointcloud_plugin, '_generate_structured_tokens')}")
        
        # 5. 测试_generate_structured_tokens方法
        if hasattr(pointcloud_plugin, '_generate_structured_tokens'):
            try:
                tokens = pointcloud_plugin._generate_structured_tokens(pointcloud_data.patch_coords)
                print(f"\n生成的结构化标记: {tokens}")
                patch_count = tokens.count("<point_patch>")
                print(f"标记中的<point_patch>数量: {patch_count}")
            except Exception as e:
                print(f"_generate_structured_tokens失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 6. 测试完整消息处理
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH)
        processor = PointCloudProcessor(tokenizer)
        test_message = {"role": "user", "content": f"Describe this point cloud: {IMAGE_PLACEHOLDER}"}
        
        try:
            processed_messages = pointcloud_plugin.process_messages(
                [deepcopy(test_message)], [pointcloud_data], [], [], processor
            )
            
            print(f"\n处理结果：")
            print(f"原始消息: {test_message['content']}")
            print(f"处理后消息: {processed_messages[0]['content']}")
            
            # 计算patch数量
            processed_content = processed_messages[0]['content']
            patch_count = processed_content.count("<point_patch>")
            print(f"处理后消息中的<point_patch>数量: {patch_count}")
            
            return True
        except Exception as e:
            print(f"\n消息处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"加载或处理文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generate_with_pointcloud(model_path):
    """
    测试模型使用点云数据生成文本
    """
    print("加载模型和tokenizer...")
    model = MultimodalQwen2ForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()  # 设置为评估模式
    
    print("创建测试输入...")
    # 创建提示文本
    prompt = f"{IMAGE_PLACEHOLDER}What is this point cloud?"
    
    # 使用tokenizer处理文本
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 创建随机点云数据（模拟真实点云）
    n_patches = 10  # 点云patch数量
    point_patches = torch.rand(n_patches, 512 * 6)  # (n_patches, 512*6) 每个patch有512个点，每个点6维特征
    
    # 创建点云索引
    # 找出所有点云标记位置
    patch_token_id = tokenizer.convert_tokens_to_ids("<point_patch>")
    point_indices = torch.full_like(inputs["input_ids"], -1, dtype=torch.long)
    
    # 处理特殊情况：tokenizer可能没有<point_patch>标记
    if patch_token_id == tokenizer.unk_token_id:
        print(f"警告: <point_patch>标记不存在，使用手动位置...")
        # 手动找出图像占位符的位置
        text = tokenizer.decode(inputs["input_ids"][0])
        placeholder_pos = text.find(IMAGE_PLACEHOLDER)
        if placeholder_pos != -1:
            # 估计位置并手动设置点云索引
            est_token_pos = len(tokenizer.encode(text[:placeholder_pos]))
            point_indices[0, est_token_pos] = 0  # 使用第一个点云patch
    else:
        # 正常情况：找出所有点云标记位置
        patch_positions = (inputs["input_ids"][0] == patch_token_id).nonzero().squeeze(-1)
        
        # 如果找到点云标记位置
        if len(patch_positions) > 0:
            print(f"找到 {len(patch_positions)} 个点云标记位置")
            for idx, pos in enumerate(patch_positions):
                # 如果标记位置超过了我们的点云数量，循环使用
                point_index = idx % n_patches
                point_indices[0, pos] = point_index
        else:
            print("警告: 未找到点云标记位置，尝试添加默认位置...")
            # 尝试找到一个合理的位置插入点云
            point_indices[0, len(inputs["input_ids"][0]) // 2] = 0  # 使用第一个点云
    
    print("执行文本生成...")
    # 设置生成参数
    gen_kwargs = {
        "max_new_tokens": 100,
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "point_patches": point_patches,
        "point_patch_indices": point_indices
    }
    
    try:
        # 尝试生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n生成的文本:")
        print("="*50)
        print(generated_text)
        print("="*50)
        
        # 验证生成成功
        return True, generated_text
    except Exception as e:
        print(f"生成过程出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def test_forward_with_pointcloud(model_path):
    """
    测试模型前向传播是否正常工作
    """
    print("\n测试模型前向传播...")
    model = MultimodalQwen2ForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    
    # 创建提示文本
    prompt = f"以下是一个3D点云数据：{IMAGE_PLACEHOLDER} 请描述"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 创建随机点云数据
    n_patches = 5
    point_patches = torch.rand(n_patches, 512 * 6)
    
    # 创建点云索引
    point_indices = torch.full_like(inputs["input_ids"], -1, dtype=torch.long)
    patch_token_id = tokenizer.convert_tokens_to_ids("<point_patch>")
    
    # 处理特殊情况
    if patch_token_id == tokenizer.unk_token_id:
        # 模拟位置
        point_indices[0, 5] = 0  # 假设位置5是点云位置
    else:
        patch_positions = (inputs["input_ids"][0] == patch_token_id).nonzero().squeeze(-1)
        if len(patch_positions) > 0:
            for idx, pos in enumerate(patch_positions):
                point_index = idx % n_patches
                point_indices[0, pos] = point_index
    
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                point_patch_indices=point_indices,
                point_patches=point_patches
            )
        
        print(f"前向传播成功！输出logits形状: {outputs.logits.shape}")
        return True
    except Exception as e:
        print(f"前向传播出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 配置路径
    BASE_MODEL_PATH = "saves/qwen2.5_3d/full/sft_7b_stage1_abl_lr"  # 或你本地的Qwen2模型路径
    OUTPUT_PATH = "saves/qwen2.5_3d/full/sft_7b_stage1_lr" 
    
    # 创建模型
    print("创建MultimodalQwen2模型...")
    model, tokenizer, config, processor = create_multimodal_qwen2_model(BASE_MODEL_PATH, OUTPUT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_PATH, trust_remote_code=True)
    processor = PointCloudProcessor(tokenizer)
    print(f"✅ 模型已保存到 {OUTPUT_PATH}")
    
    # 测试模型
    print("\n测试模型处理能力...")
    success = test_model(OUTPUT_PATH)
    
    # 测试LLaMA Factory集成
    if success:
        print("\n测试与LLaMA Factory集成...")
        test_with_llamafactory_collator(OUTPUT_PATH, tokenizer, processor)
    
    if success:
        print("\n测试结构化点云处理...")
        test_structured_pointcloud()
    
    if success:
        print("\n测试纯文本输入...")
        test_text_only_input()

    if success:
        print("\n测试plugin处理真实点云数据...")
        test_plugin_with_real_data("data/objaverse/patches_objects.shard_000/objaverse_000001.npz")

    # 测试生成
    if success:
        gen_success, gen_text = test_generate_with_pointcloud(OUTPUT_PATH)
        if gen_success:
            print("\n✅ 测试成功完成！模型能够使用点云数据生成文本。")
        else:
            print("\n❌ 生成测试失败。")

    # 测试前向传播
    if success:
        forward_success = test_forward_with_pointcloud(OUTPUT_PATH)
        if forward_success:
            print("\n✅ 测试成功完成！模型能够正常前向传播。")
        else:
            print("\n❌ 前向传播测试失败。")
