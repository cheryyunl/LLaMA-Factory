# 从 Qwen2.5-7B-Instruct 初始化多模态模型

import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 添加路径
sys.path.append('/code/LLaMA-Factory')
from modeling_lemon import MultimodalQwen2Config, MultimodalQwen2ForCausalLM

def create_multimodal_from_qwen25():
    """从 Qwen2.5-7B-Instruct 创建多模态模型"""
    
    print("=== 步骤1: 加载 Qwen2.5-7B-Instruct ===")
    qwen25_path = "/code/Qwen2.5-7B-Instruct"
    
    # 加载原始模型和配置
    original_model = AutoModelForCausalLM.from_pretrained(qwen25_path)
    tokenizer = AutoTokenizer.from_pretrained(qwen25_path)
    
    print(f"✅ 原始模型加载完成: {type(original_model).__name__}")
    
    # 检查预留空间
    original_embed_size = original_model.get_input_embeddings().weight.shape[0]  # 152064
    tokenizer_size = len(tokenizer)  # 151643
    reserved_space = original_embed_size - tokenizer_size  # 421个预留位置
    
    print(f"原始embedding大小: {original_embed_size}")
    print(f"Tokenizer大小: {tokenizer_size}")
    print(f"预留空间: {reserved_space} 个位置")
    
    # 🔧 手动分配特殊token ID到预留空间
    special_tokens = ["<pointcloud>", "</pointcloud>", "<point_patch>", "<row_sep>", "<layer_sep>"]
    
    if len(special_tokens) <= reserved_space:
        print(f"✅ 预留空间足够，分配特殊token到位置 {tokenizer_size}-{tokenizer_size + len(special_tokens) - 1}")
        
        # 手动添加token到预留位置
        for i, token in enumerate(special_tokens):
            token_id = tokenizer_size + i
            tokenizer.add_tokens([token])
            print(f"  {token} -> {token_id}")
            
        print(f"新的tokenizer大小: {len(tokenizer)}")
    else:
        print(f"❌ 预留空间不够！需要 {len(special_tokens)}，只有 {reserved_space}")
        return None, None
    
    print("\n=== 步骤2: 创建多模态配置 ===")
    # 🔧 关键：保持原始的vocab_size不变
    multimodal_config = MultimodalQwen2Config.from_pretrained(qwen25_path)
    multimodal_config.point_patch_size = 512
    # 不修改vocab_size，保持152064
    
    print(f"✅ 多模态配置创建完成")
    print(f"保持原始vocab_size: {multimodal_config.vocab_size}")
    
    print("\n=== 步骤3: 创建多模态模型 ===")
    # 创建多模态模型
    multimodal_model = MultimodalQwen2ForCausalLM(multimodal_config)

    print(f"新模型embeddings大小: {multimodal_model.get_input_embeddings().weight.shape}")
    print(f"原始模型embeddings大小: {original_model.get_input_embeddings().weight.shape}")

    print("\n=== 步骤4: 权重迁移 ===")
    # 🔧 先加载原始权重
    original_state_dict = original_model.state_dict()
    multimodal_model.load_state_dict(original_state_dict, strict=False)

    print(f"✅ 权重迁移完成，利用预留空间")

    # 🔧 手动初始化特殊token的embedding和lm_head
    print("手动初始化特殊tokens的embedding和lm_head...")

    with torch.no_grad():
        # 获取现有有效token的权重（排除特殊token区域）
        valid_token_count = tokenizer_size - len(special_tokens)  # 151643 - 5 = 151638
        
        valid_embeddings = multimodal_model.get_input_embeddings().weight[:valid_token_count]
        valid_lm_head = multimodal_model.lm_head.weight[:valid_token_count]
        
        # 计算现有token的统计信息
        embed_mean = valid_embeddings.mean(dim=0)
        embed_std = valid_embeddings.std().item()
        
        lm_head_mean = valid_lm_head.mean(dim=0)  
        lm_head_std = valid_lm_head.std().item()
        
        print(f"现有token统计:")
        print(f"  embedding: 均值norm={embed_mean.norm().item():.4f}, std={embed_std:.6f}")
        print(f"  lm_head:   均值norm={lm_head_mean.norm().item():.4f}, std={lm_head_std:.6f}")
        
        print(f"\n初始化 {len(special_tokens)} 个特殊tokens:")
        
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            
            # 记录初始化前的值
            old_embed_norm = multimodal_model.get_input_embeddings().weight[token_id].norm().item()
            old_lm_head_norm = multimodal_model.lm_head.weight[token_id].norm().item()
            
            # 初始化embedding：使用均值 + 小的随机扰动
            multimodal_model.get_input_embeddings().weight[token_id] = (
                embed_mean + torch.randn_like(embed_mean) * embed_std * 0.1
            )
            
            # 初始化lm_head：使用均值 + 小的随机扰动  
            multimodal_model.lm_head.weight[token_id] = (
                lm_head_mean + torch.randn_like(lm_head_mean) * lm_head_std * 0.1
            )
            
            # 记录初始化后的值
            new_embed_norm = multimodal_model.get_input_embeddings().weight[token_id].norm().item()
            new_lm_head_norm = multimodal_model.lm_head.weight[token_id].norm().item()
            
            print(f"  {token} (ID:{token_id}):")
            print(f"    embedding: {old_embed_norm:.6f} -> {new_embed_norm:.4f}")
            print(f"    lm_head:   {old_lm_head_norm:.6f} -> {new_lm_head_norm:.4f}")

    # 最终验证
    print(f"\n=== 验证特殊token初始化结果 ===")
    all_good = True

    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        embed_norm = multimodal_model.get_input_embeddings().weight[token_id].norm().item()
        lm_head_norm = multimodal_model.lm_head.weight[token_id].norm().item()
        
        # 检查是否合理（不是0，不是异常大值）
        embed_ok = 0.01 < embed_norm < 1.0
        lm_head_ok = 0.01 < lm_head_norm < 1.0
        
        status = "✅" if (embed_ok and lm_head_ok) else "❌"
        print(f"  {status} {token} (ID:{token_id}): embedding={embed_norm:.4f}, lm_head={lm_head_norm:.4f}")
        
        if not (embed_ok and lm_head_ok):
            all_good = False

    if all_good:
        print("✅ 所有特殊token初始化成功！")
    else:
        print("❌ 部分特殊token初始化异常，请检查")

    print("✅ 特殊token手动初始化完成")
    
    print("\n=== 步骤5: 保存多模态模型 ===")
    save_path = "/code/MultimodalQwen2.5-7B-Instruct"
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型和配置
    multimodal_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # 复制 modeling_lemon.py 到模型目录
    import shutil
    shutil.copy('/code/LLaMA-Factory/modeling_lemon.py', os.path.join(save_path, 'modeling_lemon.py'))
    
    # 修改 config.json 添加 auto_map
    import json
    config_path = os.path.join(save_path, 'config.json')
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    
    config_json.update({
        "model_type": "multimodal_qwen2",
        "architectures": ["MultimodalQwen2ForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_lemon.MultimodalQwen2Config",
            "AutoModelForCausalLM": "modeling_lemon.MultimodalQwen2ForCausalLM"
        }
    })
    
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    
    print(f"✅ 模型已保存到: {save_path}")
    print(f"✅ 配置文件已更新")
    
    return save_path, multimodal_model

def test_multimodal_model(model_path):
    """测试多模态模型"""
    
    print(f"\n=== 测试模型: {model_path} ===")
    
    # 加载模型
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"✅ 模型加载成功: {type(model).__name__}")
    print(f"✅ 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查多模态组件
    has_point_patch = hasattr(model.model, 'embed_point_patch')
    print(f"✅ 包含点云嵌入层: {has_point_patch}")
    
    if has_point_patch:
        point_patch_shape = model.model.embed_point_patch.weight.shape
        print(f"✅ 点云嵌入层形状: {point_patch_shape}")
    
    print("\n--- 测试1: 纯文本 Forward ---")
    text = "Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="pt")
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ 文本Forward成功, logits形状: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ 文本Forward失败: {e}")
        return False
    
    print("\n--- 测试2: 纯文本 Generate ---")
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✅ 文本Generate成功")
        print(f"   输入: {text}")
        print(f"   输出: {generated_text}")
    except Exception as e:
        print(f"❌ 文本Generate失败: {e}")
        return False
    
    print("\n--- 测试3: 多模态 Forward ---")
    if has_point_patch:
        batch_size, seq_len = 1, 10
        n_patches = 3
        
        # 模拟多模态输入
        dummy_input_ids = torch.randint(1, 1000, (batch_size, seq_len))  # 避免特殊token
        dummy_attention_mask = torch.ones_like(dummy_input_ids)
        dummy_point_patches = torch.randn(n_patches, 512 * 6)
        dummy_point_patch_indices = torch.full((batch_size, seq_len), -1, dtype=torch.long)
        dummy_point_patch_indices[:, :3] = torch.tensor([0, 1, 2])
        
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask,
                    point_patches=dummy_point_patches,
                    point_patch_indices=dummy_point_patch_indices
                )
            print(f"✅ 多模态Forward成功, logits形状: {outputs.logits.shape}")
        except Exception as e:
            print(f"❌ 多模态Forward失败: {e}")
            return False
        
        print("\n--- 测试4: 多模态 Generate ---")
        try:
            with torch.no_grad():
                generated = model.generate(
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask,
                    point_patches=dummy_point_patches,
                    point_patch_indices=dummy_point_patch_indices,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            print(f"✅ 多模态Generate成功, 输出形状: {generated.shape}")
        except Exception as e:
            print(f"❌ 多模态Generate失败: {e}")
            return False
    
    print("\n🎉 所有测试通过！模型工作正常")
    return True

def main():
    """主函数"""
    print("开始从 Qwen2.5-7B-Instruct 创建多模态模型...")
    
    try:
        # 创建多模态模型
        model_path, model = create_multimodal_from_qwen25()
        
        # 测试模型
        success = test_multimodal_model(model_path)
        
        if success:
            print(f"\n🎉 成功！多模态模型已创建并测试通过")
            print(f"📂 模型保存位置: {model_path}")
            print(f"🚀 可以开始训练了！")
            
            print(f"\n💡 使用方法:")
            print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
            print(f"model = AutoModelForCausalLM.from_pretrained('{model_path}', trust_remote_code=True)")
            print(f"tokenizer = AutoTokenizer.from_pretrained('{model_path}')")
        else:
            print(f"\n❌ 测试失败，请检查错误信息")
            
    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()