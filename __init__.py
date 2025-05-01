from .modeling_intj import MultimodalQwen2Config, MultimodalQwen2ForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

# 注册配置
CONFIG_MAPPING.register("multimodal_qwen2", MultimodalQwen2Config)
# 注册模型
MODEL_FOR_CAUSAL_LM_MAPPING.register(MultimodalQwen2Config, MultimodalQwen2ForCausalLM)
