from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2PreTrainedModel,
    Qwen2Model,
    Qwen2ForCausalLM
)

class MultimodalQwen2Config(Qwen2Config):
    """
    Configuration class for MultimodalQwen2Model.
    Extends Qwen2Config to include multimodal specific parameters.
    """
    def __init__(
        self,
        point_patch_size=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.point_patch_size = point_patch_size


class MultimodalQwen2Model(Qwen2Model):
    """
    Multimodal Qwen2 model that supports point cloud data alongside text input.
    """
    def __init__(self, config: MultimodalQwen2Config):
        super().__init__(config)
        
        # Point cloud patch embedding layer
        self.embed_point_patch = nn.Linear(
            config.point_patch_size * 6,  # 512 * 6
            config.hidden_size,
            bias=False
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        point_patch_indices: torch.LongTensor = None,  # (batch_size, seq_length), "-1" for text token
        point_patches: torch.FloatTensor = None,  # (n_patches, point_patch_size * 6)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
            # Process point cloud patches
            if point_patches is not None and point_patch_indices is not None:
                # Validate input shape consistency
                assert point_patch_indices.shape == input_ids.shape, \
                    "point_patch_indices and input_ids should have the same shape"
                
                # Embed point cloud patches
                point_embeds = self.embed_point_patch(point_patches)  # (n_patches, hidden_size)
                
                # Add dummy token for text (index -1)
                point_embeds = torch.cat([
                    point_embeds, 
                    torch.zeros(1, self.config.hidden_size).to(point_embeds.device)
                ])  # (n_patches + 1, hidden_size)
                
                # Arrange embeddings according to point_patch_indices
                point_embeds = point_embeds[point_patch_indices]  # (batch_size, seq_length, hidden_size)
                
                # Merge point cloud embeddings with text embeddings
                inputs_embeds = inputs_embeds + point_embeds

        # Call parent's forward method to handle the rest
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )


class MultimodalQwen2ForCausalLM(Qwen2ForCausalLM):
    """
    Multimodal Qwen2 model for causal language modeling.
    Supports both text and point cloud inputs.
    """
    def __init__(self, config: MultimodalQwen2Config):
        super().__init__(config)
        # Replace base model with multimodal model
        self.model = MultimodalQwen2Model(config)
        # Re-apply initialization
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        point_patch_indices: torch.LongTensor = None,
        point_patches: torch.FloatTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
  
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call multimodal model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_patch_indices=point_patch_indices,
            point_patches=point_patches,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )
  
        hidden_states = outputs[0]
        # Only compute necessary logits based on logits_to_keep
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Use Qwen's loss calculation function
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )