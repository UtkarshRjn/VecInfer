from ...utils.Injector import Injector
from ...utils.Namespace import UniConfig
from ...utils.ContextList import ContextList
from ...utils.Timer import Timer
from ...utils.pq_utils import DynamicPQCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    logger,
    apply_rotary_pos_emb,
    repeat_kv,
    Cache
)

import math
import torch
from typing import Optional, Tuple
from .ModelContext import ModelContext
from ..Errors import SamplingComplete
from fast_hadamard_transform import hadamard_transform

def save_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # save to disk
    config = UniConfig()
    root = config.sample_root
    threshold = config.threshold

    import os
    import random 
    os.makedirs(root, exist_ok=True)
    from ...utils.fvecio import write_fvecs
    
    scale = config.scales[f"model.layers.{self.layer_idx}.self_attn.k_proj"]
    key_states = key_states/scale
    query_states = query_states*repeat_kv(scale, self.num_key_value_groups)
    
    hadamard_dtype = torch.float16
    key_states = hadamard_transform(key_states.float(), scale=1/math.sqrt(key_states.shape[-1])).to(hadamard_dtype) 
    query_states = hadamard_transform(query_states.float(), scale=1/math.sqrt(query_states.shape[-1])).to(hadamard_dtype)  

    bs, nh, _, head_size = key_states.shape

    for b in range(bs):
        for h in range(nh):
            if random.random() > threshold:
                continue
            key_ = key_states[b, h].view(-1, head_size).cpu().detach().numpy()
            
            write_fvecs(config.sample_root / f'key_sampled_layer{self.layer_idx}_{config.M_key}_{config.nbits_key}.fvecs', key_, 'ab')
            del key_
            
            value_ = value_states[b, h].view(-1, head_size).cpu().detach().numpy()
            
            write_fvecs(config.sample_root / f'value_sampled_layer{self.layer_idx}_{config.M_value}_{config.nbits_value}.fvecs', value_, 'ab')
            del value_
            if self.layer_idx == 0: 
                config.sampled_nums += 1
            if config.sampled_nums > config.expected_sample_nums:
                raise SamplingComplete(f"Sampled {config.sampled_nums} vectors, expected {config.expected_sample_nums}, stopping...")

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.config.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None

def prepare_inputs_for_generation(
    self: LlamaForCausalLM,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=True,
    **kwargs,
):
    past_length = 0
    if DynamicPQCache.has_instance():
        cache = DynamicPQCache()
        past_length = cache.seen_tokens[0]
        cache_length = past_length
        max_cache_length = None


        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if DynamicPQCache.has_instance():
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        # TODO: use `next_tokens` directly instead.
        model_inputs = {"input_ids": input_ids.contiguous()}

    input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    if cache_position is None:
        cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
    elif use_cache:
        cache_position = cache_position[-input_length:]

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    
    distort_recent = UniConfig().distort_recent
    config = UniConfig()
        
    with Timer("LlamaSdpaAttention.forward"):
        with Timer("qkv_proj"):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            

            torch.cuda.synchronize()

        with Timer("rotary_emb"):
            if position_embeddings is None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                
            scales = torch.load(config.scales_path, map_location=key_states.device)[f"model.layers.{self.layer_idx}.self_attn.k_proj"]

            key_states.div_(scales)
            query_states.mul_(repeat_kv(scales, self.num_key_value_groups))

            hadamard_dtype = torch.float16
            key_states = hadamard_transform(key_states.float(), scale=1/math.sqrt(key_states.shape[-1])).to(hadamard_dtype) 
            query_states = hadamard_transform(query_states.float(), scale=1/math.sqrt(query_states.shape[-1])).to(hadamard_dtype)  

            torch.cuda.synchronize()
        with Timer("LlamaSdpaAttention.update_cache"):
            if use_cache:
                assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
                cache = DynamicPQCache()
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx, distort_recent=distort_recent)
            torch.cuda.synchronize()
            
        with Timer("LlamaSdpaAttention.repeat_kv"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.causal_mask"):
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.contiguous"):
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.sdpa"):
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
            # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.contiguous"):
            attn_output = attn_output.transpose(1, 2).contiguous()
            torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.o_proj"):
            attn_output = attn_output.view(bsz, q_len, self.config.hidden_size)
            attn_output = self.o_proj(attn_output)
            torch.cuda.synchronize()

        return attn_output, None

def baseline_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()
    Timer.disabled = True if q_len != 1 else False
    breakdown = UniConfig().breakdown
    with Timer("LlamaSdpaAttention.forward"):
        with Timer("LlamaSdpaAttention.forward.qkv_proj"):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.rotary_emb"):
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.cat"):    
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.repeat_kv"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.causal_mask"):
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            if breakdown is True: torch.cuda.synchronize()
        
        with Timer("LlamaSdpaAttention.forward.contiguous"):
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.sdpa"):
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.o_proj"):
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)

            attn_output = self.o_proj(attn_output)
            if breakdown is True: torch.cuda.synchronize()

    return attn_output, None

def attn_forward_custom_kernel(      
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
    
    bsz, q_len, _ = hidden_states.size()
    Timer.disabled = True if q_len != 1 else False
    breakdown = UniConfig().breakdown
    distort_recent = UniConfig().distort_recent



    with Timer("LlamaSdpaAttention.forward"):
        with Timer("LlamaSdpaAttention.forward.qkv_proj"):

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            if breakdown is True: torch.cuda.synchronize()

        with Timer("LlamaSdpaAttention.forward.rotary_emb"):
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if breakdown is True: torch.cuda.synchronize()
        
        assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
        cache = DynamicPQCache()


        if q_len > 1:
            attn_output = cache.prefill(
                query_states, key_states, value_states, self.layer_idx, distort_recent=distort_recent
            )
        else:
            attn_output = cache.decoding(
                query_states, key_states, value_states, self.layer_idx
            )

        with Timer("LlamaSdpaAttention.forward.o_proj"):
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.config.hidden_size)
            attn_output = self.o_proj(attn_output)
            if breakdown is True: torch.cuda.synchronize()

    return attn_output, None

model_context = ModelContext(
    init_context=ContextList([]),
    baseline_context=ContextList([
        Injector(LlamaAttention, 'forward', baseline_forward)
    ]),
    sampling_context=ContextList([
        Injector(LlamaAttention, 'forward', save_forward)
    ]),
    evaluation_context=ContextList([ # Currently we only dispatch to custom kernel when nbits=8.
        Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation),
        Injector(LlamaAttention, 'forward', attn_forward_custom_kernel)
        # Injector(LlamaAttention, 'forward', attn_forward)
    ]) if (UniConfig().nbits_key==8 and UniConfig().nbits_value==8) else ContextList([ # Otherwise, we use torch implementation to do PQ.
        Injector(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation),
        Injector(LlamaAttention, 'forward', attn_forward)
    ])
)