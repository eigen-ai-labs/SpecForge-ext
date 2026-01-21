# coding=utf-8
# Copyright 2025 Qwen Team and SpecForge Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen3 EAGLE3 Draft Model Implementation.

This module implements the EAGLE3 draft model using Qwen3's architecture,
which differs from Llama in the following ways:
- QK Norms: RMSNorm applied to Q and K projections before RoPE
- Configurable attention bias
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import Qwen3Config
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from specforge.modeling.draft.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
    generate_eagle3_mask,
)
from specforge.utils import print_with_rank

from .base import Eagle3DraftModel


# Qwen3 uses a simple RMSNorm
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.compile(dynamic=True)
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(dynamic=True)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads to match the number of query heads for GQA.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    """Prepare decoder attention mask."""
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class Qwen3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Qwen3."""
    
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        orig_max_position=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        
        # Llama3 style rotary embedding frequency scaling
        if all(
            v is not None
            for v in [
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                orig_max_position,
            ]
        ):
            print_with_rank(
                f"Using Llama3 style rotary embedding with scaling_factor={scaling_factor}, "
                f"low_freq_factor={low_freq_factor}, high_freq_factor={high_freq_factor}, "
                f"orig_max_position={orig_max_position}"
            )
            self.scaling_factor = scaling_factor
            self.low_freq_factor = low_freq_factor
            self.high_freq_factor = high_freq_factor
            self.orig_max_position = orig_max_position

            low_freq_wavelen = orig_max_position / low_freq_factor
            high_freq_wavelen = orig_max_position / high_freq_factor
            wave_len = 2 * math.pi / inv_freq

            if low_freq_factor != high_freq_factor:
                smooth = (orig_max_position / wave_len - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
            else:
                smooth = 0

            new_freqs = torch.where(
                wave_len < high_freq_wavelen,
                inv_freq,
                torch.where(
                    wave_len > low_freq_wavelen,
                    inv_freq / self.scaling_factor,
                    (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
                ),
            )
            inv_freq = new_freqs
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings + 20,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    @torch.compile(dynamic=True)
    def forward(self, x, seq_len=None):
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class Qwen3MLP(nn.Module):
    """Qwen3 MLP with SwiGLU activation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    """
    Qwen3 Multi-headed attention with QK-Norm.
    
    Key difference from Llama: Qwen3 applies RMSNorm to Q and K after projection,
    before applying rotary embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_bias = getattr(config, "attention_bias", False)

        # For EAGLE3: input is concatenated hidden_states (2x hidden_size)
        self.q_proj = nn.Linear(
            self.hidden_size * 2, self.num_heads * self.head_dim, bias=self.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=self.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=self.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias
        )
        
        # Qwen3's QK-Norm: apply RMSNorm to Q and K on head_dim
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self._init_rope()

    def _init_rope(self):
        if getattr(self.config, "rope_scaling", None) is None:
            self.rotary_emb = Qwen3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(self.config, "rope_theta", 10000),
            )
        else:
            rope_scaling = self.config.rope_scaling

            def rope_get(key, default=None):
                if isinstance(rope_scaling, dict):
                    return rope_scaling.get(key, default)
                return getattr(rope_scaling, key, default)

            scaling_type = rope_get("rope_type", rope_get("type"))
            scaling_factor = rope_get("factor")

            if scaling_type == "llama3":
                self.rotary_emb = Qwen3RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                    scaling_factor=(
                        scaling_factor if scaling_factor is not None else 1.0
                    ),
                    low_freq_factor=rope_get("low_freq_factor"),
                    high_freq_factor=rope_get("high_freq_factor"),
                    orig_max_position=rope_get("original_max_position_embeddings"),
                )
            else:
                # Default to basic rotary embedding for unsupported scaling types
                print_with_rank(
                    f"Warning: rope_scaling type '{scaling_type}' not fully supported, "
                    f"using basic rotary embedding"
                )
                self.rotary_emb = Qwen3RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Qwen3 QK-Norm: apply RMSNorm before transpose
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if cache_hidden is None:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
            )
        else:
            lck = len(cache_hidden[0])
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]

            cache_k = cache_hidden[0]
            cache_v = cache_hidden[1]

            k0 = cache_k[0]
            v0 = cache_v[0]

            # Causal attention with cache
            attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            lck = len(cache_k)
            attn_weights = attn_weights + attention_mask

            for i in range(1, lck):
                ki = cache_k[i]
                attn_weightsi = (query_states * ki).sum(-1) / math.sqrt(self.head_dim)
                attn_weights = torch.cat(
                    (attn_weights, attn_weightsi[..., None]), dim=-1
                )

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights0 = attn_weights[..., :q_len]

            attn_output = torch.matmul(attn_weights0, v0)

            for i in range(1, lck):
                vi = cache_v[i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_outputi = attn_weightsi[..., None] * vi
                attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3FlexAttention(Qwen3Attention):
    """
    Qwen3 Attention with flex attention backend for efficient masking.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Qwen3 QK-Norm: apply RMSNorm before transpose
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        lck = past_seen_tokens // q_len
        cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids + lck
        )

        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + q_len, device=hidden_states.device
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        key_cache, value_cache = past_key_values.update(
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            cache_kwargs=cache_kwargs,
        )

        seq_lengths = attention_mask.sum(dim=-1)
        seq_lengths -= lck
        
        if q_len <= 128:
            create_block_mask_func = create_block_mask
            flex_attention_func = flex_attention
        else:
            create_block_mask_func = compile_friendly_create_block_mask
            flex_attention_func = compile_friendly_flex_attention

        block_mask = create_block_mask_func(
            mask_mod=generate_eagle3_mask(
                seq_lengths=seq_lengths,
                Q_LEN=q_len,
                KV_LEN=key_cache.shape[-2],
                lck=lck,
            ),
            B=bsz,
            H=1,
            Q_LEN=q_len,
            KV_LEN=key_cache.shape[-2],
            device=query_states.device,
        )
        attn_output = flex_attention_func(
            query=query_states,
            key=key_cache.contiguous(),
            value=value_cache.contiguous(),
            block_mask=block_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 Decoder Layer for EAGLE3."""
    
    def __init__(self, config, layer_idx: int = 0, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if attention_backend == "sdpa":
            self.self_attn = Qwen3Attention(config=config)
        elif attention_backend == "flex_attention":
            print_with_rank("Using flex attention on Qwen3 draft model training!")
            self.self_attn = Qwen3FlexAttention(config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown attention backend {attention_backend}")

        self.attention_backend = attention_backend
        self.mlp = Qwen3MLP(config)
        
        # Layer norms for EAGLE3
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Forward pass for EAGLE3 decoder layer.
        
        Args:
            input_emb: Token embeddings
            hidden_states: Hidden states from projection
            cache_hidden: Cache for hidden states (for TTT)
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: KV cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cache
        """
        residual = hidden_states

        # Normalize both inputs
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        # Concatenate for EAGLE3
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        
        # Self Attention
        hidden_states = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3ForCausalLMEagle3(Eagle3DraftModel):
    """
    Qwen3 EAGLE3 Draft Model for speculative decoding.
    
    This model uses Qwen3's architecture with QK-Norm for the draft model
    in EAGLE3 speculative decoding.
    """
    
    config_class = Qwen3Config

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config
        self.num_hidden_layers = config.num_hidden_layers

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        
        # Support multiple decoder layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i, attention_backend=attention_backend)
            for i in range(config.num_hidden_layers)
        ])

        # FC layer to project concatenated hidden states (3 layers from target model)
        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # Create vocab mapping buffers
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        """
        Forward pass for EAGLE3 training.
        
        Args:
            hidden_states: Concatenated hidden states from target model (3 layers)
            inputs_embeds: Token embeddings
            attention_mask: Attention mask
            ttt_length: Test-time training length
        """
        if ttt_length == 1:
            print_with_rank("using ttt_length 1, no need to cache hidden states")
            cache_hidden = None
        else:
            print_with_rank(f"using ttt_length {ttt_length}, caching hidden states")
            cache_hidden = [[], []]

        batch_size, seq_length, _ = hidden_states.size()

        # Create position ids
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        # Project concatenated hidden states
        hidden_states = self.fc(hidden_states)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected_size = getattr(self.config, "target_hidden_size", self.config.hidden_size) * 3
        assert hidden_states.size(-1) == expected_size
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Backbone forward pass through all decoder layers."""
        for layer in self.layers:
            hidden_states = layer(
                input_emb=input_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=False,
            )
        return hidden_states

