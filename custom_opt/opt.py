import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import DynamicCache, Cache


class OPTAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, layer_idx=None):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # causal_mask 미리 생성 (2048은 max_position_embeddings 기준)
        self.register_buffer("causal_mask", torch.tril(torch.ones(2048, 2048)), persistent=False)

    def forward(self, x, attention_mask=None, past_key_values: Optional[Tuple[torch.Tensor]] = None, cache_position = None):

        x = x.reshape(-1, x.size()[-2], x.size()[-1])
        B, T, D = x.size()  # [batch, seq_len, hidden_dim]

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # print(f"self.layer_idx in attention: {self.layer_idx}")
        if past_key_values is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            k, v = past_key_values.update(
                k, v, self.layer_idx, {"cache_position": cache_position}
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal_mask = attention_mask
        # if attention_mask is not None:
        #     causal_mask = causal_mask[:, :, :, : k.shape[-2]]

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        
        is_causal = True if causal_mask is None and T > 1 else False

        dropout = 0
        if self.training:
            dropout = 0.1

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_values

class OPTDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, n_heads, dropout=0.1, layer_idx=None):
        super().__init__()
        self.self_attn = OPTAttention(hidden_dim, n_heads,layer_idx=layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = dropout
        self.do_layer_norm_before = True
        self.layer_idx = layer_idx

    def forward(self, x, attention_mask=None, use_cache=False, cache_position=None, past_key_values = None):
        residual = x
        if self.do_layer_norm_before:
            x = self.self_attn_layer_norm(x)

        x, present_key_value = self.self_attn(x, attention_mask=attention_mask, past_key_values = past_key_values)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if not self.do_layer_norm_before:
            x = self.self_attn_layer_norm(x)

        # Huggingface OPT 참조
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        residual = x

        if self.do_layer_norm_before:
            x = self.final_layer_norm(x)

        x = self.fc1(x)
        x = self.activation_fn(x)

        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = (residual + x).view(x_shape)
        
        if not self.do_layer_norm_before:
            x = self.final_layer_norm(x)
        
        outputs = (x,)

        if use_cache:
            outputs += (present_key_value,)
        # print(f"outputs in decoder: {outputs}")
        return outputs

class CustomOPTModel(nn.Module):
    def __init__(self, vocab_size=50272, max_position=2048, hidden_dim=768, ffn_dim=3072, n_layers=12, n_heads=12, dropout=0.1, device="cpu"):
        super().__init__()
        self.device = device

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim, padding_idx=1).to(device)
        self.embed_positions = OPTLearnedPositionalEmbedding(max_position, hidden_dim).to(device)

        self.layers = nn.ModuleList([
            OPTDecoderLayer(hidden_dim, ffn_dim, n_heads, dropout=dropout, layer_idx=i).to(device) for i in range(n_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(hidden_dim).to(device)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False).to(device)
        
    def make_mask(self, input_ids):
        padding_idx = 1
        attention_mask = (input_ids != padding_idx).to(dtype=torch.float32, device=input_ids.device)  # [B, T]
        attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, T]
        attention_mask = (1.0 - attention_mask) * -1e9
        return attention_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # if self.config._attn_implementation == "flash_attention_2":
        #     if attention_mask is not None and (attention_mask == 0.0).any():
        #         return attention_mask
        #     return None
        # if self.config._attn_implementation == "flex_attention":
        #     if isinstance(attention_mask, torch.Tensor):
        #         attention_mask = make_flex_block_causal_mask(attention_mask)
        #     return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        # print(f"head_mask: {head_mask}")
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = use_cache if use_cache is not None else True


        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        # print(f"use_cache: {use_cache}")
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if attention_mask is None:
            seq_length = past_seen_tokens + inputs_embeds.shape[1]
            attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        # print(f"attention_mask: {attention_mask}")
        # print(f"causal_mask: {causal_mask}")

        # embed positions
        if position_ids is None:
            # position_ids = cache_position.unsqueeze(0)
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_seen_tokens` is > 0
            position_ids = position_ids[:, past_seen_tokens:]

        pos_embeds = self.embed_positions(attention_mask, past_seen_tokens, position_ids=position_ids)

        hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)

        next_past_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                use_cache = use_cache,
                cache_position = cache_position,
                past_key_values = past_key_values 
            )
            if use_cache:
                hidden_states = outputs[0]
                next_past_key_values = past_key_values 

        
        hidden_states = self.final_layer_norm(hidden_states)

        logits = self.lm_head(hidden_states)
        # print("=========================================")
        # print(f"use_cache: {use_cache}")
        # print(f"past_key_values: {past_key_values}")
        # print(f"next_past_key_values: {next_past_key_values}")
        # print(f"logits: {logits}")
        if use_cache:
            return logits, next_past_key_values
        else:
            return logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Autoregressive text generation with support for padding-aware attention_mask.
        """
        # print("Start generate")
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        past_key_values = None
        cache_position = None

        if attention_mask is None:
            attention_mask = (input_ids != 1).float()  # padding_idx=1

        for step in range(max_length):
            if step == 0:
                current_input_ids = generated.to(device)
            else:
                current_input_ids = generated[:, -1:].to(device)
            # print(f"generated: {generated}")
            # print(current_input_ids)

            # 현재까지 생성된 토큰 기준으로 attention_mask 업데이트
            current_attention_mask = (generated != 1).float()  # shape: [B, T]

            logits, past_key_values = self.forward(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=False
            )

            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0) 
            # logits = logits[:, -1, :] / temperature  # 마지막 토큰만 사용

            # # Top-k filtering (optional)
            # if top_k > 0:
            #     top_k_values, _ = torch.topk(logits, top_k)
            #     min_topk = top_k_values[:, -1].unsqueeze(-1)
            #     logits = torch.where(logits < min_topk, torch.full_like(logits, -float("inf")), logits)

            # probs = torch.softmax(logits, dim=-1)

            # if do_sample:
            #     next_token = torch.multinomial(probs, num_samples=1)
            # else:
            #     next_token = torch.argmax(probs, dim=-1, keepdim=True)


            generated = torch.cat([generated, next_token_id], dim=1)

            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break

        return generated
