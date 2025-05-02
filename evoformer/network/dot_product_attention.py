from typing import Optional
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
import time
# from torch.nn.attention.flex_attention import (
#     flex_attention,
# )
#
# flex_attention = torch.compile(flex_attention,dynamic=False)
# print("compile")
# torch._dynamo.config.cache_size_limit = 1024
# torch._dynamo.config.accumulated_cache_size_limit = 1024
def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    # scaling = q.size(-1) ** -0.5
    # q = q * scaling
    logits = torch.matmul(q, k.transpose(-1, -2))

    if bias is not None:
        logits += bias

    if mask is not None:
        if mask.dim() == 1:
            mask = mask[None, None, None, :].to(dtype=torch.bool)
        elif mask.dim() == 2:
            mask = mask[:, None, None, :].to(dtype=torch.bool)
        logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)

    return torch.matmul(weights, v)

import torch

@torch.compile
def dot_product_attention_sdpa_full(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # 确保输入维度对齐
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Inputs must be 4D: (batch, heads, seq_len, head_dim)"

    # 合并 mask 和 bias 的逻辑
    attn_mask = None

    if mask is not None or bias is not None:
        # 初始化 attn_mask 为全 0
        device = q.device
        dtype = q.dtype
        attn_mask = torch.zeros(
            (q.size(0), q.size(1), q.size(2), k.size(2)),
            dtype=dtype, device=device
        )
        # print("attn_mask", attn_mask.shape)
        # 添加 bias 到 attn_mask
        if bias is not None:
            if bias.dim() == 3:  # (heads, seq_len_q, seq_len_k)
                bias = bias.unsqueeze(0)  # 扩展 batch 维度
            attn_mask += bias
        mask=mask.to(dtype=torch.bool)
        # 添加 mask 到 attn_mask
        if mask is not None:
            # 统一 mask 格式为 (batch, seq_len_k)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # (1, seq_len_k)
            elif mask.dim() == 2:
                mask = mask  # (batch, seq_len_k)
            else:
                raise ValueError("mask 必须是 1D (seq_len) 或 2D (batch, seq_len)")

            # 将 bool mask 转换为 float mask (-inf/0)
            mask_float = torch.zeros_like(mask, dtype=dtype, device=device)
            mask_float = mask_float.masked_fill(~mask, float('-inf'))  # True 位置保留 0，False 设为 -inf

            # 扩展 mask 到 (batch, 1, 1, seq_len_k) 以便广播
            mask_float = mask_float[:, None, None, :]
            attn_mask += mask_float

    # 调用 PyTorch SDPA
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,  # 合并后的掩码和偏置
        dropout_p=0.0,  # 无 dropout
        is_causal=False, # 非因果掩码（由 mask/bias 显式控制）
        enable_gqa=True
    )


def get_attn_mask(mask,dtype,device,seq_len,num_heads,batch_size):
    attn_mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len),
        dtype=dtype, device=device
    )
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)  # (1, seq_len_k)
    mask_float = torch.zeros_like(mask, dtype=dtype, device=device)
    mask_float = mask_float.masked_fill(~mask.to(dtype=torch.bool), -torch.inf)  # True 位置保留 0，False 设为 -inf

    # 扩展 mask 到 (batch, 1, 1, seq_len_k) 以便广播
    mask_float = mask_float[:, None, None, :]
    attn_mask += mask_float
    return attn_mask


def get_attn_mask_withqk(mask,q,k):
    device = q.device
    dtype = q.dtype
    attn_mask = torch.zeros(
        (q.size(0), q.size(1), q.size(2), k.size(2)),
        dtype=q.dtype, device=q.device
    )
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)  # (1, seq_len_k)
    mask_float = torch.zeros_like(mask, dtype=dtype, device=device)
    mask_float = mask_float.masked_fill(~mask.to(dtype=torch.bool), -torch.inf)  # True 位置保留 0，False 设为 -inf

    # 扩展 mask 到 (batch, 1, 1, seq_len_k) 以便广播
    mask_float = mask_float[:, None, None, :]
    attn_mask += mask_float
    return attn_mask


execute_time=0

# @torch._dynamo.disallow_in_graph
def dot_product_attention_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask,
        bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    sdpa_mask=attn_mask+bias
    # print('q.shape',q.shape,'k.shape',k.shape,'v.shape',v.shape,'attn_mask.shape',attn_mask.shape)
    # print("sdpa execute")
    # with sdpa_kernel(backends=[SDPBackend.MATH]):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=sdpa_mask,  # 合并后的掩码和偏置
        dropout_p=0.0,  # 无 dropout
        is_causal=False , # 非因果掩码（由 mask/bias 显式控制）
        # enable_gqa = True

    )


def dot_product_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None):
    # if q,k,v is 3-dimensional, add a batch dimension
    qkv_dims = q.dim()

    if qkv_dims == 3:
        print("qkv_dims",qkv_dims)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    # out = dot_product_attention_torch(q, k, v, mask, bias)
    # out = dot_product_attention_flex(q, k, v, mask, bias)
    # if k.shape[-1] not in {16, 32, 64, 128}:
    print("attn mask",get_attn_mask_withqk(mask,q,k).shape)
    out = dot_product_attention_sdpa_full(q, k, v, mask, bias)
    # else:
    #     out = dot_product_attention_flex(q, k, v, mask, bias)

    if qkv_dims == 3:
        print("qkv dim 3")
        out = out.squeeze(0)

    return out
#
# def dot_product_attention_flex(
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         bias: Optional[torch.Tensor] = None
# ) :
#     if mask==None and bias==None:
#         return flex_attention(
#             q,
#             k,  # 显式转置以匹配原版逻辑
#             v,
#             score_mod=None,
#             block_mask=None,  # 关键：禁用内置掩码
#             scale=None  # 关键：禁用内置缩放
#         )
#
#     if mask is not None and bias is not None:
#         mask=mask.to(dtype=torch.bool)
#         # # 调整掩码维度：[batch, seq_len] → [batch, 1, 1, seq_len]
#         if mask.dim() == 1:
#             mask = mask.unsqueeze(0).repeat(q.size(0), 1)
#         # elif mask.dim() == 2:
#         #     mask = mask[:, None, None, :]
#         # 将布尔掩码转换为加法掩码（-inf表示需要掩码的位置）
#         additive_mask = torch.zeros_like(mask, dtype=q.dtype,device=q.device)
#         additive_mask = additive_mask.masked_fill_(~mask, -torch.inf)
#         attn_mask = additive_mask.to()
#         def bias_mod(score, b, h, q_idx, kv_idx):
#             return score + bias[h, q_idx, kv_idx]+attn_mask[b,kv_idx]
#         return flex_attention(
#         q,k, v,score_mod=bias_mod)
#
#     if bias is not None and mask is None:
#         def bias_mod(score, b, h, q_idx, kv_idx):
#             return score + bias[h, q_idx, kv_idx]
#         return flex_attention.flex_attention(
#         q,k, v,score_mod=bias_mod)
#
#     if bias is  None and mask is not None:
#         mask=mask.to(dtype=torch.bool)
#         # # 调整掩码维度：[batch, seq_len] → [batch, 1, 1, seq_len]
#         if mask.dim() == 1:
#             mask = mask.unsqueeze(0).repeat(q.size(0), 1)
#         additive_mask = torch.zeros_like(mask, dtype=q.dtype,device=q.device)
#         additive_mask = additive_mask.masked_fill(~mask, -torch.inf)
#         attn_mask = additive_mask.to(device=q.device)
#         print("attn_mask",attn_mask)
#         def bias_mod(score, b, h, q_idx, kv_idx):
#             return score + attn_mask[b,kv_idx]
#         return flex_attention.flex_attention(
#         q,k, v,score_mod=bias_mod)
