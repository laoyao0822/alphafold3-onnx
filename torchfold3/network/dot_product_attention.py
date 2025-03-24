import math
from typing import Optional
import torch
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
    scaling = q.size(-1) ** -0.5
    q = q * scaling
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

def dot_product_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None):
    # if q,k,v is 3-dimensional, add a batch dimension
    qkv_dims = q.dim()
    if qkv_dims == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    # out = dot_product_attention_torch(q, k, v, mask, bias)
    # out = dot_product_attention_flex(q, k, v, mask, bias)
    # if k.shape[-1] not in {16, 32, 64, 128}:
    out = dot_product_attention_torch(q, k, v, mask, bias)
    # else:
    #     out = dot_product_attention_flex(q, k, v, mask, bias)

    if qkv_dims == 3:
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
