# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import einops
import torch
import torch.nn as nn

# from torch.nn import RMSNorm as LayerNorm
from torch.nn import LayerNorm
from evoformer.network.dot_product_attention import dot_product_attention
# from evoformer.network.dot_product_attention import dot_product_attention_flex
import torch.distributed as dist
import time
# class GridSelfAttention(nn.Module):
#
#     def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
#         super(GridSelfAttention, self).__init__()
#         self.c_pair = c_pair
#         self.num_head = num_head
#         self.qkv_dim = self.c_pair // self.num_head
#         self.transpose = transpose
#         self.block_shape=None
#         self.act_norm = LayerNorm(self.c_pair)
#         self.pair_bias_projection = nn.Linear(
#             self.c_pair, self.num_head, bias=False)
#
#         self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
#         self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
#         self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
#
#         self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
#         self.output_projection = nn.Linear(
#             self.c_pair, self.c_pair, bias=False)
#
#
#
#         self.gating_query1 = nn.Linear(self.c_pair, self.c_pair, bias=False)
#
#         self.output_projection1 = nn.Linear(
#             self.c_pair//2, self.c_pair, bias=False)
#         self.q_projection1 = nn.Linear(self.c_pair, self.c_pair//2, bias=False)
#         self.k_projection1 = nn.Linear(self.c_pair, self.c_pair//2, bias=False)
#         self.v_projection1 = nn.Linear(self.c_pair, self.c_pair// 2, bias=False)
#         self.pair_bias_projection1 = nn.Linear(
#             self.c_pair, self.num_head//2, bias=False)
#
#         self.gating_query2 = nn.Linear(self.c_pair, self.c_pair, bias=False)
#
#         self.output_projection2 = nn.Linear(
#             self.c_pair // 2, self.c_pair, bias=False)
#         self.q_projection2 = nn.Linear(self.c_pair, self.c_pair // 2, bias=False)
#         self.k_projection2 = nn.Linear(self.c_pair, self.c_pair // 2, bias=False)
#         self.v_projection2 = nn.Linear(self.c_pair, self.c_pair // 2, bias=False)
#         self.pair_bias_projection2 = nn.Linear(
#             self.c_pair, self.num_head // 2, bias=False)
#
#         self.isFirst=True
#
#     def _attention(self, pair: torch.Tensor, mask: torch.Tensor):
#
#         if self.isFirst:
#             # self.num_head=self.num_head//2
#             self.q_projection1.weight.data=self.q_projection.weight.data[:self.c_pair//2,:]
#             self.k_projection1.weight.data = self.k_projection.weight.data[:self.c_pair // 2, :]
#             self.v_projection1.weight.data = self.v_projection.weight.data[:self.c_pair // 2, :]
#             self.gating_query1.weight.data= self.gating_query.weight.data[:self.c_pair // 2, :]
#             self.output_projection1.weight.data=self.output_projection.weight.data[:,:self.c_pair//2]
#             self.pair_bias_projection1.weight.data = self.pair_bias_projection.weight.data[:self.num_head // 2, :]
#
#             self.q_projection2.weight.data = self.q_projection.weight.data[self.c_pair // 2:, :]
#             self.k_projection2.weight.data = self.k_projection.weight.data[self.c_pair // 2:, :]
#             self.v_projection2.weight.data = self.v_projection.weight.data[self.c_pair // 2:, :]
#             self.gating_query2.weight.data = self.gating_query.weight.data[self.c_pair // 2:, :]
#             self.output_projection2.weight.data = self.output_projection.weight.data[:, self.c_pair // 2:]
#             self.pair_bias_projection2.weight.data = self.pair_bias_projection.weight.data[self.num_head // 2:, :]
#             self.isFirst=False
#
#         q1 = self.q_projection1(pair)
#         k1 = self.k_projection1(pair)
#         v1 = self.v_projection1(pair)
#         #q shape torch.Size([107, 4, 107, 32]) q1 shape torch.Size([107, 2, 107, 32])
#         # bias shape torch.Size([4, 107, 107]) v shape torch.Size([107, 2, 107, 32])
#
#         q1,k1,v1= map(lambda t: einops.rearrange(
#              t, 'b n (h d) -> b h n d', h=self.num_head//2), [q1,k1,v1])
#         #mask shape torch.Size([107, 107])
#
#         q2 = self.q_projection2(pair)
#         k2 = self.k_projection2(pair)
#         v2 = self.v_projection2(pair)
#         q2, k2, v2 = map(lambda t: einops.rearrange(
#             t, 'b n (h d) -> b h n d', h=self.num_head//2), [q2, k2, v2])
#
#
#         bias1=self.pair_bias_projection(pair).permute(2, 0, 1).chunk(2,dim=0)[0]
#         bias2 =self.pair_bias_projection(pair).permute(2, 0, 1).chunk(2,dim=0)[1]
#
#
#         weighted_avg1 = dot_product_attention(q1, k1, v1,
#                                                     mask=mask,
#                                                     bias=bias1)
#         weighted_avg1 = einops.rearrange(weighted_avg1, 'b h n d -> b n (h d)')
#
#
#         #weighted_avg1 shape torch.Size([107, 107, 64]) weighted_avg2 shape torch.Size([107, 107, 64])
#         weighted_avg2 = dot_product_attention(q2, k2, v2,
#                                               mask=mask,
#                                               bias=bias2)
#         weighted_avg2 = einops.rearrange(weighted_avg2, 'b h n d -> b n (h d)')
#         gate_values1 = self.gating_query1(pair)
#         gate_values2 = self.gating_query2(pair)
#
#         # print("gate_values1 shape",gate_values1.shape,"gate_values2 shape",gate_values2.shape)
#
#         weighted_avg1 *= torch.sigmoid(gate_values1)
#         weighted_avg2 *= torch.sigmoid(gate_values2)
#
#         out_proj2 = self.output_projection2(weighted_avg2)
#
#
#         out_proj1 = self.output_projection1(weighted_avg1)
#
#         return out_proj1+out_proj2
#
#
#
#     def forward(self, pair, mask):
#         """
#         Args:
#             pair (torch.Tensor): [N_token, N_token, c_pair]
#             mask (torch.Tensor): [N_token, N_token]
#         Returns:
#             torch.Tensor: [N_token, N_token, c_pair]
#         """
#
#         pair = self.act_norm(pair)
#         # nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)
#         #torch.Size([583, 583, 64]) torch.Size([583, 583]) torch.Size([4, 583, 583])
#
#         if self.transpose:
#             pair = pair.permute(1, 0, 2)
#         # print("dtype",pair.dtype,mask.dtype,.dtype)
#
#
#         pair = self._attention(pair, mask).contiguous()
#
#         if self.transpose:
#             pair = pair.permute(1, 0, 2)
#         return pair


from evoformer.network.dot_product_attention import dot_product_attention_sdpa

class GridSelfAttention(nn.Module):

    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttention, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose
        self.block_shape=None

        self.act_norm = LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
    def _attention(self, pair: torch.Tensor, mask=None,attn_mask=None) -> torch.Tensor:

        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)
        # q shape torch.Size([107, 4, 107, 32]) q1 shape torch.Size([107, 2, 107, 32])
        # bias shape torch.Size([4, 107, 107]) v shape torch.Size([107, 2, 107, 32])
        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])
        bias = self.pair_bias_projection(pair).permute(2, 0, 1)
        # print("mask bias",mask.shape,bias.shape)

        # if attn_mask is not None:
        weighted_avg=dot_product_attention_sdpa(q, k, v, attn_mask=attn_mask,bias=bias)
        # else:
        #     weighted_avg = dot_product_attention(q, k, v,
        #                                       mask=mask,
        #                                       bias=bias)

        # print("weighted_avg", weighted_avg.shape)

        batch_size, num_heads, seq_len, head_dim = weighted_avg.size()
        # weighted_avg1 shape torch.Size([107, 107, 64]) weighted_avg2 shape torch.Size([107, 107, 64])
        weighted_avg = weighted_avg.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        gate = torch.sigmoid(self.gating_query(pair))
        weighted_avg.mul_(gate)

        return self.output_projection(weighted_avg)


    def forward(self, pair, mask=None,attn_mask=None):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
        pair = self.act_norm(pair)
        #torch.Size([583, 583, 64]) torch.Size([583, 583]) torch.Size([4, 583, 583])
        if self.transpose:
            pair = pair.permute(1, 0, 2)

        pair = self._attention(pair, mask=mask,attn_mask=attn_mask).contiguous()
        # if self.c_pair==128:
        #     print("attention time:",time.time()-time1)
        if self.transpose:
            pair = pair.permute(1, 0, 2)
        return pair



class MSAAttention(nn.Module):
    def __init__(self, c_msa=64, c_pair=128, num_head=8):
        super(MSAAttention, self).__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = LayerNorm(self.c_msa)
        self.pair_norm = LayerNorm(self.c_pair)
        self.pair_logits = nn.Linear(self.c_pair, self.num_head, bias=False)
        self.v_projection = nn.Linear(
            self.c_msa, self.num_head * self.value_dim, bias=False)
        self.gating_query = nn.Linear(self.c_msa, self.c_msa, bias=False)
        self.output_projection = nn.Linear(self.c_msa, self.c_msa, bias=False)

    def forward(self, msa, msa_mask, pair):
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1)

        logits += 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head)

        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        gate_values = self.gating_query(msa)
        v_avg *= torch.sigmoid(gate_values)

        return self.output_projection(v_avg)
