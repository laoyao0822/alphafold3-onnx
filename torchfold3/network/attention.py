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

from torchfold3.network.layer_norm import LayerNorm
from torchfold3.network.dot_product_attention import dot_product_attention
import torch.distributed as dist

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



        self.gating_query1 = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.output_projection1 = nn.Linear(
            self.c_pair//2, self.c_pair, bias=False)
        self.q_projection1 = nn.Linear(self.c_pair, self.c_pair//2, bias=False)
        self.k_projection1 = nn.Linear(self.c_pair, self.c_pair//2, bias=False)
        self.v_projection1 = nn.Linear(self.c_pair, self.c_pair// 2, bias=False)
        self.isFirst=True

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):

        if self.isFirst:
            # print("first")
            # self.num_head=self.num_head
            self.q_projection1.weight.data=self.q_projection.weight.data[:self.c_pair//2,:]
            self.k_projection1.weight.data = self.k_projection.weight.data[:self.c_pair // 2, :]
            self.v_projection1.weight.data = self.v_projection.weight.data[:self.c_pair // 2, :]
            self.gating_query1.weight.data= self.gating_query.weight.data[:self.c_pair // 2, :]
            self.output_projection1.weight.data=self.output_projection.weight.data[:,:self.c_pair//2]
            self.isFirst=False

        q1 = self.q_projection1(pair)
        k1 = self.k_projection1(pair)
        v1 = self.v_projection1(pair)
        #q shape torch.Size([107, 4, 107, 32]) q1 shape torch.Size([107, 2, 107, 32])
        # bias shape torch.Size([4, 107, 107]) v shape torch.Size([107, 2, 107, 32])

        q1,k1,v1= map(lambda t: einops.rearrange(
             t, 'b n (h d) -> b h n d', h=self.num_head//2), [q1,k1,v1])
        #mask shape torch.Size([107, 107])

        bias1=None
        if bias is not None:
            bias1=bias.chunk(2, dim=0)[0]
        weighted_avg1 = dot_product_attention(q1, k1, v1,
                                                    mask=mask,
                                                    bias=bias1)
        weighted_avg1 = einops.rearrange(weighted_avg1, 'b h n d -> b n (h d)')
        #weighted_avg1 shape torch.Size([107, 107, 64]) weighted_avg2 shape torch.Size([107, 107, 64])

        gate_values1 = self.gating_query1(pair)
        # print("gate_values1 shape",gate_values1.shape,"gate_values2 shape",gate_values2.shape)

        weighted_avg1 *= torch.sigmoid(gate_values1)
        out_proj1 = self.output_projection1(weighted_avg1)

        return out_proj1
    def forward(self, pair, mask):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)
        #torch.Size([583, 583, 64]) torch.Size([583, 583]) torch.Size([4, 583, 583])

        if self.transpose:
            pair = pair.permute(1, 0, 2)
        print("dtype",pair.dtype,mask.dtype,nonbatched_bias.dtype)
        print(" shape",pair.shape,mask.shape,nonbatched_bias.shape)
        pair = pair.to(dtype=torch.bfloat16).contiguous()
        mask = mask.to(dtype=torch.bfloat16).contiguous()
        nonbatched_bias = nonbatched_bias.to(dtype=torch.bfloat16).contiguous()

        combined_tensor = torch.cat([
            pair.view(-1),  # 展平为 1D
            mask.view(-1),
            nonbatched_bias.view(-1)
        ]).contiguous()


        # print("combined_tensor",combined_tensor.shape)
        # dist.isend(mask, dst=1)
        # dist.isend(nonbatched_bias.chunk(2, dim=0)[1], dst=1)
        dist.isend(combined_tensor, dst=1)

        output_proj2 = torch.zeros([583,583,self.c_pair], dtype=torch.bfloat16,device='cuda:0').contiguous()
        res = dist.irecv(tensor=output_proj2, src=1)
        output_proj1 = self._attention(pair, mask, nonbatched_bias).contiguous()

        res.wait()
        pair=output_proj1+output_proj2
        if self.transpose:
            pair = pair.permute(1, 0, 2)
        return pair
# class GridSelfAttention(nn.Module):
#     def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
#         super(GridSelfAttention, self).__init__()
#         self.c_pair = c_pair
#         self.num_head = num_head
#         self.qkv_dim = self.c_pair // self.num_head
#         self.transpose = transpose
#
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
#     def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
#         q = self.q_projection(pair)
#         k = self.k_projection(pair)
#         v = self.v_projection(pair)
#
#         q, k, v = map(lambda t: einops.rearrange(
#             t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])
#
#         weighted_avg = dot_product_attention(q, k, v,
#                                                     mask=mask,
#                                                     bias=bias)
#
#         weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')
#
#         gate_values = self.gating_query(pair)
#
#         weighted_avg *= torch.sigmoid(gate_values)
#         return self.output_projection(weighted_avg)
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
#         nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)
#
#         if self.transpose:
#             pair = pair.permute(1, 0, 2)
#
#         pair = self._attention(pair, mask, nonbatched_bias)
#
#         if self.transpose:
#             pair = pair.permute(1, 0, 2)
#
#         return pair


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
