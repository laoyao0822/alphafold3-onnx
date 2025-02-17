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
from torch.nn import Linear
from torch import distributed as dist

from torchWorker.network.dot_product_attention import dot_product_attention, dot_product_attention_triton


# from torchWorker.network.dot_product_attention import dot_product_attention_torch

# from torchWorker.network.dot_product_attention import dot_product_attention_flex

#
# class ParallelGridSelfAttention(Module):
#     def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False,
#                  rank: int = 0, world_size: int = 2):
#         super().__init__()
#         assert num_head % world_size == 0, "头数必须能被GPU数整除"
#
#         self.rank = rank  # 当前GPU编号 (0或1)
#         self.world_size = world_size  # GPU总数
#         self.c_pair = c_pair
#         self.num_head = num_head
#         self.transpose = transpose
#
#         # 拆分后的局部参数
#         self.local_heads = num_head // world_size
#         self.local_dim = c_pair // world_size
#
#         # 分片投影层
#         self._init_sharded_projections()
#
#         # 公共层
#         self.act_norm = LayerNorm(c_pair)
#         self.pair_bias_proj = Linear(c_pair, self.local_heads, bias=False)
#
#     def _init_sharded_projections(self):
#         """ 初始化分片权重 """
#         # Q/K/V 投影层 (按输出维度分片)
#         self.q_proj = Linear(self.c_pair, self.local_dim, bias=False)
#         self.k_proj = Linear(self.c_pair, self.local_dim, bias=False)
#         self.v_proj = Linear(self.c_pair, self.local_dim, bias=False)
#
#         # 门控层 (输入全量，输出分片)
#         self.gating_query = Linear(self.c_pair, self.c_pair, bias=False)
#
#         # 输出投影 (按输入维度分片)
#         self.out_proj = Linear(self.local_dim, self.c_pair, bias=False)
#
#     def load_sharded_weights(self, orig_module: GridSelfAttention):
#         """ 从原模型加载并分片权重 """
#         # Q/K/V 投影分片
#         self.q_proj.weight.data = orig_module.q_projection.weight.chunk(2, dim=0)[self.rank]
#         self.k_proj.weight.data = orig_module.k_projection.weight.chunk(2, dim=0)[self.rank]
#         self.v_proj.weight.data = orig_module.v_projection.weight.chunk(2, dim=0)[self.rank]
#
#         # 门控层全量复制
#         self.gating_query.weight.data = orig_module.gating_query.weight.data
#
#         # 输出投影分片
#         self.out_proj.weight.data = orig_module.output_projection.weight.chunk(2, dim=1)[self.rank]

class GridSelfAttention(nn.Module):

    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttention, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose
        self.block_shape=None
        self.world_size=dist.get_world_size()
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query2 = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)

        self.output_projection2 = nn.Linear(
            self.c_pair//self.world_size, self.c_pair, bias=False)
        self.q_projection2 = nn.Linear(self.c_pair, self.c_pair//self.world_size, bias=False)
        self.k_projection2 = nn.Linear(self.c_pair, self.c_pair//self.world_size, bias=False)
        self.v_projection2 = nn.Linear(self.c_pair, self.c_pair//self.world_size, bias=False)
        self.pair_bias_projection2 = nn.Linear(
            self.c_pair, self.num_head // self.world_size, bias=False)
        self.isFirst=True
    def chunk_weight(self):
        if self.isFirst:
            with torch.no_grad():
                # print("first")
                # self.num_head = self.num_head // 2
                self.q_projection2.weight.data = self.q_projection.weight.data[self.c_pair // self.world_size:, :]
                self.k_projection2.weight.data = self.k_projection.weight.data[self.c_pair // self.world_size:, :]
                self.v_projection2.weight.data = self.v_projection.weight.data[self.c_pair // self.world_size:, :]
                self.gating_query2.weight.data = self.gating_query.weight.data[self.c_pair // self.world_size:, :]
                self.output_projection2.weight.data = self.output_projection.weight.data[:, self.c_pair // self.world_size:]
                self.pair_bias_projection2.weight.data = self.pair_bias_projection.weight.data[self.num_head // self.world_size:,:]
                self.isFirst = False

    def _attention(self,num_res):

        seq_len=num_res

        pair_size = seq_len * seq_len * self.c_pair
        mask_size = seq_len * seq_len


        total_size = pair_size + mask_size

        combined_buffer = torch.zeros(total_size, dtype=torch.bfloat16, device='cuda:1').contiguous()
        # print("start to receive",combined_buffer.shape)
        dist.recv(tensor=combined_buffer, src=0)


        # 按顺序拆分张量
        pair = combined_buffer[:pair_size].view(seq_len, seq_len,self.c_pair)
        mask = combined_buffer[pair_size: pair_size + mask_size].view(seq_len, seq_len)
        # bias = combined_buffer[pair_size + mask_size:].view(self.num_head , seq_len, seq_len)

        q2 = self.q_projection2(pair)
        k2 = self.k_projection2(pair)
        v2 = self.v_projection2(pair)
        q2,k2,v2= map(lambda t: einops.rearrange(
             t, 'b n (h d) -> b h n d', h=self.num_head//2), [q2,k2,v2])

        bias2 = self.pair_bias_projection2(pair).permute(2, 0, 1)

        weighted_avg2=dot_product_attention_triton(q2, k2, v2,
                                                    mask=mask,
                                                    bias=bias2)
        weighted_avg2 = einops.rearrange(weighted_avg2, 'b h n d -> b n (h d)', h=self.num_head//2)


        gate_values2 = self.gating_query2(pair)

        weighted_avg2 *= torch.sigmoid(gate_values2)
        out_proj2 = self.output_projection2(weighted_avg2).contiguous()
        # print("out_proj2",out_proj2.shape,out_proj2.dtype)
        dist.isend(tensor=out_proj2, dst=0)
        # return out_proj2

    def forward(self,num_res):

        # print("start forward")
        self.chunk_weight()


        # print("success receive",pair.shape,mask.shape,bias.shape)
        self._attention(num_res)
        # print("out_proj mask")

        # print("send done")
        # print("one attention")

