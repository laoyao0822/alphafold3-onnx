
from torch.nn import LayerNorm
import torch.nn as nn
import torch
import einops


class DiffusionTransformer(nn.Module):
    def __init__(self,
                 c_act: int = 768,
                 c_single_cond: int = 384,
                 c_pair_cond: int = 128,
                 num_head: int = 16,
                 num_blocks: int = 24,
                 super_block_size: int = 4) -> None:

        super(DiffusionTransformer, self).__init__()
        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        self.pair_input_layer_norm = LayerNorm(self.c_pair_cond)
        self.pair_logits_projection = nn.ModuleList(
            [nn.Linear(self.c_pair_cond, self.super_block_size * self.num_head, bias=False) for _ in range(self.num_super_blocks)])


    def forward(self,
                pair_cond:  torch.Tensor):
        pair_logits_cat=[]
        pair_act = self.pair_input_layer_norm(pair_cond)
        for super_block_i in range(self.num_super_blocks):
            pair_logits = self.pair_logits_projection[super_block_i](pair_act)
            pair_logits = einops.rearrange(
                pair_logits, 'n s (b h) -> b h n s', h=self.num_head).unsqueeze(0)
            print(pair_logits.shape)
            pair_logits_cat.append(pair_logits)

        return torch.stack(pair_logits_cat).contiguous()