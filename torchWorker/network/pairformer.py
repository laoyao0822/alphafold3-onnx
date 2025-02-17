# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from typing import Optional

import torch
import torch.nn as nn

from torchWorker.network.attention import GridSelfAttention


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3
    Ref to: openfold/model/evoformer.py and protenix/model/modules/pairformer.py
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_pair: int = 128,
        n_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
    ) -> None:
        """
        Args:
            n_heads (int, optional): number of head [for SelfAttention]. Defaults to 16.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
                Defaults to 128.
            n_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
        """
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        self.num_intermediate_factor = num_intermediate_factor

        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )


    def forward(self,num_res) :
        self.pair_attention1(num_res)
        # print("pair_attention1")
        self.pair_attention2(num_res)
        # print("pair_attention2")

class EvoformerBlock(nn.Module):
    def __init__(self, c_msa: int = 64, c_pair: int = 128, n_heads_pair: int = 4) -> None:
        super(EvoformerBlock, self).__init__()
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
    def forward(
        self,num_res
    ) :
        print("num_res",num_res)
        #([107, 107])
        self.pair_attention1(num_res=num_res)
        self.pair_attention2(num_res=num_res)

        return
