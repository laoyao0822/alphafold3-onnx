# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import torch
import torch.nn as nn
# from torch.nn import RMSNorm as LayerNorm


from torch.nn import LayerNorm
from evoformer.network.gated_linear_unit import gated_linear_unit_torch

import time

class TriangleMultiplication(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplication, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self._outgoing=False
        self._outgoing=_outgoing

        self.equation='ckj,cki->cij'
        if _outgoing is True:
            self.equation='cik,cjk->cij'

    shard_transition_blocks: bool = True

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.left_norm_input(pair)
        input_pair = pair
        # time1 = time.time()
        projection = self.projection(pair)
        projection = projection.permute(2, 0, 1)
        # if mask is not None:
        #     projection *= mask[None, ...]

        gate = self.gate(pair)
        gate = gate.permute(2, 0, 1)
        projection *= torch.sigmoid(gate)

        projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])

        a, b = torch.chunk(projection, 2, dim=1)
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
        pair = torch.einsum(self.equation, a, b)

        pair = pair.permute(1, 2, 0)
        # print("trimul cost time:", time.time() - time1)
        pair = self.center_norm(pair)
        pair = self.output_projection(pair)

        gate_out = self.gating_linear(input_pair)
        pair *= torch.sigmoid(gate_out)

        return pair
    
# ref:
# class OuterProductMean(hk.Module):
#   """Computed mean outer product."""

#   class Config(base_config.BaseConfig):
#     chunk_size: int = 128
#     num_outer_channel: int = 32

#   def __init__(
#       self,
#       config: Config,
#       global_config: model_config.GlobalConfig,
#       num_output_channel,
#       *,
#       name,
#   ):
#     super().__init__(name=name)
#     self.global_config = global_config
#     self.config = config
#     self.num_output_channel = num_output_channel

class OuterProductMean(nn.Module):
    def __init__(self, c_msa: int = 64, num_output_channel: int = 128, num_outer_channel: int = 32) -> None:
        super(OuterProductMean, self).__init__()

        self.c_msa = c_msa
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.epsilon = 1e-3

        self.layer_norm_input = LayerNorm(self.c_msa)
        self.left_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)
        self.right_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)

        self.output_w = nn.Parameter(
            torch.randn(self.num_outer_channel, self.num_outer_channel, self.num_output_channel))
        self.output_b = nn.Parameter(
            torch.randn(self.num_output_channel))

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left_act = mask * self.left_projection(msa)
        right_act = mask * self.right_projection(msa)


        left_act = left_act.permute(0, 2, 1)
        act = torch.einsum('acb,ade->dceb', left_act, right_act)
        act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
        act = act.permute(1, 0, 2)

        norm = torch.einsum('abc,adc->bdc', mask, mask)
        return act / (self.epsilon + norm)



class Transition(nn.Module):

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = LayerNorm(c_x)
        self.transition1 = nn.Linear(
            c_x, self.num_intermediate_factor * c_x * 2, bias=False)
        self.transition2 = nn.Linear(
            self.num_intermediate_factor * c_x, c_x, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        c = gated_linear_unit_torch(x, self.transition1.weight.T)
        return self.transition2(c)