# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

from dataclasses import dataclass

import torch
import torch.nn as nn


from evoWorker.network import pairformer



class TemplateEmbedding(nn.Module):
    """Embed a set of templates."""

    def __init__(self, pair_channel: int = 128, num_channels: int = 64):
        super(TemplateEmbedding, self).__init__()

        self.pair_channel = pair_channel
        self.num_channels = num_channels

        self.single_template_embedding = SingleTemplateEmbedding()

    def forward(
        self,num_res,attn_mask,
    ) :
        num_templates=4
        for template_idx in range(num_templates):
           self.single_template_embedding(num_res,attn_mask)
        return

class SingleTemplateEmbedding(nn.Module):
    """Embed a single template."""
    def __init__(self, num_channels: int = 64):
        super(SingleTemplateEmbedding, self).__init__()
        self.num_channels = num_channels
        self.template_stack_num_layer = 2
        self.template_embedding_iteration = nn.ModuleList(
            [pairformer.PairformerBlock(c_pair=self.num_channels, num_intermediate_factor=2)
             for _ in range(self.template_stack_num_layer)]
        )
    def forward(
        self,num_res,attn_mask
    ) :
        for pairformer_block in self.template_embedding_iteration:
            pairformer_block(num_res,attn_mask)



