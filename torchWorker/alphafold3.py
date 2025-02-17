# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
import sys
from typing import Tuple

import torch
import torch.nn as nn
import time

from torchWorker.network.pairformer import EvoformerBlock, PairformerBlock
from torchWorker.network.head import  ConfidenceHead
from torchWorker.network.template import TemplateEmbedding

SAVE_TO_TENSORRT = True
TENSORRT_PATH='/root/alphafold3/tensorrt_origin/af3.ep'
class Evoformer(nn.Module):
    def __init__(self, msa_channel: int = 64):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024


        self.seq_channel = 384
        self.pair_channel = 128
        self.c_target_feat = 447


        self.c_rel_feat = 139

        self.template_embedding = TemplateEmbedding(
            pair_channel=self.pair_channel)

        self.msa_stack = nn.ModuleList(
            [EvoformerBlock() for _ in range(self.msa_stack_num_layer)])

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock() for _ in range(self.pairformer_num_layer)])

    def _embed_template_pair(
        self,  ) :
        """Embeds Templates and merges into pair activations."""
        # templates = batch.templates
        # asym_id = batch.token_features.asym_id
        self.template_embedding()


    def _embed_process_msa(
        self
    ) :
        """Processes MSA and returns updated pair activations."""

        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
         msa_block( )


    def forward(
        self,
    ) :
        # T1 = time.time()
        self._embed_template_pair()
        # T2 = time.time()
        # print(f"pair embedding time: {T2 - T1}")
        self._embed_process_msa(
        )


        for pairformer_b in self.trunk_pairformer:
            pairformer_b()


        return

class AlphaFold3(nn.Module):

    def __init__(self, num_recycles: int = 10, num_samples: int = 5):
        super(AlphaFold3, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer = Evoformer()
        self.confidence_head = ConfidenceHead()

    def forward(self):
        for _ in range(self.num_recycles + 1):
           self.evoformer()
        for _ in range(self.num_samples):
            self.confidence_head()


