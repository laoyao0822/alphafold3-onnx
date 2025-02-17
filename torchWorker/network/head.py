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
import einops


from torchWorker.network import  pairformer



_CONTACT_THRESHOLD = 8.0
_CONTACT_EPSILON = 1e-3

class ConfidenceHead(nn.Module):
    """
    Implements Algorithm 31 in AF3
    """

    def __init__(self, c_single: int = 384, c_pair: int = 128, n_pairformer_layers=4):
        super(ConfidenceHead, self).__init__()

        self.c_single = c_single
        self.c_pair = c_pair


        self.confidence_pairformer = nn.ModuleList([
            pairformer.PairformerBlock(
                c_pair=self.c_pair,
            ) for _ in range(n_pairformer_layers)
        ])

    def forward(
        self,num_res
    ) :
        """
        Args:
            target_feat (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            dense_atom_positions (torch.Tensor): array of positions.
                [N_tokens, N_atom, 3] 
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            token_atoms_to_pseudo_beta (atom_layout.GatherInfo): Pseudo beta info for atom tokens.
        """
        # pairformer stack
        for layer in self.confidence_pairformer:
            layer(num_res)







