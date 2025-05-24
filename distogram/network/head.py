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
# ref:
# _CONTACT_THRESHOLD: Final[float] = 8.0
# _CONTACT_EPSILON: Final[float] = 1e-3
_CONTACT_THRESHOLD = 8.0
_CONTACT_EPSILON = 1e-3

class DistogramHead(nn.Module):
    def __init__(self,
                 c_pair: int = 128,
                 num_bins: int = 64,
                 first_break: float = 2.3125,
                 last_break: float = 21.6875) -> None:
        super(DistogramHead, self).__init__()
        self.c_pair = c_pair
        self.num_bins = num_bins
        self.first_break = first_break
        self.last_break = last_break
        self.half_logits = nn.Linear(self.c_pair, self.num_bins, bias=False)
        breaks = torch.linspace(
            self.first_break,
            self.last_break,
            self.num_bins - 1,
        )
        # self.register_buffer('breaks', breaks)
        self.breaks = nn.Parameter(breaks,requires_grad=False)
        bin_tops = torch.cat(
            (breaks, (breaks[-1] + (breaks[-1] - breaks[-2])).reshape(1)))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)
        # self.register_buffer('is_contact_bin', is_contact_bin)
        self.is_contact_bin = nn.Parameter(is_contact_bin,requires_grad=False)
    def forward(
        self,pair
        # embeddings: dict[str, torch.Tensor]
    ) :
        """
        Args:
            pair (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, num_bins]
        """
        pair_act = pair.clone()
        # seq_mask = batch.token_features.mask.to(dtype=torch.bool)
        # pair_mask = seq_mask[:, None] * seq_mask[None, :]
        left_half_logits = self.half_logits(pair_act)
        right_half_logits = left_half_logits
        logits = left_half_logits + right_half_logits.transpose(-2, -3)
        probs = torch.softmax(logits, dim=-1)
        contact_probs = torch.einsum('ijk,k->ij', probs, self.is_contact_bin)
        # contact_probs = pair_mask * contact_probs
        return self.breaks,contact_probs

        # return {
        #     'bin_edges': self.breaks,
        #     'contact_probs': contact_probs,
        # }

