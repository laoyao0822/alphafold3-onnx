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


from target_feat.network import featurization
from target_feat.network import atom_cross_attention


class TargetFeat(nn.Module):

    def __init__(self):
        super(TargetFeat, self).__init__()

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()
    def create_target_feat_embedding(self,
            aatype,
            profile,deletion_mean,
            ref_ops, ref_mask, ref_element, ref_charge,
            ref_atom_name_chars, ref_space_uid,
            pred_dense_atom_mask,
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,
            acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask,
            acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask,
            acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask,
            ) -> torch.Tensor:
        target_feat = featurization.create_target_featV2(
            aatype=aatype,
            profile=profile, deletion_mean=deletion_mean,
            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element,
            append_per_atom_features=False,
        )

        enc = self.evoformer_conditioning(
            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
            pred_dense_atom_mask=pred_dense_atom_mask,
            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            # acat_atoms_to_q_input_shape=acat_atoms_to_q_input_shape,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            # acat_q_to_k_input_shape=acat_q_to_k_input_shape,
            acat_t_to_q_gather_idxs=acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask=acat_t_to_q_gather_mask,
            # acat_t_to_q_input_shape=acat_t_to_q_input_shape,
            acat_q_to_atom_gather_idxs=acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask=acat_q_to_atom_gather_mask,
            # acat_q_to_atom_input_shape=acat_q_to_atom_input_shape,
            acat_t_to_k_gather_idxs=acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask=acat_t_to_k_gather_mask,
            # acat_t_to_k_input_shape=acat_t_to_k_input_shape,
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            # batch=batch,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat

    def forward(self,
                aatype,
                profile, deletion_mean,

                pred_dense_atom_mask,

                acat_atoms_to_q_gather_idxs,
                acat_atoms_to_q_gather_mask,

                acat_q_to_k_gather_idxs,
                acat_q_to_k_gather_mask,

                acat_t_to_q_gather_idxs,
                acat_t_to_q_gather_mask,

                acat_q_to_atom_gather_idxs,
                acat_q_to_atom_gather_mask,

                acat_t_to_k_gather_idxs,
                acat_t_to_k_gather_mask,

                ref_ops, ref_mask, ref_element, ref_charge,
                ref_atom_name_chars, ref_space_uid,
                ) :

        target_feat = self.create_target_feat_embedding(
            aatype=aatype,profile=profile,deletion_mean=deletion_mean,
            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
            pred_dense_atom_mask=pred_dense_atom_mask,
            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            acat_t_to_q_gather_idxs=acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask=acat_t_to_q_gather_mask,
            acat_q_to_atom_gather_idxs=acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask=acat_q_to_atom_gather_mask,
            acat_t_to_k_gather_idxs=acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask=acat_t_to_k_gather_mask,
        )
        # sys.exit(" ---------")

        return  target_feat

