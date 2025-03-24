# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from typing import Optional, Sequence

import torch

from target_feat.network import utils
from alphafold3.constants import residue_names


def create_target_featV2(
    aatype,
    profile,deletion_mean,
    ref_ops, ref_mask, ref_element,
    append_per_atom_features: bool,
) -> torch.Tensor:
    """Make target feat."""
    # token_features = batch.token_features
    target_features = []

    target_features.append(
        torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
        )
    )
    target_features.append(profile)
    target_features.append(deletion_mean[..., None])

    # Reference structure features
    if append_per_atom_features:
        ref_mask = ref_mask
        element_feat = torch.nn.functional.one_hot(
            ref_element, 128)
        element_feat = utils.mask_mean(
            mask=ref_mask[..., None], value=element_feat, axis=-2, eps=1e-6
        )
        target_features.append(element_feat)
        pos_feat = ref_ops
        pos_feat = pos_feat.reshape([pos_feat.shape[0], -1])
        target_features.append(pos_feat)
        target_features.append(ref_mask)

    return torch.concatenate(target_features, dim=-1)


