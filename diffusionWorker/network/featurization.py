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

from diffusionWorker.network import utils
from alphafold3.constants import residue_names


def gumbel_noise(
    shape,
    device: torch.device,
    eps=1e-6,
    # generator=None,
) -> torch.Tensor:
    """Generate Gumbel Noise of given Shape.

    This generates samples from Gumbel(0, 1).

    Args:
        shape: Shape of noise to return.

    Returns:
        Gumbel noise of given shape.
    """
    uniform_noise = torch.rand(
        shape, dtype=torch.float32, device=device
    )
    gumbel = -torch.log(-torch.log(uniform_noise + eps) + eps)
    return gumbel


def gumbel_argsort_sample_idx(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Samples with replacement from a distribution given by 'logits'.

    This uses Gumbel trick to implement the sampling an efficient manner. For a
    distribution over k items this samples k times without replacement, so this
    is effectively sampling a random permutation with probabilities over the
    permutations derived from the logprobs.

    Args:
      key: prng key
      logits: logarithm of probabilities to sample from, probabilities can be
        unnormalized.

    Returns:
      Sample from logprobs in one-hot form.
    """
    z = gumbel_noise(logits.shape, device=logits.device)
    return torch.argsort(logits + z, dim=-1, descending=True)



# @torch.compile
def create_msa_feat(rows,deletion_matrix) -> torch.Tensor:
    """Create and concatenate MSA features."""
    msa_1hot = torch.nn.functional.one_hot(
        rows.to(
            dtype=torch.int64), residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1
    )
    # deletion_matrix = msa.deletion_matrix
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / torch.pi))[
        ..., None
    ]
    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
    ]
    return torch.concatenate(msa_feat, dim=-1)




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


# @torch.compile
def create_relative_encodingV2(
    # seq_features: features.TokenFeatures,
    token_index,residue_index,asym_id,entity_id,sym_id,
    max_relative_idx: int,
    max_relative_chain: int
) -> torch.Tensor:
    """Add relative position encodings."""
    # rel_feats = []
    # token_index = seq_features.token_index
    # residue_index = seq_features.residue_index
    # asym_id = seq_features.asym_id
    # entity_id = seq_features.entity_id
    # sym_id = seq_features.sym_id
    # rel_feats = []
    left_asym_id = asym_id[:, None]
    right_asym_id = asym_id[None, :]

    left_residue_index = residue_index[:, None]
    right_residue_index = residue_index[None, :]

    left_token_index = token_index[:, None]
    right_token_index = token_index[None, :]

    left_entity_id = entity_id[:, None]
    right_entity_id = entity_id[None, :]

    left_sym_id = sym_id[:, None]
    right_sym_id = sym_id[None, :]

    # Embed relative positions using a one-hot embedding of distance along chain
    offset = left_residue_index - right_residue_index
    clipped_offset = torch.clip(
        offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = torch.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
    )
    rel_pos = torch.nn.functional.one_hot(final_offset.to(
        dtype=torch.int64), 2 * max_relative_idx + 2)
    # rel_feats.append(rel_pos)
    # Embed relative token index as a one-hot embedding of distance along residue
    token_offset = left_token_index - right_token_index
    clipped_token_offset = torch.clip(
        token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    residue_same = (left_asym_id == right_asym_id) & (left_residue_index == right_residue_index)
    final_token_offset = torch.where(residue_same,clipped_token_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_token_offset),
    )
    rel_token = torch.nn.functional.one_hot(
        final_token_offset.to(dtype=torch.int64), 2 * max_relative_idx + 2)
    # rel_feats.append(rel_token)

    # Embed same entity ID
    entity_id_same = left_entity_id == right_entity_id
    # rel_entity=entity_id_same.to(dtype=rel_pos.dtype)[..., None]
    # rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype)[..., None])
    # Embed relative chain ID inside each symmetry class
    rel_sym_id = left_sym_id - right_sym_id

    max_rel_chain = max_relative_chain

    clipped_rel_chain = torch.clip(
        rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
    )
    final_rel_chain = torch.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
    )
    rel_chain = torch.nn.functional.one_hot(final_rel_chain.to(
        dtype=torch.int64), 2 * max_relative_chain + 2)
    # rel_entity = entity_id_same.to(dtype=rel_pos.dtype)[..., None]
    # print("rel_pos",rel_pos.shape,"rel_token",rel_token.shape,"entity_id_same",entity_id_same[..., None].shape,"rel_chain",rel_chain.shape)
    rel_feats=[rel_pos,rel_token,
               entity_id_same.to(dtype=rel_pos.dtype)[..., None],rel_chain]
    # rel_feats.append(rel_chain)
    return torch.concatenate(rel_feats, dim=-1)


#0135->0134 little precision up
# @torch.compile
def shuffle_msa_runcate(
    rows,mask,deletion_matrix,num_msa: int
) :
    """Shuffle MSA randomly, return batch with shuffled MSA.

    Returns:
        rows,mask,deletion_matrix
      Protein with sampled msa.
    """
    # Sample uniformly among sequences with at least one non-masked position.
    logits = (torch.clip(torch.sum(mask, dim=-1), 0.0, 1.0) - 1.0) * 1e6
    index_order = torch.argsort(logits + gumbel_noise(logits.shape, device=logits.device), dim=-1, descending=True)
    indices = torch.arange(num_msa, device=rows.device, dtype=torch.int64)
    # rows, mask, deletion_matrix= rows[index_order, :],mask[index_order, :],deletion_matrix[index_order, :]
    return rows[index_order, :][indices, :],mask[index_order, :][indices, :],deletion_matrix[index_order, :][indices, :]