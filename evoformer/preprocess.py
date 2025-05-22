import torch
from evoformer.network import featurization

def get_contact_matrix(
        gather_idxs_polymer_ligand,
        tokens_to_polymer_ligand_bonds_gather_mask,
        gather_idxs_ligand_ligand,
        tokens_to_ligand_ligand_bonds_gather_mask,
        num_tokens,
        # pair_activations: torch.Tensor
) -> torch.Tensor:
    contact_matrix = torch.zeros(
        (num_tokens, num_tokens), dtype=torch.float32)

    gather_mask_polymer_ligand = tokens_to_polymer_ligand_bonds_gather_mask.prod(dim=1)[:,
                                 None] * gather_idxs_polymer_ligand

    gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds_gather_mask.prod(dim=1)[:,
                                None] * gather_idxs_ligand_ligand

    gather_idxs = torch.concatenate(
        [gather_mask_polymer_ligand.to(dtype=torch.int64),
         gather_mask_ligand_ligand.to(dtype=torch.int64)]
    )

    contact_matrix[
        gather_idxs[:, 0], gather_idxs[:, 1]
    ] = 1.0
    # print("after contact_matrix",contact_matrix.shape)
    contact_matrix[
        gather_idxs[:, 0], gather_idxs[:, 1]
    ] = torch.tensor(
        1.0,
        dtype=contact_matrix.dtype,
        device=contact_matrix.device
    ).expand(gather_idxs.shape[0])
    # Because all the padded index's are 0's.
    contact_matrix[0, 0] = 0.0

    return contact_matrix[:, :, None]


def create_relative_encodingV2(
        # seq_features: features.TokenFeatures,
        token_index,
        residue_index,
        asym_id: torch.Tensor,
        entity_id: torch.Tensor,
        sym_id,
        max_relative_idx: int,
        max_relative_chain: int
) -> torch.Tensor:
    """Add relative position encodings."""
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
    final_token_offset = torch.where(residue_same, clipped_token_offset,
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

    rel_feats = [rel_pos, rel_token,
                 entity_id_same.to(dtype=rel_pos.dtype)[..., None], rel_chain]
    # rel_feats.append(rel_chain)
    return torch.concatenate(rel_feats, dim=-1)

def get_rel_feat(token_index,residue_index,asym_id,entity_id,sym_id,dtype,device='cpu'):
    rel_feat = featurization.create_relative_encodingV2(
        token_index=token_index,
        residue_index=residue_index,
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        max_relative_idx=32,
        max_relative_chain=2
    ).to(dtype=dtype,device=device)
    return rel_feat