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

from diffusionWorker.network import featurization, utils
from diffusionWorker.network.diffusion_transformer import DiffusionTransformer, DiffusionTransition
from diffusionWorker.network.atom_cross_attention import AtomCrossAttEncoder, AtomCrossAttDecoder
from diffusionWorker.network.layer_norm import LayerNorm

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


class FourierEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super(FourierEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not hasattr(self, "weight") or not hasattr(self, "bias"):
            raise RuntimeError("FourierEmbeddings not initialized")            

        return torch.cos(2 * torch.pi * (x[..., None] * self.weight + self.bias))


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
    return (
        SIGMA_DATA
        * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
    )


def random_rotation(device, dtype):
    # Create a random rotation (Gram-Schmidt orthogonalization of two
    # random normal vectors)
    v0, v1 = torch.randn(size=(2, 3), dtype=dtype, device=device)
    e0 = v0 / torch.clamp(torch.linalg.norm(v0), min=1e-10)
    v1 = v1 - e0 * torch.dot(v1, e0)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1), min=1e-10)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2])


def random_augmentation(
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply random rigid augmentation.

    Args:
      positions: atom positions of shape (<common_axes>, 3)
      mask: per-atom mask of shape (<common_axes>,)

    Returns:
      Transformed positions with the same shape as input positions.
    """

    center = utils.mask_mean(
        mask[..., None], positions, dim=(-2, -3), keepdim=True, eps=1e-6
    )
    rot = random_rotation(device=positions.device, dtype=positions.dtype)
    translation = torch.randn(
        size=(3,), dtype=positions.dtype, device=positions.device)

    augmented_positions = (
        torch.einsum(
            '...i,ij->...j',
            positions - center,
            rot,
        )
        + translation
    )
    return augmented_positions * mask[..., None]


class DiffusionHead(nn.Module):
    def __init__(self):
        super(DiffusionHead, self).__init__()

        self.c_act = 768
        self.pair_channel = 128
        self.seq_channel = 384

        self.c_pair_cond_initial = 267
        self.pair_cond_initial_norm = LayerNorm(
            self.c_pair_cond_initial, bias=False)
        self.pair_cond_initial_projection = nn.Linear(
            self.c_pair_cond_initial, self.pair_channel, bias=False)

        self.pair_transition_0 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)
        self.pair_transition_1 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)

        self.c_single_cond_initial = 831
        self.single_cond_initial_norm = LayerNorm(
            self.c_single_cond_initial, bias=False)
        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial, self.seq_channel, bias=False)

        self.c_noise_embedding = 256
        self.noise_embedding_initial_norm = LayerNorm(
            self.c_noise_embedding, bias=False)
        self.noise_embedding_initial_projection = nn.Linear(
            self.c_noise_embedding, self.seq_channel, bias=False)

        self.single_transition_0 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)
        self.single_transition_1 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)

        self.atom_cross_att_encoder = AtomCrossAttEncoder(per_token_channels=self.c_act,
                                                          with_token_atoms_act=True,
                                                          with_trunk_pair_cond=True,
                                                          with_trunk_single_cond=True)

        self.single_cond_embedding_norm = LayerNorm(
            self.seq_channel, bias=False)
        self.single_cond_embedding_projection = nn.Linear(
            self.seq_channel, self.c_act, bias=False)

        self.transformer = DiffusionTransformer()

        self.output_norm = LayerNorm(self.c_act, bias=False)

        self.atom_cross_att_decoder = AtomCrossAttDecoder()

        self.fourier_embeddings = FourierEmbeddings(dim=256)

    def _conditioning(
        self,
        token_index,residue_index,asym_id,entity_id,sym_id,
        single,pair,target_feat,
        noise_level: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_embedding = use_conditioning * single
        pair_embedding = use_conditioning * pair

        rel_features = featurization.create_relative_encodingV2(
            token_index,residue_index,asym_id,entity_id,sym_id, max_relative_idx=32, max_relative_chain=2
        ).to(dtype=pair_embedding.dtype)
        features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)

        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )

        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)

        target_feat = target_feat
        features_1d = torch.concatenate(
            [single_embedding, target_feat], dim=-1)
        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d))

        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA)
        )

        single_cond += self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )

        single_cond += self.single_transition_0(single_cond)
        single_cond += self.single_transition_1(single_cond)

        return single_cond, pair_cond

    def forward(
        self,

        token_index, residue_index, asym_id, entity_id, sym_id,
        seq_mask,pred_dense_atom_mask,
        ref_ops, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,
        acat_atoms_to_q_gather_idxs,
        acat_atoms_to_q_gather_mask,
        acat_atoms_to_q_input_shape,
        acat_q_to_k_gather_idxs,
        acat_q_to_k_gather_mask,
        acat_q_to_k_input_shape,
        acat_t_to_q_gather_idxs,
        acat_t_to_q_gather_mask,
        acat_t_to_q_input_shape,
        acat_q_to_atom_gather_idxs,
        acat_q_to_atom_gather_mask,
        acat_q_to_atom_input_shape,
        acat_t_to_k_gather_idxs,
        acat_t_to_k_gather_mask,
        acat_t_to_k_input_shape,
        # batch: feat_batch.Batch,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool
    ) -> torch.Tensor:
        # Get conditioning
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            token_index=token_index, residue_index=residue_index,
            asym_id=asym_id, entity_id=entity_id,sym_id=sym_id,
            single=embeddings['single'],pair=embeddings['pair'],target_feat=embeddings['target_feat'],
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # Extract features
        #atom_mask = pred_dense_atom_mask

        # Position features
        act = positions_noisy * pred_dense_atom_mask[..., None]
        act = act / torch.sqrt(noise_level**2 + SIGMA_DATA**2)

        enc = self.atom_cross_att_encoder(

            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
            pred_dense_atom_mask=pred_dense_atom_mask,
            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_atoms_to_q_input_shape=acat_atoms_to_q_input_shape,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            acat_q_to_k_input_shape=acat_q_to_k_input_shape,
            acat_t_to_q_gather_idxs=acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask=acat_t_to_q_gather_mask,
            acat_t_to_q_input_shape=acat_t_to_q_input_shape,
            acat_q_to_atom_gather_idxs=acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask=acat_q_to_atom_gather_mask,
            acat_q_to_atom_input_shape=acat_q_to_atom_input_shape,
            acat_t_to_k_gather_idxs=acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask=acat_t_to_k_gather_mask,
            acat_t_to_k_input_shape=acat_t_to_k_input_shape,

            token_atoms_act=act,
            trunk_single_cond=embeddings['single'],
            trunk_pair_cond=trunk_pair_cond,
        )
        act = enc.token_act

        act += self.single_cond_embedding_projection(
            self.single_cond_embedding_norm(trunk_single_cond)
        )

        act = self.transformer(
            act=act,
            single_cond=trunk_single_cond,
            mask=seq_mask,
            pair_cond=trunk_pair_cond,
        )
        act = self.output_norm(act)

        # (Possibly) atom-granularity decoder
        position_update = self.atom_cross_att_decoder(
            # batch=batch,
            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_atoms_to_q_input_shape=acat_atoms_to_q_input_shape,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            acat_q_to_k_input_shape=acat_q_to_k_input_shape,
            acat_q_to_atom_gather_idxs=acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask=acat_q_to_atom_gather_mask,
            acat_q_to_atom_input_shape=acat_q_to_atom_input_shape,
            token_act=act,
            enc=enc,
        )

        skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
        out_scaling = (
            noise_level * SIGMA_DATA /
            torch.sqrt(noise_level**2 + SIGMA_DATA**2)
        )

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * pred_dense_atom_mask[..., None]
