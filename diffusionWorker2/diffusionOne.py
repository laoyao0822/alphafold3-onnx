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
import torch.distributed as dist
import pathlib
import diffusionWorker2.misc.params as params
import diffusionWorker2.misc.feat_batch as feat_batch
from diffusionWorker2.network import diffusion_head

class diffusion():
    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(diffusion, self).__init__()
        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.diffusion_head = diffusion_head.DiffusionHead()

    def import_diffusion_head_params(self,model_path: pathlib.Path):
        params.import_diffusion_head_params(self.diffusion_head,model_path)

    def _apply_denoising_step(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_level_prev: torch.Tensor,
        # mask: torch.Tensor,
        noise_level: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        positions = diffusion_head.random_augmentation(
            positions=positions, mask=pred_dense_atom_mask
        )
        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)
        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        noise = noise_scale * \
            torch.randn(size=positions.shape, device=noise_scale.device)
        # noise = noise_scale
        positions_noisy = positions + noise
        positions_denoised = self.diffusion_head(
            single=embeddings['single'], pair=embeddings['pair'], target_feat=embeddings['target_feat'],
            token_index=batch.token_features.token_index,
            residue_index=batch.token_features.residue_index,
            asym_id=batch.token_features.asym_id,
            entity_id=batch.token_features.entity_id,
            sym_id=batch.token_features.sym_id,
            seq_mask = batch.token_features.mask,
            pred_dense_atom_mask = pred_dense_atom_mask,
            ref_ops=batch.ref_structure.positions,
            ref_mask=batch.ref_structure.mask,
            ref_element=batch.ref_structure.element,
            ref_charge=batch.ref_structure.charge,
            ref_atom_name_chars=batch.ref_structure.atom_name_chars,
            ref_space_uid=batch.ref_structure.ref_space_uid,
            acat_atoms_to_q_gather_idxs=batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
            acat_atoms_to_q_gather_mask=batch.atom_cross_att.token_atoms_to_queries.gather_mask,
            acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
            acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,
            acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
            acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,
            acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
            acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
            acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
            acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
            positions_noisy=positions_noisy,
            noise_level=t_hat,
            # batch=batch,
            # embeddings=embeddings,
            # use_conditioning=True
            )
        grad = (positions_noisy - positions_denoised) / t_hat
        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad
        return positions_out, noise_level
    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) :
        """Sample using denoiser on batch."""
        mask = batch.predicted_structure_info.atom_mask
        device = mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))
        noise_level = noise_levels[0]
        positions = torch.randn(
            mask.shape + (3,), device=device).contiguous()
        positions *= noise_level
        # noise_level = torch.tile(noise_levels[None, 0], (num_samples,))

        print("diffusion2 start sample diffusion",positions.shape)
        # print("bug shape ",batch.atom_cross_att.queries_to_token_atoms.shape)
        # positions_t=positions[self.rank].to(device=device,dtype=torch.float32).contiguous()

        for step_idx in range(self.diffusion_steps):
            positions, noise_level = self._apply_denoising_step(
            batch, embeddings, positions, noise_level, noise_levels[1 + step_idx])
        return positions

    def forward(self,batch: dict[str, torch.Tensor],single,pair,target_feat):
        batch = feat_batch.Batch.from_data_dict(batch)
        embeddings = {
            'pair': pair,
            'single': single,
            'target_feat': target_feat,  # type: ignore
        }
        return self._sample_diffusion(batch, embeddings)


class DiffusionOne(nn.Module):

    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(DiffusionOne, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps


        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.tracedFunction=None

        self.diffusion_head = diffusion_head.DiffusionHead()

    def _apply_denoising_step(
        self,
        single,pair,target_feat,
        token_index, residue_index, asym_id, entity_id, sym_id,
        seq_mask,
        pred_dense_atom_mask,
        ref_ops, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,
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
        # batch: feat_batch.Batch,
        # embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_level_prev: torch.Tensor,
        # mask: torch.Tensor,
        noise_level: torch.Tensor
    ) :
        # pred_dense_atom_mask = batch.predicted_structure_info.atom_mask

        positions = diffusion_head.random_augmentation(
            positions=positions, mask=pred_dense_atom_mask
        )

        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        noise = noise_scale * \
            torch.randn(size=positions.shape, device=noise_scale.device)
        # noise = noise_scale
        positions_noisy = positions + noise

        positions_denoised = self.diffusion_head(
            single=single, pair=pair, target_feat=target_feat,
            token_index=token_index, residue_index=residue_index,
            asym_id=asym_id,entity_id=entity_id, sym_id=sym_id,

            seq_mask=seq_mask,

            pred_dense_atom_mask = pred_dense_atom_mask,

            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,


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

            positions_noisy=positions_noisy,
            noise_level=t_hat,
            # batch=batch,
            # embeddings=embeddings,
            # use_conditioning=True
            )
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad

        return positions_out, noise_level


    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        # single,pair,target_feat,
        embeddings: dict[str, torch.Tensor],
    ) :
        """Sample using denoiser on batch."""

        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        device = pred_dense_atom_mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        noise_level = noise_levels[0]
        positions = torch.randn(
            pred_dense_atom_mask.shape + (3,), device=device).contiguous()
        positions *= noise_level

        # noise_level = torch.tile(noise_levels[None, 0], (num_samples,))


        print("diffusion2 start sample diffusion",positions.shape)

        for step_idx in range(self.diffusion_steps):
            positions, noise_level = self._apply_denoising_step(
                single=embeddings['single'], pair=embeddings['pair'], target_feat=embeddings['target_feat'],
                # single=single, pair=pair, target_feat=target_feat,
                token_index=batch.token_features.token_index,
                residue_index=batch.token_features.residue_index,
                asym_id=batch.token_features.asym_id,
                entity_id=batch.token_features.entity_id,
                sym_id=batch.token_features.sym_id,
                seq_mask=batch.token_features.mask,
                pred_dense_atom_mask=pred_dense_atom_mask,
                ref_ops=batch.ref_structure.positions,
                ref_mask=batch.ref_structure.mask,
                ref_element=batch.ref_structure.element,
                ref_charge=batch.ref_structure.charge,
                ref_atom_name_chars=batch.ref_structure.atom_name_chars,
                ref_space_uid=batch.ref_structure.ref_space_uid,
                acat_atoms_to_q_gather_idxs=batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
                acat_atoms_to_q_gather_mask=batch.atom_cross_att.token_atoms_to_queries.gather_mask,
                acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
                acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,

                acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,

                acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
                acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
                acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
                positions= positions,noise_level_prev= noise_level,noise_level= noise_levels[1 + step_idx]

            )
        return positions




    def forward(self,batch: dict[str, torch.Tensor],single,pair,target_feat):
        batch = feat_batch.Batch.from_data_dict(batch)
        embeddings = {
            'pair': pair,
            'single': single,
            'target_feat': target_feat,  # type: ignore
        }
        return self._sample_diffusion(batch, embeddings
                                      # single,pair,target_feat
                                      )






