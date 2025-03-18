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
from torch import Tensor
from diffusionWorker.network import featurization
from diffusionWorker.network import atom_cross_attention
from diffusionWorker.network import diffusion_head



class AlphaFold3(nn.Module):
# ref:
# def make_model_config(
#     *,
#     flash_attention_implementation: attention.Implementation = 'triton',
#     num_diffusion_samples: int = 5,
#     num_recycles: int = 10,
#     return_embeddings: bool = False,
# )
    # eval: SampleConfig = base_config.autocreate(
    #     num_samples=5,
    #     steps=200,
    # )
    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(AlphaFold3, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()

        self.diffusion_head = diffusion_head.DiffusionHead()

    def create_target_feat_embedding(self,
            aatype,
            profile,deletion_mean,
            ref_ops, ref_mask, ref_element, ref_charge,
            ref_atom_name_chars, ref_space_uid,
            pred_dense_atom_mask,
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
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            # batch=batch,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat

    def _apply_denoising_step(
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
        embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_level_prev: torch.Tensor,
        mask: torch.Tensor,
        noise_level: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        positions = diffusion_head.random_augmentation(
            positions=positions, mask=mask
        )

        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        # noise = noise_scale *torch.randn(size=positions.shape, device=noise_scale.device)
        noise = noise_scale
        positions_noisy = positions + noise

        positions_denoised = self.diffusion_head(
            token_index=token_index, residue_index=residue_index, asym_id=asym_id,
            entity_id=entity_id, sym_id=sym_id,
            seq_mask=seq_mask, pred_dense_atom_mask=pred_dense_atom_mask,
            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
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
            positions_noisy=positions_noisy,
            noise_level=t_hat,
            # batch=batch,
            embeddings=embeddings,
            use_conditioning=True)
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad

        return positions_out, noise_level

    def _sample_diffusion(
        self,
        pred_dense_atom_mask: torch.Tensor,
        token_index, residue_index, asym_id, entity_id, sym_id,
        seq_mask,
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
        embeddings: dict[str, torch.Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Sample using denoiser on batch."""

        # pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        num_samples = self.num_samples

        device = pred_dense_atom_mask.device

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        positions = torch.randn(
            (num_samples,) + pred_dense_atom_mask.shape + (3,), device=device)
        positions *= noise_levels[0]

        noise_level = torch.tile(noise_levels[None, 0], (num_samples,))

        for sample_idx in range(num_samples):
            for step_idx in range(self.diffusion_steps):
                positions[sample_idx], noise_level[sample_idx] = self._apply_denoising_step(
                    token_index=token_index, residue_index=residue_index, asym_id=asym_id,
                    entity_id=entity_id, sym_id=sym_id,
                    seq_mask=seq_mask, pred_dense_atom_mask=pred_dense_atom_mask,
                    ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
                    ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
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
                    embeddings=embeddings,positions= positions[sample_idx],
                    noise_level_prev=noise_level[sample_idx],mask= pred_dense_atom_mask,noise_level= noise_levels[1 + step_idx])

        final_dense_atom_mask = torch.tile(pred_dense_atom_mask[None], (num_samples, 1, 1))

        return positions, final_dense_atom_mask

    def forward(self,
                aatype,seq_mask, token_index, residue_index, asym_id, entity_id, sym_id,
                profile, deletion_mean,
                pred_dense_atom_mask,


                acat_atoms_to_q_input_shape,
                acat_atoms_to_q_gather_idxs,
                acat_atoms_to_q_gather_mask,

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

                ref_ops, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,
                ) :
        num_res = aatype.shape[-1]

        target_feat = self.create_target_feat_embedding(
            aatype=aatype,profile=profile,deletion_mean=deletion_mean,
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
        )
        print("device:", target_feat.device)
        print("num_res:", num_res)
        pair = torch.zeros([num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                           dtype=torch.bfloat16)
        single = torch.zeros(
            [num_res, self.evoformer_seq_channel], device=target_feat.device, dtype=torch.bfloat16
        )

        print("recv diffusion data")
        dist.recv(tensor=pair, src=0)
        dist.recv(tensor=single, src=0)
        embeddings = {
            'pair': pair,
            'single': single,
            'target_feat': target_feat,  # type: ignore
        }
        atom_positions, final_dense_atom_mask = self._sample_diffusion(
                pred_dense_atom_mask=pred_dense_atom_mask,
                token_index=token_index, residue_index=residue_index, asym_id=asym_id,
                    entity_id=entity_id, sym_id=sym_id,
                    seq_mask=seq_mask,
                    ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
                    ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
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
                    embeddings=embeddings)

        return  None
