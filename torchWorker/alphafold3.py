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
from torchWorker.network.pairformer import EvoformerBlock, PairformerBlock
from torchWorker.network.head import  ConfidenceHead
from torchWorker.network.template import TemplateEmbedding
import torchWorker.misc.feat_batch as feat_batch
from torchWorker.network import diffusion_head
from torchWorker.network import atom_cross_attention

from torchWorker.network import featurization
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
        self, num_res ) :
        """Embeds Templates and merges into pair activations."""
        # templates = batch.templates
        # asym_id = batch.token_features.asym_id
        self.template_embedding(num_res)


    def _embed_process_msa(
        self,num_res
    ) :
        """Processes MSA and returns updated pair activations."""

        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
         msa_block(num_res)


    def forward(
        self,num_res: int
    ) :
        # T1 = time.time()
        self._embed_template_pair(num_res)
        # T2 = time.time()
        # print(f"pair embedding time: {T2 - T1}")

        self._embed_process_msa(
            num_res
        )

        for pairformer_b in self.trunk_pairformer:
            pairformer_b(num_res)

        return

class AlphaFold3(nn.Module):

    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(AlphaFold3, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384
        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()

        self.evoformer = Evoformer()
        self.diffusion_head = diffusion_head.DiffusionHead()

        self.confidence_head = ConfidenceHead()
    def create_target_feat_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
        )

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat
    def _apply_denoising_step(
        self,
        batch: feat_batch.Batch,
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
        noise = noise_scale * \
            torch.randn(size=positions.shape, device=noise_scale.device)
        # noise = noise_scale
        positions_noisy = positions + noise

        positions_denoised = self.diffusion_head(positions_noisy=positions_noisy,
                                                 noise_level=t_hat,
                                                 batch=batch,
                                                 embeddings=embeddings,
                                                 use_conditioning=True)
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
        num_samples = self.num_samples
        device = mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device).contiguous()
        positions *= noise_levels[0]

        noise_level = torch.tile(noise_levels[None, 0], (num_samples,))


        print("start sample diffusion",positions.shape)
        # positions_t=positions[self.rank].to(device=device,dtype=torch.float32).contiguous()
        i =self.rank
        # print("sample :",i)
        #三卡运行
        if self.world_size==2:
            for idx in (3, 4):
                print("sample :", idx)
                for step_idx in range(self.diffusion_steps):
                    positions[idx], noise_level[idx] = self._apply_denoising_step(
                        batch, embeddings, positions[idx], noise_level[idx], mask, noise_levels[1 + step_idx])
            dist.isend(tensor=positions[3], dst=0)
            dist.isend(tensor=positions[4], dst=0)
        elif self.world_size>2 and self.world_size<self.num_samples:
            if self.rank == 1:
                for idx in (3, 4):
                    print("sample :", idx)
                    for step_idx in range(self.diffusion_steps):
                        positions[idx], noise_level[idx] = self._apply_denoising_step(
                            batch, embeddings, positions[idx], noise_level[idx], mask, noise_levels[1 + step_idx])
                    # dist.isend(tensor=positions[idx], dst=0)
            if self.rank == 2:
                idx = 2
                print("sample :", idx)
                for step_idx in range(self.diffusion_steps):
                    positions[idx], noise_level[idx] = self._apply_denoising_step(
                        batch, embeddings, positions[idx], noise_level[idx], mask, noise_levels[1 + step_idx])
            if self.rank == 2:
                print("send", self.rank, positions.dtype)
                dist.send(tensor=positions[2].to(dtype=torch.float32).contiguous(), dst=0)
            dist.barrier(group=dist.group.WORLD)
            if self.rank == 1:
                print("send", self.rank, positions.dtype)
                dist.send(tensor=positions[3], dst=0)
                dist.send(tensor=positions[4], dst=0)

        elif self.world_size>=self.num_samples:
            if self.rank<self.num_samples:
                idx = self.rank
                print("sample :", idx)
                for step_idx in range(self.diffusion_steps):
                    positions[idx], noise_level[idx] = self._apply_denoising_step(
                        batch, embeddings, positions[idx], noise_level[idx], mask, noise_levels[1 + step_idx])
                positions[idx] = positions[idx].to(dtype=torch.float32).contiguous()
                # dist.barrier(group=dist.group.WORLD)

                print("send", self.rank, positions.dtype)
                dist.send(tensor=positions[self.rank].to(dtype=torch.float32), dst=0)

        # gather_list =[]
        # dist.gather(
        #     tensor=positions[self.rank],  # 当前进程要发送的数据
        #     gather_list=gather_list,  # 仅在 Rank 0 有效，其他进程传空列表
        #     dst=0  # 目标 Rank 是 0
        # )




    def forward(self,batch: dict[str, torch.Tensor]):
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        target_feat = self.create_target_feat_embedding(batch)
        print("device:",target_feat.device)
        print("num_res:",num_res)
        pair= torch.zeros([num_res, num_res, self.evoformer_pair_channel], device=target_feat.device, dtype=torch.bfloat16)
        single=torch.zeros(
            [num_res, self.evoformer_seq_channel], device=target_feat.device,dtype=torch.bfloat16
        )

        for _ in range(self.num_recycles + 1):
           self.evoformer(num_res=num_res)

        # print("pair:", pair.shape)
        # print("single:", single.shape)
        print("recv diffusion data")
        dist.recv(tensor=pair, src=0)
        dist.recv(tensor=single, src=0)
        embeddings = {
            'pair': pair,
            'single': single,
            'target_feat': target_feat,  # type: ignore
        }
        self._sample_diffusion(batch, embeddings)
        print("diffusion over")
        for _ in range(self.num_samples):
            self.confidence_head(num_res=num_res)

        return
        # self._sample_diffusion(batch, embeddings)




