# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
import time

import torch
import torch.nn as nn
from torch import distributed as dist

from torchfold3.misc import feat_batch, features
from torchfold3.network import featurization
from torchfold3.network.pairformer import EvoformerBlock, PairformerBlock
from torchfold3.network.head import DistogramHead, ConfidenceHead
from torchfold3.network.template import TemplateEmbedding
from torchfold3.network import atom_cross_attention
from torchfold3.network import diffusion_head


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

        self.left_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)
        self.right_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)

        self.prev_embedding_layer_norm = nn.LayerNorm(self.pair_channel)
        self.prev_embedding = nn.Linear(
            self.pair_channel, self.pair_channel, bias=False)

        self.c_rel_feat = 139
        self.position_activations = nn.Linear(
            self.c_rel_feat, self.pair_channel, bias=False)

        self.bond_embedding = nn.Linear(
            1, self.pair_channel, bias=False)

        self.template_embedding = TemplateEmbedding(
            pair_channel=self.pair_channel)

        self.msa_activations = nn.Linear(34, self.msa_channel, bias=False)
        self.extra_msa_target_feat = nn.Linear(
            self.c_target_feat, self.msa_channel, bias=False)
        self.msa_stack = nn.ModuleList(
            [EvoformerBlock() for _ in range(self.msa_stack_num_layer)])

        self.single_activations = nn.Linear(
            self.c_target_feat, self.seq_channel, bias=False)

        self.prev_single_embedding_layer_norm = nn.LayerNorm(self.seq_channel)
        self.prev_single_embedding = nn.Linear(
            self.seq_channel, self.seq_channel, bias=False)

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock(with_single=True) for _ in range(self.pairformer_num_layer)])

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            max_relative_idx,
            max_relative_chain,
        ).to(dtype=pair_activations.dtype)

        pair_activations += self.position_activations(rel_feat)
        return pair_activations

    def _seq_pair_embedding(
        self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated Pair embedding from sequence."""
        left_single = self.left_single(target_feat)[:, None]
        right_single = self.right_single(target_feat)[None]
        pair_activations = left_single + right_single

        mask = token_features.mask

        pair_mask = (mask[:, None] * mask[None, :]).to(dtype=left_single.dtype)

        return pair_activations, pair_mask

    def _embed_bonds(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations."""
        # Construct contact matrix.
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device)

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).to(
                dtype=gather_idxs_polymer_ligand.dtype)[:, None]
        )
        # If valid mask then it will be all 1's, so idxs should be unchanged.
        gather_idxs_polymer_ligand = (
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        )

        tokens_to_ligand_ligand_bonds = (
            batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
        )
        gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
            dim=1
        ).to(dtype=gather_idxs_ligand_ligand.dtype)[:, None]
        gather_idxs_ligand_ligand = (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )

        gather_idxs = torch.concatenate(
            [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
        )
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0

        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0

        bonds_act = self.bond_embedding(contact_matrix[:, :, None])

        return pair_activations + bonds_act

    def _embed_template_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        templates = batch.templates
        asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        multichain_mask = (asym_id[:, None] ==
                           asym_id[None, :]).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask
        )

        return pair_activations + template_act

    def _embed_process_msa(
        self, msa_batch: features.MSA,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype

        msa_batch = featurization.shuffle_msa(msa_batch)
        msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat)[None]

        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )

        return pair_activations

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prev: dict[str, torch.Tensor],
        target_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        time1=time.time()
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )

        pair_activations += self.prev_embedding(
            self.prev_embedding_layer_norm(prev['pair']))

        pair_activations = self._relative_encoding(batch, pair_activations)
        # t_c=target_feat.clone()
        # single_c=prev['single'].clone()
        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )





        single_activations = self.single_activations(target_feat)
        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single']))

        print("evoformer1 cost time:", time.time() - time1)
        # exit(0)

        pair_activations = self._embed_template_pair(
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
        )

        pair_activations = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
        )
        # if torch.allclose(t_c, target_feat, rtol=1e-6):
        #     print("target_feat 张量没有变化")
        # else:
        #     print("target_feat 张量发生了变化")
        # if torch.allclose(prev['single'], single_c, rtol=1e-6):
        #     print("single 张量没有变化")
        # else:
        #     print("single 张量发生了变化")


        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = pairformer_b(
                pair_activations, pair_mask, single_activations, batch.token_features.mask)
        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output


class AlphaFold3(nn.Module):

    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(AlphaFold3, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()

        self.evoformer = Evoformer()

        self.diffusion_head = diffusion_head.DiffusionHead()

        self.distogram_head = DistogramHead()
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
    ) -> dict[str, torch.Tensor]:
        """Sample using denoiser on batch."""

        mask = batch.predicted_structure_info.atom_mask
        num_samples = self.num_samples

        device = mask.device

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device,dtype=torch.float32)
        # positions *= noise_levels[0]
        #noise_level torch.Size([5]) noise_levels torch.Size([201])
        noise_level = torch.tile(noise_levels[None, 0], (num_samples,))
        #noise_level [2560., 2560., 2560., 2560., 2560.]
        # print("noise_level", noise_level)
        time1=time.time()
        # print("start sample diffusion",self.num_samples,"-",positions.shape)
        # print("noise_level", noise_level.shape,"noise_levels",noise_levels.shape)

        if self.world_size==1:
            # for sample_idx in range(self.num_samples):
            #     print("sample_idx", sample_idx)
            #     for step_idx in range(self.diffusion_steps):
            #         positions[sample_idx], noise_level[sample_idx] = self._apply_denoising_step(
            #             batch, embeddings, positions[sample_idx], noise_level[sample_idx], mask,
            #             noise_levels[1 + step_idx])
            for sample_idx in range(self.num_samples):
                print("sample_idx", sample_idx)
                # print("noise_level",noise_levels[None, 0])
                noise_level = noise_levels[0]
                print("noise_level", noise_level)
                position_c=torch.randn(mask.shape + (3,), device=device,dtype=torch.float32)

                position_c*=noise_level
                for step_idx in range(self.diffusion_steps):
                    position_c, noise_level= self._apply_denoising_step(
                        batch, embeddings, position_c, noise_level, mask,
                        noise_levels[1 + step_idx])
                positions[sample_idx]=position_c


        elif self.world_size==2:
            rec_1 = dist.irecv(tensor=positions[3], src=1)
            rec_2 = dist.irecv(tensor=positions[4], src=1)
            for sample_idx in (0, 1,2):
                print("sample_idx", sample_idx)
                for step_idx in range(self.diffusion_steps):
                    positions[sample_idx], noise_level[sample_idx] = self._apply_denoising_step(
                        batch, embeddings, positions[sample_idx], noise_level[sample_idx], mask,
                        noise_levels[1 + step_idx])

            # print("positions", positions.shape, positions.dtype)
            rec_1.wait()
            rec_2.wait()
        elif self.world_size==3 or self.world_size==4:
            for sample_idx in (0, 1):
                print("sample_idx", sample_idx)
                for step_idx in range(self.diffusion_steps):
                    positions[sample_idx], noise_level[sample_idx] = self._apply_denoising_step(
                        batch, embeddings, positions[sample_idx], noise_level[sample_idx], mask,
                        noise_levels[1 + step_idx])
            # print("positions", positions.shape, positions.dtype)
            dist.recv(tensor=positions[2], src=2)
            dist.barrier(group=dist.group.WORLD)
            dist.recv(tensor=positions[3], src=1)
            dist.recv(tensor=positions[4], src=1)
        elif self.world_size==5:
            sample_idx=0
            print("sample_idx", sample_idx)
            for step_idx in range(self.diffusion_steps):
                positions[sample_idx], noise_level[sample_idx] = self._apply_denoising_step(
                    batch, embeddings, positions[sample_idx], noise_level[sample_idx], mask,
                    noise_levels[1 + step_idx])
            # dist.barrier(group=dist.group.WORLD)
            positions=positions.contiguous()
            recs=[]
            for sample_idx in range(1, self.num_samples):
                # print("rec sample_idx", sample_idx)
                recs.append(dist.irecv(tensor=positions[sample_idx], src=sample_idx))
            for rec in recs:
                rec.wait()

        print("sample diffusion time:",time.time()-time1)

        # if torch.allclose(mask_c, mask, rtol=1e-5):
        #     print("target_feat 张量没有变化")
        # else:
        #     print("target_feat 张量发生了变化")

        #single 张量没有变化
        #pair 张量没有变化
        #target_feat 张量没有变化
        #mask 张量没有变化

        final_dense_atom_mask = torch.tile(mask[None], (num_samples, 1, 1))

        return {'atom_positions': positions, 'mask': final_dense_atom_mask}

    def forward(self, batch: dict[str, torch.Tensor]
                ,embeddings,positions
                ) :
        batch = feat_batch.Batch.from_data_dict(batch)
        # num_res = batch.num_res
        # #target_feat torch.Size([37, 447])
        # time1 = time.time()
        # target_feat = self.create_target_feat_embedding(batch)
        # print("create target feat cost time:", time.time() - time1)
        # # return target_feat
        # print("target_feat", target_feat.shape)
        # target_feat1=self.create_target_feat_embedding(batch)
        # embeddings = {
        #     'pair': torch.zeros(
        #         [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
        #         dtype=torch.float32,
        #     ),
        #     'single': torch.zeros(
        #         [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
        #     ),
        #     'target_feat': target_feat,  # type: ignore
        # }
        # time1 = time.time()
        # for _ in range(self.num_recycles + 1):
        #     # ref:
        #     # Number of recycles is number of additional forward trunk passes.
        #     # num_iter = self.config.num_recycles + 1
        #     # embeddings, _ = hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))
        #
        #     embeddings = self.evoformer(
        #         batch=batch,
        #         prev=embeddings,
        #         target_feat=target_feat
        #     )
        # # print("evoformer cost time:",time.time()-time1)
        # c_pair=embeddings['pair'].contiguous()
        # c_single=embeddings['single'].contiguous()
        # # print("dtype",c_pair.dtype,c_single.dtype)
        # # print("shape",c_pair.shape,c_single.shape)
        #
        # for send_rank in range(1,min(self.world_size,self.num_samples)):
        #     print("send",send_rank)
        #     dist.send(tensor=c_pair, dst=send_rank)
        #     dist.send(tensor=c_single, dst=send_rank)

        # sample_mask = batch.predicted_structure_info.atom_mask
        # # samples = self._sample_diffusion(batch, embeddings)
        # final_dense_atom_mask = torch.tile(sample_mask[None], (self.num_samples, 1, 1))
        # samples={'atom_positions': positions, 'mask': final_dense_atom_mask}
        # time2=time.time()
        # confidence_output_per_sample = []
        # print("positions shape:",positions.shape)
        # positions_c=positions.clone()
        # for sample_dense_atom_position in samples['atom_positions']:
        #     print("sample_dense_atom_position", sample_dense_atom_position.shape)
        #     confidence_output_per_sample.append(self.confidence_head(
        #         dense_atom_positions=sample_dense_atom_position,
        #         embeddings=embeddings,
        #         seq_mask=batch.token_features.mask,
        #         token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
        #         asym_id=batch.token_features.asym_id
        #     ))
        # print("confidence_head cost time:",time.time()-time2)
        # print("confidence_output_per_sample:",confidence_output_per_sample[0].shape)
        confidence_output = {}
        # for key in confidence_output_per_sample[0]:
        #     confidence_output[key] = torch.stack(
        #         [sample[key] for sample in confidence_output_per_sample], dim=0)
        #
        # if torch.allclose(positions_c, positions, rtol=1e-5):
        #     print("positions 张量没有变化")
        # else:
        #     print("positions_c 张量发生了变化")
        # exit(0)

        distogram = self.distogram_head(batch, embeddings)
        return distogram

        # if torch.allclose(target_feat_clone, embeddings['target_feat'], rtol=1e-5):
        #     print("target_feat 张量没有变化")
        # else:
        #     print("target_feat 张量发生了变化")

        # return {
        #     'diffusion_samples': samples,
        #     'distogram': distogram,
        #     **confidence_output,
        # }
