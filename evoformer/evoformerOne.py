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

from evoformer.misc import feat_batch, features
from evoformer.network import featurization
from evoformer.network.pairformer import EvoformerBlock, PairformerBlock
from evoformer.network.template import TemplateEmbedding
# from evoformer.network import atom_cross_attention


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

    # def _relative_encoding(
    #     self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    # ) -> torch.Tensor:
    #     max_relative_idx = 32
    #     max_relative_chain = 2
    #
    #     rel_feat = featurization.create_relative_encoding(
    #         batch.token_features,
    #         max_relative_idx,
    #         max_relative_chain,
    #     ).to(dtype=pair_activations.dtype)
    #
    #     pair_activations += self.position_activations(rel_feat)
    #     return pair_activations
    def _relative_encoding(
        self, token_index,residue_index,asym_id,entity_id,sym_id,
            pair_activations: torch.Tensor
    ) -> torch.Tensor:
        # max_relative_idx = 32
        # max_relative_chain = 2

        rel_feat = featurization.create_relative_encodingV2(
            token_index=token_index,
            residue_index=residue_index,
            asym_id=asym_id,
            entity_id=entity_id,
            sym_id=sym_id,
            # max_relative_idx,
            # max_relative_chain,
            max_relative_idx=32,
            max_relative_chain=2
        ).to(dtype=pair_activations.dtype)

        pair_activations += self.position_activations(rel_feat)
        return pair_activations
    # def _seq_pair_embedding(
    #     self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Generated Pair embedding from sequence."""
    #     left_single = self.left_single(target_feat)[:, None]
    #     right_single = self.right_single(target_feat)[None]
    #     pair_activations = left_single + right_single
    #
    #     mask = token_features.mask
    #
    #     pair_mask = (mask[:, None] * mask[None, :]).to(dtype=left_single.dtype)
    #
    #     return pair_activations, pair_mask

    def _seq_pair_embedding(
        self, mask, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated Pair embedding from sequence."""
        # left_single = self.left_single(target_feat)[:, None]
        # right_single = self.right_single(target_feat)[None]
        # pair_activations = left_single + right_single
        # pair_mask = (mask[:, None] * mask[None, :]).to(dtype=target_feat.dtype)
        return self.left_single(target_feat)[:, None]+self.right_single(target_feat)[None], (mask[:, None] * mask[None, :]).to(dtype=target_feat.dtype)

    # def _embed_bonds(
    #     self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    # ) -> torch.Tensor:
    #     """Embeds bond features and merges into pair activations."""
    #     # Construct contact matrix.
    #     num_tokens = batch.token_features.token_index.shape[0]
    #     contact_matrix = torch.zeros(
    #         (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device)
    #
    #     tokens_to_polymer_ligand_bonds = (
    #         batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
    #     )
    #     gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
    #     gather_mask_polymer_ligand = (
    #         tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).to(
    #             dtype=gather_idxs_polymer_ligand.dtype)[:, None]
    #     )
    #     # If valid mask then it will be all 1's, so idxs should be unchanged.
    #     gather_idxs_polymer_ligand = (
    #         gather_idxs_polymer_ligand * gather_mask_polymer_ligand
    #     )
    #
    #     tokens_to_ligand_ligand_bonds = (
    #         batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
    #     )
    #     gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
    #     gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
    #         dim=1
    #     ).to(dtype=gather_idxs_ligand_ligand.dtype)[:, None]
    #     gather_idxs_ligand_ligand = (
    #         gather_idxs_ligand_ligand * gather_mask_ligand_ligand
    #     )
    #
    #     gather_idxs = torch.concatenate(
    #         [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
    #     )
    #     contact_matrix[
    #         gather_idxs[:, 0], gather_idxs[:, 1]
    #     ] = 1.0
    #
    #     # Because all the padded index's are 0's.
    #     contact_matrix[0, 0] = 0.0
    #
    #     bonds_act = self.bond_embedding(contact_matrix[:, :, None])
    #
    #     return pair_activations + bonds_act
    def _embed_bonds(
        self,
            gather_idxs_polymer_ligand,
            tokens_to_polymer_ligand_bonds_gather_mask,
            gather_idxs_ligand_ligand,
            tokens_to_ligand_ligand_bonds_gather_mask,
            num_tokens,
            pair_activations: torch.Tensor
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations.
        tokens_to_polymer_ligand_bonds_gather_idxs,
            tokens_to_polymer_ligand_bonds_gather_mask,
            tokens_to_ligand_ligand_bonds_gather_mask,
            tokens_to_ligand_ligand_bonds_gather_idxs,
            num_tokens,
            pair_activations: torch.Tensor
        """
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device)

        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds_gather_mask.prod(dim=1).to(
                dtype=gather_idxs_polymer_ligand.dtype)[:, None]
        )
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds_gather_mask.prod(
            dim=1
        ).to(dtype=gather_idxs_ligand_ligand.dtype)[:, None]

        gather_idxs = torch.concatenate(
            [(
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        ), (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )]
        )
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0
        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0
        # bonds_act = self.bond_embedding(contact_matrix[:, :, None])
        return pair_activations + self.bond_embedding(contact_matrix[:, :, None])


    def _embed_template_pair(
        self,
        # batch: feat_batch.Batch,
        asym_id,
        template_aatype, template_atom_positions, template_atom_mask,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        # templates = batch.templates
        # asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        multichain_mask = (asym_id[:, None] ==
                           asym_id[None, :]).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            # templates=templates,
            template_aatype=template_aatype, template_atom_positions=template_atom_positions,
            template_atom_mask=template_atom_mask,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask
        )

        return pair_activations + template_act

    def _embed_process_msa(
        self,
        # msa_batch: features.MSA,
        rows, mask, deletion_matrix,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype
        # rows=msa_batch.rows
        # mask=msa_batch.mask
        # deletion_matrix=msa_batch.deletion_matrix

        # msa_batch = featurization.shuffle_msa(msa_batch)
        # msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)
        rows, mask, deletion_matrix= featurization.shuffle_msa_runcate(rows, mask, deletion_matrix,num_msa=self.num_msa)

        msa_mask = mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(rows,deletion_matrix).to(dtype=dtype)

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
        # batch,
        rows, mask, deletion_matrix,

        token_index, residue_index, asym_id, entity_id, sym_id,

        seq_mask,

        t_o_pol_idx, t_o_pol_mask, t_o_lig_masks, t_o_lig_idxs,
        template_aatype, template_atom_positions, template_atom_mask,
        single,pair,
        # prev: dict[str, torch.Tensor],
        target_feat: torch.Tensor
    ) :
        # target_feat_c=target_feat.clone()

        # pair=prev['pair']
        # single=prev['single']
        num_tokens = token_index.shape[0]
        # seq_mask=batch.token_features.mask
        # template_aatype = batch.templates.aatype
        # template_atom_positions = batch.templates.atom_positions
        # template_atom_mask = batch.templates.atom_mask


        pair_activations, pair_mask = self._seq_pair_embedding(
            seq_mask, target_feat
        )

        pair_activations += self.prev_embedding(
            self.prev_embedding_layer_norm(pair))

        # pair_activations = self._relative_encoding(batch, pair_activations)
        pair_activations=self._relative_encoding(token_index=token_index,
                                                 residue_index=residue_index,
                                                 asym_id=asym_id,
                                                 entity_id=entity_id,
                                                 sym_id=sym_id,
                                                 pair_activations=pair_activations)
        # pair_activations = self._embed_bonds(
        #     batch=batch, pair_activations=pair_activations
        # )
        pair_activations = self._embed_bonds(
            gather_idxs_polymer_ligand=t_o_pol_idx, tokens_to_polymer_ligand_bonds_gather_mask=t_o_pol_mask,
            gather_idxs_ligand_ligand=t_o_lig_idxs, tokens_to_ligand_ligand_bonds_gather_mask=t_o_lig_masks, num_tokens=num_tokens,
            pair_activations=pair_activations
        )

        single_activations = self.single_activations(target_feat)
        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(single))

        #
        pair_activations = self._embed_template_pair(
            asym_id=asym_id, template_aatype=template_aatype,
            template_atom_positions=template_atom_positions, template_atom_mask=template_atom_mask,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
        )

        pair_activations = self._embed_process_msa(
            # msa_batch=batch.msa,
            rows, mask, deletion_matrix,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
        )



        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = pairformer_b(
                pair_activations, pair_mask, single_activations, seq_mask)
        # if torch.allclose(target_feat,target_feat_c,1e-5):
        #     print("target feat  not change")

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output
        # return single,pair

class EvoFormerOne(nn.Module):

    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(EvoFormerOne, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples



        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384


        self.evoformer = Evoformer()


    def forward(self, batch: dict[str, torch.Tensor],target_feat) -> dict[str, torch.Tensor]:
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        #target_feat torch.Size([37, 447])
        
        # target_feat1=self.create_target_feat_embedding(batch)
        embeddings = {
            'pair': torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=torch.float32,
            ),
            'single': torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
            ),
            'target_feat': target_feat,  # type: ignore
        }
        pair= torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=torch.float32,
            )
        single=torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
            )
        # time1=time.time()
        for _ in range(self.num_recycles + 1):
        # for _ in range(0+1):
            # ref:
            # Number of recycles is number of additional forward trunk passes.
            # num_iter = self.config.num_recycles + 1
            # embeddings, _ = hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))
            embeddings = self.evoformer(
                # batch=batch,
                rows=batch.msa.rows,
                mask = batch.msa.mask,
                deletion_matrix = batch.msa.deletion_matrix,

                token_index=batch.token_features.token_index,
                residue_index = batch.token_features.residue_index,
                asym_id = batch.token_features.asym_id,
                entity_id = batch.token_features.entity_id,
                sym_id = batch.token_features.sym_id,
                seq_mask=batch.token_features.mask,

                t_o_pol_idx=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
                t_o_pol_mask = batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
                t_o_lig_idxs = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
                t_o_lig_masks = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
                template_aatype = batch.templates.aatype,
                template_atom_positions = batch.templates.atom_positions,
                template_atom_mask = batch.templates.atom_mask,
                # prev=embeddings,
                single=embeddings['single'],pair=embeddings['pair'],
                target_feat=target_feat
            )
        # print("dtype",c_pair.dtype,c_single.dtype)
        # print("shape",c_pair.shape,c_single.shape)
        # return embeddings
        output = {
            'single': single,
            'pair': pair,
            'target_feat': target_feat,
        }
        return output
