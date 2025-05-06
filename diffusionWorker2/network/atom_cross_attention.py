# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import dataclasses
from typing import Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusionWorker2.misc import feat_batch
from diffusionWorker2.network import atom_layout, utils
from diffusionWorker2.network.diffusion_transformer import DiffusionCrossAttTransformer
from torch.nn import LayerNorm
# from diffusionWorker2.network.layer_norm import LayerNorm

@dataclasses.dataclass(frozen=True)
class AtomCrossAttEncoderOutput:
    token_act: torch.Tensor  # (num_tokens, ch)
    skip_connection: torch.Tensor  # (num_subsets, num_queries, ch)
    queries_mask: torch.Tensor  # (num_subsets, num_queries)
    queries_single_cond: torch.Tensor  # (num_subsets, num_queries, ch)
    keys_mask: torch.Tensor  # (num_subsets, num_keys)
    keys_single_cond: torch.Tensor  # (num_subsets, num_keys, ch)
    pair_cond: torch.Tensor  # (num_subsets, num_queries, num_keys, ch)
    
# ref:
# class AtomCrossAttEncoderConfig(base_config.BaseConfig):
#   per_token_channels: int = 768
#   per_atom_channels: int = 128
#   atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
#       base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
#   )
#   per_atom_pair_channels: int = 16


class AtomCrossAttEncoder(nn.Module):
    def __init__(self,
                 per_token_channels: int = 384,
                 per_atom_channels: int = 128,
                 per_atom_pair_channels: int = 16,
                 with_token_atoms_act: bool = False,
                 with_trunk_single_cond: bool = False,
                 with_trunk_pair_cond: bool = False) -> None:
        super(AtomCrossAttEncoder, self).__init__()

        self.with_token_atoms_act = with_token_atoms_act
        self.with_trunk_single_cond = with_trunk_single_cond
        self.with_trunk_pair_cond = with_trunk_pair_cond

        self.c_positions = 3
        self.c_mask = 1
        self.c_element = 128
        self.c_charge = 1
        self.c_atom_name = 256
        self.c_pair_distance = 1
        self.per_token_channels = per_token_channels
        self.per_atom_channels = per_atom_channels
        self.per_atom_pair_channels = per_atom_pair_channels


        self.c_query = 128
        self.atom_transformer_encoder = DiffusionCrossAttTransformer(
            c_query=self.c_query)

        self.project_atom_features_for_aggr = nn.Linear(
            self.c_query, self.per_token_channels, bias=False)
        
        if self.with_trunk_single_cond is True:
            self.c_trunk_single_cond = 384
            # self.lnorm_trunk_single_cond = LayerNorm(
            #     self.c_trunk_single_cond, bias=False)
            # self.lnorm_trunk_single_cond=nn.LayerNorm(self.c_trunk_single_cond, bias=False)
            # self.embed_trunk_single_cond = nn.Linear(
            #     self.c_trunk_single_cond, self.per_atom_channels, bias=False)

        if self.with_token_atoms_act is True:
            self.atom_positions_to_features = nn.Linear(
                self.c_positions, self.per_atom_channels, bias=False)
            
        # if self.with_trunk_pair_cond is True:
        #     self.c_trunk_pair_cond = 128
        #     self.lnorm_trunk_pair_cond = LayerNorm(
        #         self.c_trunk_pair_cond, bias=False)
        #     self.embed_trunk_pair_cond = nn.Linear(
        #         self.c_trunk_pair_cond, self.per_atom_pair_channels, bias=False)

    def forward(
        self,
        queries_mask,

        pred_dense_atom_mask,
        # batch: feat_batch.Batch,
        acat_atoms_to_q_gather_idxs,
        acat_atoms_to_q_gather_mask,
        acat_q_to_k_gather_idxs,
        acat_q_to_k_gather_mask,
        # acat_t_to_q_gather_idxs,
        # acat_t_to_q_gather_mask,
        acat_q_to_atom_gather_idxs,
        acat_q_to_atom_gather_mask,
        # acat_t_to_k_gather_idxs,
        # acat_t_to_k_gather_mask,

        token_atoms_act,
        # trunk_single_cond,
        # trunk_pair_cond,
        queries_single_cond,
        pair_act, keys_mask, keys_single_cond
    ) :


        queries_act = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            # batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_act,
            layout_axes=(-3, -2),
        )

        queries_act = self.atom_positions_to_features(queries_act)
        queries_act *= queries_mask[..., None]
        queries_act += queries_single_cond

        # pair_act=self.pair_act.clone().contiguous()
        # keys_mask=self.keys_mask
        # keys_single_cond=self.keys_single_cond.clone().contiguous()

        queries_act = self.atom_transformer_encoder(
            queries_act=queries_act,
            queries_mask=queries_mask,
            # queries_to_keys=batch.atom_cross_att.queries_to_keys,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            keys_mask=keys_mask,
            queries_single_cond=queries_single_cond,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act
        )

        queries_act *= queries_mask[..., None]
        skip_connection = queries_act.clone()

        queries_act = self.project_atom_features_for_aggr(queries_act)

        token_atoms_act = atom_layout.convertV2(
            acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask,
            queries_act,
            layout_axes=(-3, -2),
        )

        token_act = utils.mask_mean(
            pred_dense_atom_mask[..., None], torch.relu(token_atoms_act), dim=-2
        )
        return token_act,skip_connection

        # return AtomCrossAttEncoderOutput(
        #     token_act=token_act,
        #     skip_connection=skip_connection,
        #     queries_mask=queries_mask,
        #     queries_single_cond=queries_single_cond,
        #     keys_mask=keys_mask,
        #     keys_single_cond=keys_single_cond,
        #     pair_cond=pair_act,
        # )


class AtomCrossAttDecoder(nn.Module):
    def __init__(self) -> None:
        super(AtomCrossAttDecoder, self).__init__()

        self.per_atom_channels = 128



        self.project_token_features_for_broadcast = nn.Linear(
            768, self.per_atom_channels, bias=False)

        self.atom_transformer_decoder = DiffusionCrossAttTransformer(
            c_query=self.per_atom_channels)

        self.atom_features_layer_norm = LayerNorm(
            self.per_atom_channels, bias=False)
        self.atom_features_to_position_update = nn.Linear(
            self.per_atom_channels, 3, bias=False)

    def forward(self,
                acat_atoms_to_q_gather_idxs,
                acat_atoms_to_q_gather_mask,

                acat_q_to_k_gather_idxs,
                acat_q_to_k_gather_mask,

                acat_q_to_atom_gather_idxs,
                acat_q_to_atom_gather_mask,

                # batch: feat_batch.Batch,
                token_act: torch.Tensor,  # (num_tokens, ch)

                skip_connection,queries_mask,keys_mask,queries_single_cond,
                keys_single_cond,pair_cond
                # enc: AtomCrossAttEncoderOutput
                ) -> torch.Tensor:
        # acat_q_to_k_gather_idxs = batch.atom_cross_att.queries_to_keys.gather_idxs
        # acat_q_to_k_gather_mask = batch.atom_cross_att.queries_to_keys.gather_mask
        token_act = self.project_token_features_for_broadcast(token_act)
        num_token, max_atoms_per_token = acat_q_to_atom_gather_idxs.shape
            # batch.atom_cross_att.queries_to_token_atoms.shape


        # print("num_token", num_token,"max_atoms_per_token", max_atoms_per_token)
        token_atom_act = torch.broadcast_to(
            token_act[:, None, :],
            (num_token, max_atoms_per_token, self.per_atom_channels),
        )

        queries_act = atom_layout.convertV2(
            # batch.atom_cross_att.token_atoms_to_queries,
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            token_atom_act,
            layout_axes=(-3, -2),
        )
        queries_act +=skip_connection
        queries_act *= queries_mask[..., None]

        # queries_act += enc.skip_connection
        # queries_act *= enc.queries_mask[..., None]

        # Run the atom cross attention transformer.
        queries_act = self.atom_transformer_decoder(
            queries_act=queries_act,
            queries_mask=queries_mask,
            # queries_to_keys=batch.atom_cross_att.queries_to_keys,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            keys_mask=keys_mask,
            queries_single_cond=queries_single_cond,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_cond,
        )

        queries_act *= queries_mask[..., None]
        queries_act = self.atom_features_layer_norm(queries_act)
        queries_position_update = self.atom_features_to_position_update(
            queries_act)
        # position_update = atom_layout.convert(
        #     batch.atom_cross_att.queries_to_token_atoms,
        #     queries_position_update,
        #     layout_axes=(-3, -2),
        # )
        position_update = atom_layout.convertV2(
            # batch.atom_cross_att.queries_to_token_atoms,
            acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask,
            queries_position_update,
            layout_axes=(-3, -2),
        )
        return position_update
