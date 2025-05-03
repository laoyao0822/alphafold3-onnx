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

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusionWorker.network import atom_layout, utils
from diffusionWorker.network.diffusion_transformer import DiffusionCrossAttTransformer

from diffusionWorker.network.layer_norm import LayerNorm

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

        self.embed_ref_pos = nn.Linear(
            self.c_positions, self.per_atom_channels, bias=False)

        self.embed_ref_mask = nn.Linear(
            self.c_mask, self.per_atom_channels, bias=False)

        self.embed_ref_element = nn.Linear(
            self.c_element, self.per_atom_channels, bias=False)
        self.embed_ref_charge = nn.Linear(
            self.c_charge, self.per_atom_channels, bias=False)

        self.embed_ref_atom_name = nn.Linear(
            self.c_atom_name, self.per_atom_channels, bias=False)

        self.single_to_pair_cond_row = nn.Linear(
            self.per_atom_channels, self.per_atom_pair_channels, bias=False)
        self.single_to_pair_cond_col = nn.Linear(
            self.per_atom_channels, self.per_atom_pair_channels, bias=False)

        self.embed_pair_offsets = nn.Linear(
            self.c_positions, self.per_atom_pair_channels, bias=False)
        self.embed_pair_distances = nn.Linear(
            self.c_pair_distance, self.per_atom_pair_channels, bias=False)

        self.single_to_pair_cond_row_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False)

        self.single_to_pair_cond_col_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False)

        self.embed_pair_offsets_1 = nn.Linear(
            self.c_positions, self.per_atom_pair_channels, bias=False)

        self.embed_pair_distances_1 = nn.Linear(
            1, self.per_atom_pair_channels, bias=False)

        self.embed_pair_offsets_valid = nn.Linear(
            1, self.per_atom_pair_channels, bias=False)

        self.pair_mlp_1 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)
        self.pair_mlp_2 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)
        self.pair_mlp_3 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)

        self.c_query = 128
        self.atom_transformer_encoder = DiffusionCrossAttTransformer(
            c_query=self.c_query)

        self.project_atom_features_for_aggr = nn.Linear(
            self.c_query, self.per_token_channels, bias=False)
        
        if self.with_trunk_single_cond is True:
            self.c_trunk_single_cond = 384
            self.lnorm_trunk_single_cond = LayerNorm(
                self.c_trunk_single_cond, bias=False)
            self.embed_trunk_single_cond = nn.Linear(
                self.c_trunk_single_cond, self.per_atom_channels, bias=False)

        if self.with_token_atoms_act is True:
            self.atom_positions_to_features = nn.Linear(
                self.c_positions, self.per_atom_channels, bias=False)
            
        if self.with_trunk_pair_cond is True:
            self.c_trunk_pair_cond = 128
            self.lnorm_trunk_pair_cond = LayerNorm(
                self.c_trunk_pair_cond, bias=False)
            self.embed_trunk_pair_cond = nn.Linear(
                self.c_trunk_pair_cond, self.per_atom_pair_channels, bias=False)

    def _per_atom_conditioning(self,ref_ops,ref_mask,ref_element,
                               ref_charge,ref_atom_name_chars) -> tuple[torch.Tensor, torch.Tensor]:
        

        # ref_ops=batch.ref_structure.positions
        # ref_mask=batch.ref_structure.mask
        # ref_element=batch.ref_structure.element
        # ref_charge=batch.ref_structure.charge
        # ref_atom_name_chars=batch.ref_structure.atom_name_chars
        # Compute per-atom single conditioning
        # Shape (num_tokens, num_dense, channels)
        act = self.embed_ref_pos(ref_ops)
        act += self.embed_ref_mask(ref_mask[:, :, None].to(
            dtype=self.embed_ref_mask.weight.dtype))

        # Element is encoded as atomic number if the periodic table, so
        # 128 should be fine.
        act += self.embed_ref_element(F.one_hot(ref_element.to(
            dtype=torch.int64), 128).to(dtype=self.embed_ref_element.weight.dtype))
        act += self.embed_ref_charge(torch.arcsinh(
            ref_charge)[:, :, None])



        # Characters are encoded as ASCII code minus 32, so we need 64 classes,
        # to encode all standard ASCII characters between 32 and 96.
        atom_name_chars_1hot = F.one_hot(ref_atom_name_chars.to(
            dtype=torch.int64), 64).to(dtype=self.embed_ref_atom_name.weight.dtype)
        num_token, num_dense, _ = act.shape
        act += self.embed_ref_atom_name(
            atom_name_chars_1hot.reshape(num_token, num_dense, -1))

        act *= ref_mask[:, :, None]

        # Compute pair conditioning
        # shape (num_tokens, num_dense, num_dense, channels)
        # Embed single features
        row_act = self.single_to_pair_cond_row(torch.relu(act))
        col_act = self.single_to_pair_cond_col(torch.relu(act))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        # Embed pairwise offsets
        pair_act += self.embed_pair_offsets(ref_ops[:, :, None, :]
                                            - ref_ops[:, None, :, :])

        sq_dists = torch.sum(
            torch.square(
                ref_ops[:, :, None, :]
                - ref_ops[:, None, :, :]
            ),
            dim=-1,
        )
        pair_act += self.embed_pair_distances(1.0 /
                                              (1 + sq_dists[:, :, :, None]))

        return act, pair_act

    def forward(
        self,
        ref_ops, ref_mask, ref_element,ref_charge, ref_atom_name_chars,ref_space_uid,
        pred_dense_atom_mask,
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
        token_atoms_act: Optional[torch.Tensor] = None,
        trunk_single_cond: Optional[torch.Tensor] = None,
        trunk_pair_cond: Optional[torch.Tensor] = None
    ) -> AtomCrossAttEncoderOutput:

        assert (token_atoms_act is not None) == self.with_token_atoms_act
        assert (trunk_single_cond is not None) == self.with_trunk_single_cond
        assert (trunk_pair_cond is not None) == self.with_trunk_pair_cond

        token_atoms_single_cond, _ = self._per_atom_conditioning(ref_ops,ref_mask,ref_element,
                               ref_charge,ref_atom_name_chars)
        # token_atoms_mask = pred_dense_atom_mask
        # acat_atoms_to_queries_input_shape = batch.atom_cross_att.token_atoms_to_queries.token_atoms_to_queries.input_shape
        # acat_atoms_to_queries_gather_idxs = batch.atom_cross_att.token_atoms_to_queries.token_atoms_to_queries.gather_idxs
        # acat_atoms_to_queries_gather_mask=batch.atom_cross_att.token_atoms_to_queries.token_atoms_to_queries.gather_mask
        # acat_q_to_k_gather_idxs = batch.atom_cross_att.queries_to_keys.gather_idxs
        # acat_q_to_k_gather_mask = batch.atom_cross_att.queries_to_keys.gather_mask
        # acat_q_to_k_input_shape = batch.atom_cross_att.queries_to_keys.input_shape
        # acat_t_to_q_gather_idxs = batch.atom_cross_att.tokens_to_queries.gather_idxs
        # acat_t_to_q_gather_mask = batch.atom_cross_att.tokens_to_queries.gather_mask
        # acat_t_to_q_input_shape = batch.atom_cross_att.tokens_to_queries.input_shape
        # acat_q_to_atom_gather_idxs = batch.atom_cross_att.queries_to_token_atoms.gather_idxs
        # acat_q_to_atom_gather_mask = batch.atom_cross_att.queries_to_token_atoms.gather_mask
        # acat_q_to_atom_input_shape = batch.atom_cross_att.queries_to_token_atoms.input_shape
        queries_single_cond = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )

        queries_mask = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            pred_dense_atom_mask,
            layout_axes=(-2, -1),
        )

        # If provided, broadcast single conditioning from trunk to all queries
        if trunk_single_cond is not None:
            trunk_single_cond = self.embed_trunk_single_cond(
                self.lnorm_trunk_single_cond(trunk_single_cond))
            queries_single_cond += atom_layout.convertV2(
                # batch.atom_cross_att.tokens_to_queries,
                acat_t_to_q_gather_idxs,
                acat_t_to_q_gather_mask,
                trunk_single_cond,
                layout_axes=(-2,),
            )

        if token_atoms_act is None:
            queries_act = queries_single_cond.clone()
        else:
            # Convert token_atoms_act to queries layout and map to per_atom_channels
            # (num_subsets, num_queries, channels)
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

        keys_single_cond = atom_layout.convertV2(
            # batch.atom_cross_att.queries_to_keys,
            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,
            queries_single_cond,
            layout_axes=(-3, -2),
        )
        keys_mask = atom_layout.convertV2(
            # batch.atom_cross_att.queries_to_keys,
            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,
            queries_mask, layout_axes=(
                -2, -1)
        )

        # Embed single features into the pair conditioning.
        # shape (num_subsets, num_queries, num_keys, ch)
        row_act = self.single_to_pair_cond_row_1(
            torch.relu(queries_single_cond))
        pair_cond_keys_input = atom_layout.convertV2(
            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,
            queries_single_cond,
            layout_axes=(-3, -2),
        )

        col_act = self.single_to_pair_cond_col_1(
            torch.relu(pair_cond_keys_input))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        if trunk_pair_cond is not None:
            trunk_pair_cond = self.embed_trunk_pair_cond(
                self.lnorm_trunk_pair_cond(trunk_pair_cond))
            
            # Create the GatherInfo into a flattened trunk_pair_cond from the
            # queries and keys gather infos.
            num_tokens = trunk_pair_cond.shape[0]

            pair_act += atom_layout.convertV2(
                num_tokens * acat_t_to_q_gather_idxs[:, :, None]
                + acat_t_to_k_gather_idxs[:, None, :],
                (
                        acat_t_to_q_gather_mask[:, :, None]
                        & acat_t_to_k_gather_mask[:, None, :]
                ), trunk_pair_cond, layout_axes=(-3, -2)
            )

        # Embed pairwise offsets
        queries_ref_pos = atom_layout.convertV2(
            # batch.atom_cross_att.token_atoms_to_queries,
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            ref_ops,
            layout_axes=(-3, -2),
        )
        queries_ref_space_uid = atom_layout.convertV2(
            # batch.atom_cross_att.token_atoms_to_queries,
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            ref_space_uid,
            layout_axes=(-2, -1),
        )
        keys_ref_pos = atom_layout.convertV2(
            # batch.atom_cross_att.queries_to_keys,
            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,
            queries_ref_pos,
            layout_axes=(-3, -2),
        )
        keys_ref_space_uid = atom_layout.convertV2(
            # batch.atom_cross_att.queries_to_keys,
            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,
            ref_space_uid,
            layout_axes=(-2, -1),
        )

        offsets_valid = (
            queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
        )
        offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]

        pair_act += (self.embed_pair_offsets_1(offsets)
                     * offsets_valid[:, :, :, None])

        # Embed pairwise inverse squared distances
        sq_dists = torch.sum(torch.square(offsets), dim=-1)
        pair_act += self.embed_pair_distances_1(
            1.0 / (1 + sq_dists[:, :, :, None])) * offsets_valid[:, :, :, None]
        # Embed offsets valid mask
        pair_act += self.embed_pair_offsets_valid(offsets_valid[:, :, :, None].to(
            dtype=self.embed_pair_offsets_valid.weight.dtype))

        # Run a small MLP on the pair acitvations
        pair_act2 = self.pair_mlp_1(torch.relu(pair_act))
        pair_act2 = self.pair_mlp_2(torch.relu(pair_act2))
        pair_act += self.pair_mlp_3(torch.relu(pair_act2))

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

        return AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=skip_connection,
            queries_mask=queries_mask,
            queries_single_cond=queries_single_cond,
            keys_mask=keys_mask,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )


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
                # batch: feat_batch.Batch,
                acat_atoms_to_q_gather_idxs,
                acat_atoms_to_q_gather_mask,
                acat_q_to_k_gather_idxs,
                acat_q_to_k_gather_mask,
                acat_q_to_atom_gather_idxs,
                acat_q_to_atom_gather_mask,
                token_act: torch.Tensor,  # (num_tokens, ch)
                enc: AtomCrossAttEncoderOutput) -> torch.Tensor:
        token_act = self.project_token_features_for_broadcast(token_act)
        num_token, max_atoms_per_token = (
            # batch.atom_cross_att.queries_to_token_atoms.shape
            acat_q_to_atom_gather_idxs.shape
        )
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
        queries_act += enc.skip_connection
        queries_act *= enc.queries_mask[..., None]

        # Run the atom cross attention transformer.
        queries_act = self.atom_transformer_decoder(
            queries_act=queries_act,
            queries_mask=enc.queries_mask,
            # queries_to_keys=batch.atom_cross_att.queries_to_keys,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            keys_mask=enc.keys_mask,
            queries_single_cond=enc.queries_single_cond,
            keys_single_cond=enc.keys_single_cond,
            pair_cond=enc.pair_cond,
        )

        queries_act *= enc.queries_mask[..., None]
        queries_act = self.atom_features_layer_norm(queries_act)
        queries_position_update = self.atom_features_to_position_update(
            queries_act)
        position_update = atom_layout.convertV2(
            # batch.atom_cross_att.queries_to_token_atoms,
            acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask,
            queries_position_update,
            layout_axes=(-3, -2),
        )
        return position_update
