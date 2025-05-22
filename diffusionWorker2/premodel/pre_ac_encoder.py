import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusionWorker2.network import atom_layout, utils
from diffusionWorker2.network.diffusion_transformer import DiffusionCrossAttTransformer
from torch.nn import LayerNorm


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

        self.c_trunk_single_cond = 384
        self.lnorm_trunk_single_cond = nn.LayerNorm(self.c_trunk_single_cond, bias=False)
        self.embed_trunk_single_cond = nn.Linear(
            self.c_trunk_single_cond, self.per_atom_channels, bias=False)

        self.single_to_pair_cond_row_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False)

        self.single_to_pair_cond_col_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False)

        self.c_trunk_pair_cond = 128
        self.lnorm_trunk_pair_cond = LayerNorm(
            self.c_trunk_pair_cond, bias=False)
        self.embed_trunk_pair_cond = nn.Linear(
            self.c_trunk_pair_cond, self.per_atom_pair_channels, bias=False)

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

    def _per_atom_conditioning(self,
                               # batch: feat_batch.Batch,
                               ref_ops, ref_mask, ref_element,
                               ref_charge, ref_atom_name_chars
                               ):

        act = self.embed_ref_pos(ref_ops)


        act += self.embed_ref_mask(ref_mask[:, :, None].to(
            dtype=self.embed_ref_mask.weight.dtype))
        act += self.embed_ref_element(F.one_hot(ref_element.to(
            dtype=torch.int64), 128).to(dtype=self.embed_ref_element.weight.dtype))
        act += self.embed_ref_charge(torch.arcsinh(
            ref_charge)[:, :, None])

        # Characters are encoded as ASCII code minus 32, so we need 64 classes,
        # to encode all standard ASCII characters between 32 and 96.
        # atom_name_chars_1hot = F.one_hot(batch.ref_structure.atom_name_chars.to(
        #     dtype=torch.int64), 64).to(dtype=self.embed_ref_atom_name.weight.dtype)
        atom_name_chars_1hot = F.one_hot(ref_atom_name_chars.to(
            dtype=torch.int64), 64).to(dtype=self.embed_ref_atom_name.weight.dtype)
        num_token, num_dense, _ = act.shape
        act += self.embed_ref_atom_name(
            atom_name_chars_1hot.reshape(num_token, num_dense, -1))

        act *= ref_mask[:, :, None]
        return act

    def forward(
            self,
            trunk_single_cond,trunk_pair_cond,
            ref_ops, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,
            # batch: feat_batch.Batch,
            queries_mask,

            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,

            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,

            acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask,

            acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask,
    ) :
        token_atoms_single_cond = self._per_atom_conditioning(
            ref_ops=ref_ops,
            ref_mask=ref_mask,
            ref_element=ref_element,
            ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars,
        )
        queries_single_cond = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )


        trunk_single_cond = self.embed_trunk_single_cond(
            self.lnorm_trunk_single_cond(trunk_single_cond))
        queries_single_cond += atom_layout.convertV2(
            acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask,
            # batch.atom_cross_att.tokens_to_queries,
            trunk_single_cond,
            layout_axes=(-2,),
        )

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

        # time1=time.time()
        # if trunk_pair_cond is not None:

        trunk_pair_cond = self.embed_trunk_pair_cond(
            self.lnorm_trunk_pair_cond(trunk_pair_cond))
        # Create the GatherInfo into a flattened trunk_pair_cond from the
        # queries and keys gather infos.
        num_tokens = trunk_pair_cond.shape[0]

        pair_act_add = atom_layout.convertV2(
            (
                # num_tokens * tokens_to_queries.gather_idxs[:, :, None]
                    num_tokens * acat_t_to_q_gather_idxs[:, :, None]
                    + acat_t_to_k_gather_idxs[:, None, :]
            ),
            (
                # tokens_to_queries.gather_mask[:, :, None]
                # & tokens_to_keys.gather_mask[:, None, :]
                    acat_t_to_q_gather_mask[:, :, None]
                    & acat_t_to_k_gather_mask[:, None, :]
            ),
            trunk_pair_cond, layout_axes=(-3, -2)
        )
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

        pair_act_add += (self.embed_pair_offsets_1(offsets)
                         * offsets_valid[:, :, :, None])

        # Embed pairwise inverse squared distances
        sq_dists = torch.sum(torch.square(offsets), dim=-1)
        pair_act_add += self.embed_pair_distances_1(
            1.0 / (1 + sq_dists[:, :, :, None])) * offsets_valid[:, :, :, None]
        # Embed offsets valid mask
        pair_act_add += self.embed_pair_offsets_valid(offsets_valid[:, :, :, None].to(
            dtype=self.embed_pair_offsets_valid.weight.dtype))
        # print("pair_act_add", time.time()-time1)
        # self.pair_act_add=pair_act_add.to(dtype=pair_act.dtype).contiguous()

        pair_act += pair_act_add

        pair_act2 = self.pair_mlp_1(torch.relu(pair_act))
        pair_act2 = self.pair_mlp_2(torch.relu(pair_act2))
        pair_act += self.pair_mlp_3(torch.relu(pair_act2))

        return queries_single_cond,pair_act,keys_mask,keys_single_cond