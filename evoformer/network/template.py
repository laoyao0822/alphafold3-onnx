# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

from dataclasses import dataclass

import torch
import torch.nn as nn
import time
from evoformer.misc import protein_data_processing
from evoformer.misc import geometry,geometry_method
from alphafold3.constants import residue_names
from evoformer.network import pairformer
from torch.nn import LayerNorm
# @dataclass
# class DistogramFeaturesConfig:
#     # The left edge of the first bin.
#     min_bin: float = 3.25
#     # The left edge of the final bin. The final bin catches everything larger than
#     # `max_bin`.
#     max_bin: float = 50.75
#     # The number of bins in the distogram.
#     num_bins: int = 39
#
#
# min_bin: float = 3.25
# # The left edge of the final bin. The final bin catches everything larger than
#  # `max_bin`.
# max_bin: float = 50.75
# # The number of bins in the distogram.
# num_bins: int = 39


def make_backbone_rigid_tensor(
        positions: torch.Tensor,  # [num_res, num_atoms, 3]
        mask: torch.Tensor,  # [num_res, num_atoms]
        group_indices: torch.Tensor  # [num_res, num_group, 3]
    ):
    """Make backbone rigid transformation using pure tensor operations.

    Args:
        positions: (num_res, num_atoms, 3) atom positions
        mask: (num_res, num_atoms) atom mask
        group_indices: (num_res, num_group, 3) atom indices forming groups

    Returns:
        tuple: ( (rotation_tensor, translation_tensor), mask )
               rotation_tensor: [num_res, 9] flattened rotation matrix components
               translation_tensor: [num_res, 3]
               mask: [num_res]
    """
    # 1. 提取主链原子索引 [num_res, 3]
    backbone_indices = group_indices[:, 0]  # 取每组第一个group

    # 2. 获取三个关键原子索引 (C, CA, N -> 对应索引0,1,2)
    # 调整索引顺序为 (N, CA, C) -> 对应索引2,1,0
    c, b, a = [backbone_indices[..., i] for i in range(3)]
    c, b, a = [x.to(dtype=torch.int64).unsqueeze(1) for x in [c, b, a]]
    rigid_mask = (torch.gather(mask, 1, a).squeeze(1) * torch.gather(mask, 1, b).squeeze(1) * torch.gather(mask, 1, c).squeeze(1)).to(dtype=torch.float32)

    x,y,z=geometry.unstack(positions)

    x_0=torch.gather(x, 1, a).squeeze(1)
    y_0=torch.gather(y, 1, a).squeeze(1)
    z_0=torch.gather(z, 1, a).squeeze(1)
    x_1=torch.gather(x, 1, b).squeeze(1)
    y_1=torch.gather(y, 1, b).squeeze(1)
    z_1=torch.gather(z, 1, b).squeeze(1)
    x_2=torch.gather(x, 1, c).squeeze(1)
    y_2=torch.gather(y, 1, c).squeeze(1)
    z_2=torch.gather(z, 1, c).squeeze(1)
    rotation = geometry_method.rot3_from_two_vectors(
        x_2-x_1, y_2-y_1, z_2-z_1,
        x_0-x_1, y_0-y_1, z_0-z_1
    )

    return (rotation,x_1,y_1,z_1), rigid_mask


class TemplateEmbedding(nn.Module):
    """Embed a set of templates."""

    def __init__(self, pair_channel: int = 128, num_channels: int = 64):
        super(TemplateEmbedding, self).__init__()

        self.pair_channel = pair_channel
        self.num_channels = num_channels

        self.single_template_embedding = SingleTemplateEmbedding()

        self.output_linear = nn.Linear(
            self.num_channels, self.pair_channel, bias=False)

    def forward(
        self,
        query_embedding: torch.Tensor,
        # templates: features.Templates,
        template_aatype, template_atom_positions, template_atom_mask,

        padding_mask_2d: torch.Tensor,
        attn_mask:torch.Tensor,
        multichain_mask_2d: torch.Tensor
    ) -> torch.Tensor:
        # num_templates = templates.aatype.shape[0]

        num_templates = template_aatype.shape[0]
        num_res, _, _ = query_embedding.shape

        summed_template_embeddings = query_embedding.new_zeros(
            num_res, num_res, self.num_channels)

        for template_idx in range(num_templates):
            # template_embedding = self.single_template_embedding(
            #     query_embedding, templates[template_idx], padding_mask_2d, multichain_mask_2d
            # )
            template_embedding = self.single_template_embedding(
                query_embedding,
                template_aatype[template_idx], template_atom_positions[template_idx], template_atom_mask[template_idx],
                padding_mask_2d=padding_mask_2d,attn_mask=attn_mask, multichain_mask_2d=multichain_mask_2d
            )
            summed_template_embeddings += template_embedding

        embedding = summed_template_embeddings / (1e-7 + num_templates)

        embedding = torch.relu(embedding)

        embedding = self.output_linear(embedding)

        return embedding


class SingleTemplateEmbedding(nn.Module):
    """Embed a single template."""

    def __init__(self, num_channels: int = 64):
        super(SingleTemplateEmbedding, self).__init__()

        self.num_channels = num_channels
        self.template_stack_num_layer = 2

        # self.dgram_features_config = DistogramFeaturesConfig()

        self.query_embedding_norm = LayerNorm(128)

        self.template_pair_embedding_0 = nn.Linear(
            39, self.num_channels, bias=False)
        self.template_pair_embedding_1 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_2 = nn.Linear(
            31, self.num_channels, bias=False)
        self.template_pair_embedding_3 = nn.Linear(
            31, self.num_channels, bias=False)
        self.template_pair_embedding_4 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_5 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_6 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_7 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_8 = nn.Linear(
            128, self.num_channels, bias=False)


        # self.template_pair_embeddings = nn.ModuleList([
        #     self.template_pair_embedding_0,
        #     self.template_pair_embedding_1,
        #     self.template_pair_embedding_2,
        #     self.template_pair_embedding_3,
        #     self.template_pair_embedding_4,
        #     self.template_pair_embedding_5,
        #     self.template_pair_embedding_6,
        #     self.template_pair_embedding_7,
        #     self.template_pair_embedding_8
        # ])

        self.template_embedding_iteration = nn.ModuleList(
            [pairformer.PairformerBlock(c_pair=self.num_channels, num_intermediate_factor=2, with_single=False)
             for _ in range(self.template_stack_num_layer)]
        )

        self.output_layer_norm = LayerNorm(self.num_channels)

        self.min_bin=3.25
        # The left edge of the final bin. The final bin catches everything larger than
        # `max_bin`.
        self.max_bin=50.75
        # The number of bins in the distogram.
        self.num_bins =39

        self.RESTYPE_PSEUDOBETA_INDEX=nn.Parameter(protein_data_processing.RESTYPE_PSEUDOBETA_INDEX,requires_grad=False)
        self.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP=residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        self.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX=nn.Parameter(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX,requires_grad=False)

    def dgram_from_positions(self,positions: torch.Tensor):
        """Compute distogram from amino acid positions.

        Args:
          positions: (num_res, 3) Position coordinates.

        Returns:
          Distogram with the specified number of bins.
        """
        lower_breaks = torch.linspace(
            self.min_bin, self.max_bin, self.num_bins, device=positions.device)
        lower_breaks = torch.square(lower_breaks)
        upper_breaks_last = torch.ones(1, device=lower_breaks.device) * 1e8
        # print(lower_breaks.dtype,upper_breaks_last.dtype)
        upper_breaks = torch.concatenate(
            [lower_breaks[1:], upper_breaks_last], dim=-1
        )
        dist2 = torch.sum(
            input=torch.square(
                torch.unsqueeze(positions, dim=-2) - torch.unsqueeze(positions, dim=-3)),
            dim=-1,
            keepdim=True
        )

        dgram = (dist2 > lower_breaks).to(dtype=torch.float32) * (
                dist2 < upper_breaks
        ).to(dtype=torch.float32)
        return dgram

    def pseudo_beta_fn(
            self,
            aatype: torch.Tensor,
            dense_atom_positions: torch.Tensor,
            dense_atom_masks: torch.Tensor,
            is_ligand: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Create pseudo beta atom positions and optionally mask.
        Args:
          aatype: [num_res] amino acid types.
          dense_atom_positions: [num_res, NUM_DENSE, 3] vector of all atom positions.
          dense_atom_masks: [num_res, NUM_DENSE] mask.
          is_ligand: [num_res] flag if something is a ligand.
          use_jax: whether to use jax for the computations.

        Returns:
          Pseudo beta dense atom positions and the corresponding mask.
        """
        #   if use_jax:
        #     xnp = jnp
        #   else:
        #     xnp = np

        if is_ligand is None:
            is_ligand = torch.zeros_like(aatype)
        # torch.take_along_dim(input, indices, dim=None, *, out=None) → Tensor
        # print('self.RESTYPE_PSEUDOBETA_INDEX',self.RESTYPE_PSEUDOBETA_INDEX.shape)
        # print('aatype',aatype.shape)
        #self.RESTYPE_PSEUDOBETA_INDEX torch.Size([31])
        # aatype torch.Size([37])
        # pseudobeta_index_polymer = torch.take_along_dim(
        #     self.RESTYPE_PSEUDOBETA_INDEX, aatype.to(dtype=torch.int64),
        #     dim=0
        # ).to(dtype=torch.int32)
        # convert take_along_dim to gather because torch script only support up to opset20
        pseudobeta_index_polymer = self.RESTYPE_PSEUDOBETA_INDEX.to(
            device=aatype.device
        )[aatype.to(dtype=torch.int64)].to(dtype=torch.int32)  # 结果保持 [37]

        pseudobeta_index = torch.where(
            is_ligand.to(dtype=torch.bool),
            torch.zeros_like(pseudobeta_index_polymer),
            pseudobeta_index_polymer,
        ).to(dtype=torch.int64)

        print('dense_atom_positions',dense_atom_positions.shape)
        print('pseudobeta_index[..., None, None]',pseudobeta_index[..., None, None].shape)
        #dense_atom_positions torch.Size([37, 24, 3])
        # pseudobeta_index[..., None, None] torch.Size([37, 1, 1])
        pseudo_beta = torch.take_along_dim(
            dense_atom_positions, pseudobeta_index[..., None, None], dim=-2
        )
        pseudo_beta = torch.squeeze(pseudo_beta, dim=-2)
        # row_idx = torch.arange(37, device=dense_atom_positions.device)
        # pseudo_beta = dense_atom_positions[row_idx, pseudobeta_index, :]  # 直接索引 → [37, 3]


        # print('dense_atom_masks',dense_atom_masks.shape)
        # print('pseudobeta_index[..., None]',pseudobeta_index[..., None].shape)
        #dense_atom_masks torch.Size([37, 24])
        # pseudobeta_index[..., None] torch.Size([37, 1])
        pseudo_beta_mask = torch.take_along_dim(
            dense_atom_masks, pseudobeta_index[..., None], dim=-1
        ).to(dtype=torch.float32)
        pseudo_beta_mask = torch.squeeze(pseudo_beta_mask, dim=-1)
        # pseudo_beta_mask = dense_atom_masks[row_idx, pseudobeta_index].to(dtype=torch.float32)  # 直接索引 → [37]


        return pseudo_beta, pseudo_beta_mask

    def construct_input(
        self, query_embedding,
            # templates: features.Templates,
            templates_aatype, dense_atom_positions, dense_atom_mask,

            multichain_mask_2d
    ) -> torch.Tensor:
        # Compute distogram feature for the template.
        time1=time.time()
        dtype = query_embedding.dtype

        aatype = templates_aatype
        dense_atom_mask = dense_atom_mask

        # dense_atom_positions = templates.atom_positions
        dense_atom_positions *= dense_atom_mask[..., None]

        pseudo_beta_positions, pseudo_beta_mask = self.pseudo_beta_fn(
            templates_aatype, dense_atom_positions, dense_atom_mask
        )

        pseudo_beta_mask_2d = (
            pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        )

        # print(pseudo_beta_positions.shape, pseudo_beta_mask.shape,pseudo_beta_mask_2d.shape)
        pseudo_beta_mask_2d *= multichain_mask_2d
        dgram = self.dgram_from_positions(
            pseudo_beta_positions
        )
        dgram *= pseudo_beta_mask_2d[..., None]
        dgram = dgram.to(dtype=dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

        aatype = torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            self.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        ).to(dtype=dtype)
        to_concat.append((aatype[None, :, :], 1))
        to_concat.append((aatype[:, None, :], 1))

        # Compute a feature representing the normalized vector between each
        # backbone affine - i.e. in each residues local frame, what direction are
        # each of the other residues.
        # print("protein_data_processing:",protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(device=templates_aatype.device).shape)
        # print('self.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX',self.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.shape)
        # print('templates_aatype[..., None, None]',templates_aatype[..., None, None].shape)
        #self.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX torch.Size([31, 8, 3])
        #templates_aatype[..., None, None] torch.Size([37, 1, 1])
        # template_group_indices = torch.take_along_dim(
        #     self.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(device=templates_aatype.device),
        #     templates_aatype.to(dtype=torch.int64)[..., None, None],
        #     dim=0
        # )


        # convert take_along_dim to gather because torch script only support up to opset20

        template_group_indices = self.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(
            device=templates_aatype.device
        )[templates_aatype.to(dtype=torch.int64)]  # 结果自动广播到 [37, 8, 3]



        # rigid, backbone_mask = make_backbone_rigid(
        #     geometry.Vec3Array.from_array(dense_atom_positions),
        #     dense_atom_mask,
        #     template_group_indices.to(dtype=torch.int32),
        # )
        (rotation, x, y, z), backbone_mask = make_backbone_rigid_tensor(
            dense_atom_positions,  # 直接传入三维张量 [num_res, num_atoms, 3]
            dense_atom_mask,
            template_group_indices.to(dtype=torch.int32)
        )
        # points = rigid.translation
        #
        # rigid.rotation = geometry.Rot3Array(rigid.rotation.xx[:, None],
        #                                     rigid.rotation.xy[:, None],
        #                                     rigid.rotation.xz[:, None],
        #                                     rigid.rotation.yx[:, None],
        #                                     rigid.rotation.yy[:, None],
        #                                     rigid.rotation.yz[:, None],
        #                                     rigid.rotation.zx[:, None],
        #                                     rigid.rotation.zy[:, None],
        #                                     rigid.rotation.zz[:, None])
        #
        # rigid.translation = geometry.Vec3Array(rigid.translation.x[:, None],
        #                                        rigid.translation.y[:, None],
        #                                        rigid.translation.z[:, None])
        xx, xy, xz, yx, yy, yz, zx, zy, zz = rotation
        x_p, y_p, z_p = x, y, z
        xx = xx[:, None]
        xy = xy[:, None]
        xz = xz[:, None]
        yx = yx[:, None]
        yy = yy[:, None]
        yz = yz[:, None]
        zx = zx[:, None]
        zy = zy[:, None]
        zz = zz[:, None]
        x = x[:, None]
        y = y[:, None]
        z = z[:, None]

        rot_xx, rot_xy, rot_xz, rot_yx, rot_yy, rot_yz, rot_zx, rot_zy, rot_zz, t_x, t_y, t_z = geometry_method.rigid3_inverse(
            xx, xy, xz, yx, yy, yz, zx, zy, zz,
            x, y, z
        )
        # rot_xx, rot_xy, rot_xz,rot_yx, rot_yy, rot_yz,rot_zx, rot_zy, rot_zz=inv_rot
        x_vec, y_vec, z_vec = geometry_method.rigid3_apply_to_point(rot_xx, rot_xy, rot_xz,
                                                                    rot_yx, rot_yy, rot_yz,
                                                                    rot_zx, rot_zy, rot_zz,
                                                                    t_x, t_y, t_z, x_p, y_p, z_p)
        # unit_vector = rigid_vec.normalized()
        x_vec, y_vec, z_vec = geometry_method.vec3_normalized(x_vec, y_vec, z_vec)
        unit_vector = [x_vec, y_vec, z_vec]

        # rigid_vec = rigid.inverse().apply_to_point(points)
        # unit_vector = rigid_vec.normalized()
        # unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

        unit_vector = [x.to(dtype=dtype) for x in unit_vector]
        backbone_mask = backbone_mask.to(dtype=dtype)

        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        # Note that the backbone_mask takes into account C, CA and N (unlike
        # pseudo beta mask which just needs CB) so we add both masks as features.
        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))
        # for in_concat,_ in to_concat:
        #     print("in_concat",in_concat.shape)
        print("construct_input:",time.time()-time1)
        query_embedding = self.query_embedding_norm(query_embedding)
        # for x, n_input_dims in to_concat:
        #     print(x.shape)
        to_concat.append((query_embedding, 1))
        # act = 0
        # 处理第0个元素
        x, n_input_dims = to_concat[0]
        if n_input_dims == 0:
            x = x[..., None]
        act = self.template_pair_embedding_0(x)

        # 处理第1个元素
        x, n_input_dims = to_concat[1]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_1(x)

        # 处理第2个元素
        x, n_input_dims = to_concat[2]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_2(x)

        # 处理第3个元素
        x, n_input_dims = to_concat[3]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_3(x)

        # 处理第4个元素
        x, n_input_dims = to_concat[4]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_4(x)

        # 处理第5个元素
        x, n_input_dims = to_concat[5]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_5(x)

        # 处理第6个元素
        x, n_input_dims = to_concat[6]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_6(x)

        # 处理第7个元素
        x, n_input_dims = to_concat[7]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_7(x)

        # 处理第8个元素
        x, n_input_dims = to_concat[8]
        if n_input_dims == 0:
            x = x[..., None]
        act += self.template_pair_embedding_8(x)

        return act

        # for i, (x, n_input_dims) in enumerate(to_concat):
        #     if n_input_dims == 0:
        #         x = x[..., None]
        #     print("__getattr__",i)
            # act += self.__getattr__(f'template_pair_embedding_{i}')(x)
            # act += self.template_embedding_iteration[i](x)
            # act += self.template_pair_embedding_1(x)
            # act += self.template_pair_embedding_2(x)
            # act += self.template_pair_embedding_3(x)
            # act += self.template_pair_embedding_4(x)
            # act += self.template_pair_embedding_5(x)
            # act += self.template_pair_embedding_6(x)
            # act += self.template_pair_embedding_7(x)
            # act += self.template_pair_embedding_8(x)

    def forward(
        self,
        query_embedding: torch.Tensor,
        # templates: features.Templates,
        template_aatype, template_atom_positions, template_atom_mask,

        padding_mask_2d: torch.Tensor,
        attn_mask: torch.Tensor,
        multichain_mask_2d: torch.Tensor
    ) -> torch.Tensor:

        # act = self.construct_input(
        #     query_embedding, templates.aatype,templates.atom_positions,templates.atom_mask, multichain_mask_2d)

        act = self.construct_input(
            query_embedding,
            template_aatype, template_atom_positions, template_atom_mask,
            multichain_mask_2d)
        # padding_mask_2d_c=padding_mask_2d.clone()
        for pairformer_block in self.template_embedding_iteration:
            act = pairformer_block(act, pair_mask=padding_mask_2d,pair_mask_attn=attn_mask)
        # assert torch.allclose(padding_mask_2d_c, padding_mask_2d, atol=1e-2, rtol=1e-2), "输出不一致！"

        act = self.output_layer_norm(act)

        return act
