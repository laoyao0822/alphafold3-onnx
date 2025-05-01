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

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist

from evoformer.misc import feat_batch, features
from evoformer.network import featurization
from evoformer.network.pairformer import EvoformerBlock, PairformerBlock
from evoformer.network.template import TemplateEmbedding
# from evoformer.network import atom_cross_attention
from alphafold3.constants import residue_names
from evoformer.misc import protein_data_processing
from evoformer.network.dot_product_attention import get_attn_mask

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_helper import _get_tensor_dim_size, parse_args
@parse_args("v", "v", "v", "v", "f", "b", "f", "b")
def symbolic_scaled_dot_product_attention(
    g, query, key, value, attn_mask, scale, is_causal, dropout_p, return_debug_mask
):
    batch_dim = _get_tensor_dim_size(query, 0)  # "N"
    head_dim = _get_tensor_dim_size(query, 1)  # 固定头数（若需动态则用符号）
    seq_len_dim = _get_tensor_dim_size(query, 2)  # "L"
    embed_dim = _get_tensor_dim_size(query, 3)  # 固定嵌入维度
    output_shape = [batch_dim, head_dim, seq_len_dim, embed_dim]
    # 创建输出节点并显式设置形状
    output = g.op(
        "openvino::scaled_dot_product_attention",
        query, key, value, attn_mask,
        scale_f=scale,
        causal_i=int(is_causal)
    ).setType(query.type().with_sizes(output_shape))
    # 忽略 dropout_p 和 return_debug_mask（推理时不需）
    return output
from onnxscript import opset22 as op
#适用于dynamo分解sdpa
def custom_scaled_dot_product_attention(
        query:torch.Tensor,
        key:torch.Tensor,
        value:torch.Tensor,
        attn_mask:torch.Tensor
):
    # 分解逻辑（模拟 SDPA 的计算步骤）
    # scale_factor = 1.0 / (query.size(-1) ** 0.5)
    k_transposed = op.Transpose(key, perm=[0, 2, 1, 3])
    # Step 1: Q*K^T 并缩放
    attn_scores = op.MatMul( query, k_transposed)
    # scaled_scores = g.op("Div", attn_scores, g.op("Constant", value_t=torch.tensor(scale_factor)))
    attn_scores=op.Relu(attn_scores)
    # Step 2: 应用掩码（假设 attn_mask 是加性掩码）
    # if not symbolic_helper._is_none(attn_mask):
    scaled_scores =op.Add( attn_scores, attn_mask)

    # Step 3: Softmax
    attn_weights = op.Softmax(scaled_scores, axis=-1)

    # Step 4: 聚合 Value
    output =op.MatMul( attn_weights, value)
    return output


from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args


@parse_args("v", "v", "i")
def symbolic_take_along_dim(g, input, indices, dim):
    # 添加维度对齐（若 indices 维度少于 input）
    input_rank = g.op("Size", g.op("Shape", input))  # Tensor[INT64]
    indices_rank = g.op("Size", g.op("Shape", indices))  # Tensor[INT64]

    # 计算需要扩展的维度数（动态计算）
    delta_rank = g.op("Sub", input_rank, indices_rank)  # Tensor[INT64]

    # 生成扩展轴列表 [-1, -2, ... ]（动态生成）
    axes = g.op(
        "Range",
        g.op("Constant", value_t=torch.tensor(-1, dtype=torch.int64)),
        g.op("Sub",
             g.op("Constant", value_t=torch.tensor(-1, dtype=torch.int64)),
             delta_rank),
        g.op("Constant", value_t=torch.tensor(-1, dtype=torch.int64))
    )

    # 动态扩展维度（一次性完成）
    expanded_indices = g.op("Unsqueeze", indices, axes)

    # 核心操作映射到 GatherElements
    return g.op("GatherElements", input, expanded_indices, axis_i=dim)

    # 核心操作映射到 GatherElements


# 注册自定义符号函数



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
        self.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP=residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
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


    def _embed_template_pair(
        self,
        # batch: feat_batch.Batch,
        asym_id,
        template_aatype, template_atom_positions, template_atom_mask,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_mask: torch.Tensor,
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
            padding_mask_2d=pair_mask,
            attn_mask=attn_mask,

        )

        return pair_activations + template_act

    def create_msa_feat(self,rows, deletion_matrix) -> torch.Tensor:
        """Create and concatenate MSA features."""
        deletion_matrix = deletion_matrix
        has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
        deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / torch.pi))[
            ..., None
        ]
        msa_1hot = torch.nn.functional.one_hot(
            rows.to(
                dtype=torch.int64), self.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1
        ).to(dtype=has_deletion.dtype)
        msa_feat = [
            msa_1hot,
            has_deletion,
            deletion_value,
        ]

        return torch.concatenate(msa_feat, dim=-1)

    def _embed_process_msa(
        self,
        # msa_batch: features.MSA,
        rows, mask, deletion_matrix,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
        pair_mask_attn: torch.Tensor,
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
        # time1 = time.time()
        msa_feat = self.create_msa_feat(rows,deletion_matrix).to(dtype=dtype)
        # print("created msa feat:", time.time() - time1)
        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat)[None]


        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                pair_mask_attn=pair_mask_attn,
            )

        return pair_activations

    def forward(
        self,
        single, pair,
        target_feat,
        msa, msa_mask, deletion_matrix,

        token_index, residue_index, asym_id, entity_id, sym_id,

        seq_mask,
        contact_matrix,
        template_aatype, template_atom_positions, template_atom_mask,

        # pair_activations,pair_mask,
        # attn_mask_4,
        attn_mask_seq
    ) :
        # target_feat_c=target_feat.clone()

        # pair=prev['pair']
        # single=prev['single']
        num_tokens = token_index.shape[0]

        pair_activations, pair_mask = self._seq_pair_embedding(
            seq_mask, target_feat
        )
        attn_mask_4 = get_attn_mask(mask=pair_mask, dtype=pair_activations.dtype, device=pair_activations.device,
                                    batch_size=num_tokens,
                                    num_heads=4, seq_len=num_tokens)


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
        # pair_activations = self._embed_bonds(
        #     gather_idxs_polymer_ligand=t_o_pol_idx, tokens_to_polymer_ligand_bonds_gather_mask=t_o_pol_mask,
        #     gather_idxs_ligand_ligand=t_o_lig_idxs, tokens_to_ligand_ligand_bonds_gather_mask=t_o_lig_masks, num_tokens=num_tokens,
        #     pair_activations=pair_activations
        # )
        pair_activations += self.bond_embedding(contact_matrix)

        single_activations = self.single_activations(target_feat)
        # pair_mask_c=pair_mask.clone()

        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(single))

        # attn_mask_seq = get_attn_mask(mask=seq_mask, dtype=single_activations.dtype, device=pair_activations.device,num_heads=16,seq_len=num_tokens,batch_size=1)

        #
        pair_activations = self._embed_template_pair(
            asym_id=asym_id, template_aatype=template_aatype,
            template_atom_positions=template_atom_positions, template_atom_mask=template_atom_mask,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            attn_mask=attn_mask_4
            # pair_mask=pair_mask,

        )
        # assert torch.allclose(pair_mask_c, pair_mask, atol=1e-2, rtol=1e-2), "输出不一致！"

        # print("_embed_template_pair over")
        pair_activations = self._embed_process_msa(
            # msa_batch=batch.msa,
            msa, msa_mask, deletion_matrix,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
            pair_mask_attn=attn_mask_4,
        )
        # assert torch.allclose(pair_mask_c, pair_mask, atol=1e-2, rtol=1e-2), "输出不一致！"

        # print("_embed_process_msa over")
        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = pairformer_b(
                pair_activations, pair_mask, single_activations, attn_mask_seq,attn_mask_4)
        # assert torch.allclose(pair_mask_c, pair_mask, atol=1e-2, rtol=1e-2), "输出不一致！"

        # if torch.allclose(target_feat,target_feat_c,1e-5):
        #     print("target feat  not change")
        # print("_pairformer_b over")
        # exit(0)

        # output = {
        #     'single': single_activations,
        #     'pair': pair_activations,
        #     'target_feat': target_feat,
        # }
        # return output
        return single_activations,pair_activations

class EvoFormerOne():

    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(EvoFormerOne, self).__init__()

        self.num_recycles = 1
        self.num_samples = num_samples

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer = Evoformer()


    def getOnnxModel(self, batch, target_feat, save_path, ):
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        pair = torch.zeros(
            [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
            dtype=torch.float32,
        )
        single = torch.zeros(
            [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
        )
        contact_matrix = self.get_contact_matrix(
            gather_idxs_polymer_ligand=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
            tokens_to_polymer_ligand_bonds_gather_mask=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
            gather_idxs_ligand_ligand=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
            tokens_to_ligand_ligand_bonds_gather_mask=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
            num_tokens=num_res,
        )
        seq_mask=batch.token_features.mask
        attn_mask_seq = get_attn_mask(mask=seq_mask, dtype=pair.dtype, device=pair.device,num_heads=16,seq_len=num_res,batch_size=1)
        print(attn_mask_seq.shape)

        # from onnxscript.rewriter import rewrite


        ten_length = torch.export.Dim('ten_length', min=100, max=16000)

        # ten_length = 10 * seq_len

        edge_number = torch.export.Dim('edge_number', min=10, max=1500)
        output_names = ["single_out", "pair_out"]
        ordered_keys=['single', 'pair', 'target_feat',
                        'msa', 'msa_mask', 'deletion_matrix',
                        'token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id',
                        'seq_mask',
                        'contact_matrix',
                        # 't_o_pol_idx', 't_o_pol_mask', 't_o_lig_idxs', 't_o_lig_masks',
                        'template_aatype', 'template_atom_positions', 'template_atom_mask',
                        'attn_mask_seq'
                        ]
        print('attn_seq_mask', attn_mask_seq.shape)
        kwarg_inputs = {
            'single': single,
            'pair': pair,
            'target_feat': target_feat,
            'msa': batch.msa.rows,
            'msa_mask': batch.msa.mask,
            'deletion_matrix': batch.msa.deletion_matrix,
            'token_index': batch.token_features.token_index,
            'residue_index': batch.token_features.residue_index,
            'asym_id': batch.token_features.asym_id,
            'entity_id': batch.token_features.entity_id,
            'sym_id': batch.token_features.sym_id,

            'seq_mask': batch.token_features.mask,
            'contact_matrix': contact_matrix,
            # 't_o_pol_idx': batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
            # 't_o_pol_mask': batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
            # 't_o_lig_idxs': batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
            # 't_o_lig_masks': batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
            'template_aatype': batch.templates.aatype,
            'template_atom_positions': batch.templates.atom_positions,
            'template_atom_mask': batch.templates.atom_mask,

            'attn_mask_seq':attn_mask_seq,
        }
        ordered_inputs = tuple(kwarg_inputs[key] for key in ordered_keys)

        opset_version=22
        print("opset: ", opset_version)
        print("start to save")

        # register_custom_op_symbolic(
        #     "aten::scaled_dot_product_attention",
        #     scaled_dot_product_attention_ov_symbolic,
        #     opset_version=13  # 与 OpenVINO opset 对齐
        # )

        # register_custom_op_symbolic(
        #     'aten::scaled_dot_product_attention',
        #     symbolic_scaled_dot_product_attention,
        #     opset_version=15
        # )
        # register_custom_op_symbolic(
        #     'aten::take_along_dim',
        #     symbolic_take_along_dim,
        #     opset_version=20
        # )
        seq_len = torch.export.Dim('seq_len', min=10, max=1600)
        torch.onnx.export(self.evoformer,
                      ordered_inputs, f=save_path, dynamo=True,
                      input_names=ordered_keys,
                      output_names=output_names,

                      optimize=False,
                      opset_version=opset_version,
                      export_params=True,
                      # dynamic_axes={
                      #         # 单序列输入
                      #         "single": {0: "seq_len"},
                      #
                      #         # 配对输入
                      #         "pair": {0: "seq_len", 1: "seq_len"},
                      #
                      #         # 特征类输入
                      #         "target_feat": {0: "seq_len"},
                      #
                      #         # MSA 相关输入
                      #         "msa": {1: "seq_len"},
                      #         "msa_mask": {1: "seq_len"},
                      #         "deletion_matrix": {1: "seq_len"},
                      #
                      #         # 索引类输入
                      #         "token_index": {0: "seq_len"},
                      #         "residue_index": {0: "seq_len"},
                      #         "asym_id": {0: "seq_len"},
                      #         "entity_id": {0: "seq_len"},
                      #         "sym_id": {0: "seq_len"},
                      #
                      #         # 掩码与矩阵
                      #         "seq_mask": {0: "seq_len"},
                      #         "contact_matrix": {0: "seq_len", 1: "seq_len"},
                      #
                      #         # 模板相关
                      #         "template_aatype": {1: "seq_len"},
                      #         "template_atom_positions": {1: "seq_len"},
                      #         "template_atom_mask": {1: "seq_len"},
                      #         # 注意力掩码
                      #         "attn_mask_seq": {2:"seq_len",3: "seq_len"},
                      #
                      #         "single_out": {0: "seq_len"},
                      #
                      #         # 配对输入
                      #         "pair_out": {0: "seq_len", 1: "seq_len"},
                      #     },
                      # custom_translation_table={
                      #         # torch.ops.mylibrary.sdpa_vino.default:custom_scaled_dot_product_attention
                      #         torch.ops.aten.scaled_dot_product_attention.default: custom_scaled_dot_product_attention,
                      # },
                      dynamic_shapes={
                          'single': {0: seq_len},
                          'pair': {0: seq_len, 1: seq_len},
                          'target_feat': {0: seq_len},

                          'msa': {1: seq_len},
                          'msa_mask': {1: seq_len},
                          'deletion_matrix': {1: seq_len},

                          'token_index': {0: seq_len},
                          'residue_index': {0: seq_len},
                          'asym_id': {0: seq_len},
                          'entity_id': {0: seq_len},
                          'sym_id': {0: seq_len},

                          'seq_mask': {0: seq_len},
                          'contact_matrix':{0: seq_len,1: seq_len},
                          # 模板相关
                          'template_aatype': {1: seq_len},
                          'template_atom_positions': {1: seq_len},
                          'template_atom_mask': {1: seq_len},
                          'attn_mask_seq':{2:seq_len,3:seq_len}
                      },
                    # custom_opsets = {"openvino.opset15": 15}
                          )
        # export_program=torch.export.export(self.evoformer,
        #                   ordered_inputs,
        #                    strict=False,
        #                   # dynamic_axes={
        #                   #         # 单序列输入
        #                   #         "single": {0: "seq_len"},
        #                   #
        #                   #         # 配对输入
        #                   #         "pair": {0: "seq_len", 1: "seq_len"},
        #                   #
        #                   #         # 特征类输入
        #                   #         "target_feat": {0: "seq_len"},
        #                   #
        #                   #         # MSA 相关输入
        #                   #         "msa": {1: "seq_len"},
        #                   #         "msa_mask": {1: "seq_len"},
        #                   #         "deletion_matrix": {1: "seq_len"},
        #                   #
        #                   #         # 索引类输入
        #                   #         "token_index": {0: "seq_len"},
        #                   #         "residue_index": {0: "seq_len"},
        #                   #         "asym_id": {0: "seq_len"},
        #                   #         "entity_id": {0: "seq_len"},
        #                   #         "sym_id": {0: "seq_len"},
        #                   #
        #                   #         # 掩码与矩阵
        #                   #         "seq_mask": {0: "seq_len"},
        #                   #         "contact_matrix": {0: "seq_len", 1: "seq_len"},
        #                   #
        #                   #         # 模板相关
        #                   #         "template_aatype": {1: "seq_len"},
        #                   #         "template_atom_positions": {1: "seq_len"},
        #                   #         "template_atom_mask": {1: "seq_len"},
        #                   #         # 注意力掩码
        #                   #         "attn_mask_seq": {3: "seq_len"},
        #                   #
        #                   #         "single_out": {0: "seq_len"},
        #                   #
        #                   #         # 配对输入
        #                   #         "pair_out": {0: "seq_len", 1: "seq_len"},
        #                   #     },
        #                   dynamic_shapes={
        #                       'single': {0: seq_len},
        #                       'pair': {0: seq_len, 1: seq_len},
        #                       'target_feat': {0: seq_len},
        #
        #                       'msa': {1: seq_len},
        #                       'msa_mask': {1: seq_len},
        #                       'deletion_matrix': {1: seq_len},
        #
        #                       'token_index': {0: seq_len},
        #                       'residue_index': {0: seq_len},
        #                       'asym_id': {0: seq_len},
        #                       'entity_id': {0: seq_len},
        #                       'sym_id': {0: seq_len},
        #
        #                       'seq_mask': {0: seq_len},
        #                       'contact_matrix': {0: seq_len, 1: seq_len},
        #                       # 模板相关
        #                       'template_aatype': {1: seq_len},
        #                       'template_atom_positions': {1: seq_len},
        #                       'template_atom_mask': {1: seq_len},
        #                       'attn_mask_seq': {2:seq_len ,3: seq_len}
        #                   },
        #                   # custom_opsets = {"openvino.opset15": 15}
        #                   )
        # print("export program done")
        # ov_model=ov.convert_model(export_program,verbose=True)
        # ov.save_model(ov_model,'/root/model.xml')

        exit(0)


    def get_contact_matrix(
        self,
            gather_idxs_polymer_ligand,
            tokens_to_polymer_ligand_bonds_gather_mask,
            gather_idxs_ligand_ligand,
            tokens_to_ligand_ligand_bonds_gather_mask,
            num_tokens,
            # pair_activations: torch.Tensor
    ) -> torch.Tensor:
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=torch.float32)

        gather_mask_polymer_ligand = tokens_to_polymer_ligand_bonds_gather_mask.prod(dim=1)[:, None]*gather_idxs_polymer_ligand

        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds_gather_mask.prod(dim=1)[:, None]*gather_idxs_ligand_ligand

        # gather_mask_polymer_ligand=gather_mask_polymer_ligand.to(dtype=gather_idxs_polymer_ligand.dtype)
        # gather_mask_ligand_ligand=gather_mask_ligand_ligand.to(dtype=gather_idxs_ligand_ligand.dtype)
        print(gather_mask_polymer_ligand.dtype, gather_mask_ligand_ligand.dtype)
        gather_idxs = torch.concatenate(
            [ gather_mask_polymer_ligand.to(dtype=torch.int64),
            gather_mask_ligand_ligand.to(dtype=torch.int64)]
        )
        # print("gather_idxs",gather_idxs.shape,"contact_matrix",contact_matrix.shape)
        # print(gather_idxs[:, 0].shape,gather_idxs[:, 1].shape)
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0
        # print("after contact_matrix",contact_matrix.shape)
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = torch.tensor(
            1.0,
            dtype=contact_matrix.dtype,
            device=contact_matrix.device
        ).expand(gather_idxs.shape[0])
        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0
        # print("bond_embedding:",self.bond_embedding(contact_matrix[:, :, None]).dtype)
        # bonds_act = self.bond_embedding(contact_matrix[:, :, None])
        # print(pair_activations.dtype)
        return contact_matrix[:, :, None]


    def forward(self, batch: dict[str, torch.Tensor],target_feat) -> dict[str, torch.Tensor]:
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        #target_feat torch.Size([37, 447])
        
        # target_feat1=self.create_target_feat_embedding(batch)
        pair= torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=torch.float32,
            )
        single=torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
            )

        # template_aatype = batch.templates.aatype
        # template_atom_positions = batch.templates.atom_positions
        # template_atom_mask = batch.templates.atom_mask
        # num_templates = template_aatype.shape[0]
        #
        # # print(num_templates)
        #
        # pseudo_beta_positions=torch.zeros(size=(num_templates,num_res,3))
        # pseudo_beta_mask=torch.zeros(size=(num_templates,num_res,3))

        # for template_idx in range(num_templates):
        #     pb_position, pb_mask = scoring.pseudo_beta_fn(
        #         template_aatype[template_idx], template_atom_positions[template_idx], template_atom_mask[template_idx]
        #     )
        #     pseudo_beta_positions[template_idx] = pb_position
        #     pseudo_beta_mask[template_idx] = pb_mask

        contact_matrix=self.get_contact_matrix(
            gather_idxs_polymer_ligand=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
            tokens_to_polymer_ligand_bonds_gather_mask = batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
            gather_idxs_ligand_ligand = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
            tokens_to_ligand_ligand_bonds_gather_mask = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
            num_tokens=num_res,
        )
        # pt_path='/root/pycharm/calibration/1024.pt'
        # torch.save({
        #     'single': single,
        #     'pair': pair,
        #     'target_feat': target_feat,
        #     'msa': batch.msa.rows,
        #     'msa_mask': batch.msa.mask,
        #     'deletion_matrix': batch.msa.deletion_matrix,
        #     'token_index': batch.token_features.token_index,
        #     'residue_index': batch.token_features.residue_index,
        #     'asym_id': batch.token_features.asym_id,
        #     'entity_id': batch.token_features.entity_id,
        #     'sym_id': batch.token_features.sym_id,
        #
        #     'seq_mask': batch.token_features.mask,
        #     'contact_matrix': contact_matrix,
        #     'template_aatype': batch.templates.aatype,
        #     'template_atom_positions': batch.templates.atom_positions,
        #     'template_atom_mask': batch.templates.atom_mask,
        # }, pt_path)
        # print("success save calibration pt:",pt_path)
        # exit(0)
        # print(contact_matrix.shape)

        seq_mask = batch.token_features.mask
        num_tokens = seq_mask.shape[0]

        # pair_activations, pair_mask = self.evoformer._seq_pair_embedding(
        #     seq_mask, target_feat
        # )
        #
        # attn_mask_4 = get_attn_mask(mask=pair_mask, dtype=pair_activations.dtype, device=pair_activations.device,
        #                             batch_size=num_tokens,
        #                             num_heads=4, seq_len=num_tokens)
        attn_mask_seq = get_attn_mask(mask=seq_mask, dtype=torch.float32, device='cpu',num_heads=16,seq_len=num_tokens,batch_size=1)



        # time1=time.time()
        for _ in range(self.num_recycles + 1):
        # for _ in range(0+1):
            time1=time.time()
            single,pair = self.evoformer(
                single=single, pair=pair,
                # single=embeddings['single'],pair=embeddings['pair'],
                target_feat=target_feat,
                # batch=batch,
                msa=batch.msa.rows,
                msa_mask = batch.msa.mask,
                deletion_matrix = batch.msa.deletion_matrix,

                token_index=batch.token_features.token_index,
                residue_index = batch.token_features.residue_index,
                asym_id = batch.token_features.asym_id,
                entity_id = batch.token_features.entity_id,
                sym_id = batch.token_features.sym_id,

                seq_mask=seq_mask,
                contact_matrix=contact_matrix,
                # t_o_pol_idx=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
                # t_o_pol_mask = batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
                # t_o_lig_idxs = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
                # t_o_lig_masks = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
                template_aatype = batch.templates.aatype,
                template_atom_positions = batch.templates.atom_positions,
                template_atom_mask = batch.templates.atom_mask,

                # pair_activations = pair_activations,
                # pair_mask=pair_mask,
                # attn_mask_4=attn_mask_4,
                attn_mask_seq=attn_mask_seq,
                # prev=embeddings,
            )
            print("evo one cost time:", time.time()-time1)
            # exit(0)
        # print("dtype",c_pair.dtype,c_single.dtype)
        # print("shape",c_pair.shape,c_single.shape)
        # return embeddings
        output = {
            'single': single,
            'pair': pair,
            'target_feat': target_feat,
        }
        return output
import openvino as ov
import openvino.properties as properties
class EvoformerVino():

    def __init__(self):
        super(EvoformerVino, self).__init__()
        self.num_recycles = 10
        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

    def initOpenvinoModel(self,openvino_path):
        self.core = ov.Core()
        # self.core.set_property(
        # "CPU",
        #     {   properties.hint.execution_mode: properties.hint.ExecutionMode.PERFORMANCE,
        #         },
        #     )
        # self.core.set_property("CPU",)
        cpu_optimization_capabilities =self.core.get_property("CPU","OPTIMIZATION_CAPABILITIES")
        print(cpu_optimization_capabilities)

        # 加载模型
        self.openvino = self.core.read_model(model=openvino_path)
        # 编译模型
        config = {
            properties.hint.performance_mode: properties.hint.PerformanceMode.LATENCY,
            properties.inference_num_threads: 119,
            properties.hint.inference_precision: 'bf16',
            properties.intel_cpu.denormals_optimization:True,
            # properties.hint.ModelDistributionPolicy:"TENSOR_PARALLEL",
            # properties.hint.enable_cpu_pinning(): False,
            # properties.hint.execution_mode: properties.hint.ExecutionMode.PERFORMANCE,
            # properties.hint.enable_hyper_threading(): False
            # properties.num_streams:2,
            # "CPU_THREADS_NUM": "60",
            # "CPU_BIND_THREAD": "YES",
        }
        self.compiled_model = self.core.compile_model(
            model=self.openvino,
            device_name='CPU',
            config=config,
        )
    def get_contact_matrix(
        self,
            gather_idxs_polymer_ligand,
            tokens_to_polymer_ligand_bonds_gather_mask,
            gather_idxs_ligand_ligand,
            tokens_to_ligand_ligand_bonds_gather_mask,
            num_tokens,
            # pair_activations: torch.Tensor
    ) -> torch.Tensor:
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=torch.float32)

        gather_mask_polymer_ligand = tokens_to_polymer_ligand_bonds_gather_mask.prod(dim=1)[:, None]*gather_idxs_polymer_ligand

        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds_gather_mask.prod(dim=1)[:, None]*gather_idxs_ligand_ligand
        gather_idxs = torch.concatenate(
            [ gather_mask_polymer_ligand.to(dtype=torch.int64),
            gather_mask_ligand_ligand.to(dtype=torch.int64)]
        )
        # print("gather_idxs",gather_idxs.shape,"contact_matrix",contact_matrix.shape)
        # print(gather_idxs[:, 0].shape,gather_idxs[:, 1].shape)
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0
        # print("after contact_matrix",contact_matrix.shape)
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = torch.tensor(
            1.0,
            dtype=contact_matrix.dtype,
            device=contact_matrix.device
        ).expand(gather_idxs.shape[0])
        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0
        return contact_matrix[:, :, None]

    def forward(self, batch: dict[str, torch.Tensor], target_feat_t) -> dict[str, torch.Tensor]:
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        # target_feat torch.Size([37, 447])

        single=np.zeros(shape=[num_res, self.evoformer_seq_channel], dtype=np.float32)
        pair=np.zeros(shape=[num_res, num_res, self.evoformer_pair_channel], dtype=np.float32)
        target_feat=target_feat_t.numpy()
        seq_mask = batch.token_features.mask
        attn_mask_seq = get_attn_mask(mask=seq_mask, dtype=torch.float32, device='cpu', num_heads=16,
                                      seq_len=num_res, batch_size=1)

        contact_matrix = self.get_contact_matrix(
            gather_idxs_polymer_ligand=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
            tokens_to_polymer_ligand_bonds_gather_mask=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
            gather_idxs_ligand_ligand=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
            tokens_to_ligand_ligand_bonds_gather_mask=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
            num_tokens=num_res,
        ).numpy()
        kwarg_inputs = {
            'target_feat': target_feat,
            'msa': batch.msa.rows.numpy(),
            'msa_mask': batch.msa.mask.numpy(),
            'deletion_matrix': batch.msa.deletion_matrix.numpy(),
            'token_index': batch.token_features.token_index.numpy(),
            'residue_index': batch.token_features.residue_index.numpy(),
            'asym_id': batch.token_features.asym_id.numpy(),
            'entity_id': batch.token_features.entity_id.numpy(),
            'sym_id': batch.token_features.sym_id.numpy(),

            'seq_mask': batch.token_features.mask.numpy(),
            'contact_matrix': contact_matrix,
            'template_aatype': batch.templates.aatype.numpy(),
            'template_atom_positions': batch.templates.atom_positions.numpy(),
            'template_atom_mask': batch.templates.atom_mask.numpy(),

            'attn_mask_seq':attn_mask_seq.numpy(),
        }
        infer_request = self.compiled_model.create_infer_request()
        single=ov.Tensor(single)
        pair=ov.Tensor(pair)
        # time1=time.time()
        idx = 2
        # 将数据填充到输入张量
        for value in kwarg_inputs.values():
            # input_names = input_key.names
            # input_name = input_key.get_any_name()
            # print("input_name", input_name," index" ,input_key.get_index())

            ov_tensor = ov.Tensor(value)
            # if input_name in inputs:
            infer_request.set_input_tensor(idx, ov_tensor)
            idx += 1
        for _ in range(self.num_recycles + 1):
            time1 = time.time()
            infer_request.set_input_tensor(0, single)
            infer_request.set_input_tensor(1,pair)
            infer_request.infer()
            single=infer_request.get_output_tensor(0)
            pair=infer_request.get_output_tensor(1)
            print('evo one cost time:', time.time()-time1)

        # print("dtype",c_pair.dtype,c_single.dtype)
        # print("shape",c_pair.shape,c_single.shape)
        # return embeddings
        output = {
            'single': torch.from_numpy(single.data),
            'pair': torch.from_numpy(pair.data),
            'target_feat': target_feat_t,
        }
        return output