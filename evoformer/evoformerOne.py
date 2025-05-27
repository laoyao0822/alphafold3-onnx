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
from typing import Optional
# import manual
# import fan_intergate
from evoformer import preprocess
from evoformer.network import featurization
from evoformer.network.pairformer import EvoformerBlock, PairformerBlock
from evoformer.network.template import TemplateEmbedding
from alphafold3.constants import residue_names
from evoformer.misc import protein_data_processing
from evoformer.network.dot_product_attention import get_attn_mask
# import onnxscript

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


    def _seq_pair_embedding(
        self, target_feat: torch.Tensor
    ) :
        """Generated Pair embedding from sequence."""
        # left_single = self.left_single(target_feat)[:, None]
        # right_single = self.right_single(target_feat)[None]
        # pair_activations = left_single + right_single
        # pair_mask = (mask[:, None] * mask[None, :]).to(dtype=target_feat.dtype)
        # return self.left_single(target_feat)[:, None] + self.right_single(target_feat)[None]
        return self.left_single(target_feat)[:, None]+self.right_single(target_feat)[None]


    def _embed_template_pair(
        self,
        # batch: feat_batch.Batch,
        asym_id,
        template_aatype, template_atom_positions, template_atom_mask,
        pair_activations: torch.Tensor,
        # pair_mask: Optional[torch.Tensor] = None,
        # attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        # templates = batch.templates
        # asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        # print(asym_id)
        multichain_mask = (asym_id[:, None] ==
                           asym_id[None, :])
        # print("astm_id",multichain_mask.shape)
        # print(multichain_mask.dtype)
        template_act = self.template_embedding(
            query_embedding=pair_activations,
            # templates=templates,
            template_aatype=template_aatype, template_atom_positions=template_atom_positions,
            template_atom_mask=template_atom_mask,
            multichain_mask_2d=multichain_mask,
            # padding_mask_2d=pair_mask,
            # attn_mask=attn_mask,

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
        rows, mask, deletion_matrix,
        pair_activations: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype
        rows, mask, deletion_matrix= featurization.shuffle_msa_runcate(rows, mask, deletion_matrix,num_msa=self.num_msa)

        msa_mask = mask.to(dtype=dtype).contiguous()
        msa_feat = self.create_msa_feat(rows,deletion_matrix).to(dtype=dtype).contiguous()
        msa_activations = self.msa_activations(msa_feat).contiguous()
        msa_activations += self.extra_msa_target_feat(target_feat)[None].contiguous()


        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
            )

        return pair_activations

    def forward(
        self,
        single, pair,
        target_feat,
        msa, msa_mask, deletion_matrix,

        asym_id,

        contact_matrix,
        template_aatype, template_atom_positions, template_atom_mask,
        rel_feat,
    ) :

        pair_activations = self.prev_embedding(
            self.prev_embedding_layer_norm(pair))

        pair_activations += self._seq_pair_embedding(
            target_feat
        )

        #_relative_encoding
        pair_activations += self.position_activations(rel_feat)

        pair_activations += self.bond_embedding(contact_matrix)

        single = self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(single)).to(dtype=pair.dtype)

        single += self.single_activations(target_feat)

        pair_activations=pair_activations.to(dtype=pair.dtype).contiguous()

        pair_activations = self._embed_template_pair(
            asym_id=asym_id, template_aatype=template_aatype,
            template_atom_positions=template_atom_positions, template_atom_mask=template_atom_mask,
            pair_activations=pair_activations,
        ).contiguous()


        pair_activations = self._embed_process_msa(
            msa, msa_mask, deletion_matrix,
            pair_activations=pair_activations,
            target_feat=target_feat,
        ).contiguous()


        single=single.contiguous()
        for pairformer_b in self.trunk_pairformer:
            pair_activations, single = pairformer_b(
                pair_activations, single=single)

        return single,pair_activations

class EvoFormerOne():

    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(EvoFormerOne, self).__init__()

        self.num_recycles = num_recycles
        # self.num_recycles = 2
        # self.num_recycles = 0
        self.num_samples = num_samples

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer = Evoformer()


    def getOnnxModel(self, batch, target_feat, save_path, ):
        # batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        pair = torch.zeros(
            [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
            dtype=torch.float32,
        )
        single = torch.zeros(
            [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
        )
        # target_feat=target_feat.to(device=target_feat.device)
        contact_matrix = preprocess.get_contact_matrix(
            gather_idxs_polymer_ligand=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
            tokens_to_polymer_ligand_bonds_gather_mask=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
            gather_idxs_ligand_ligand=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
            tokens_to_ligand_ligand_bonds_gather_mask=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
            num_tokens=num_res,
        )
        seq_mask=batch.token_features.mask
        edge_number = torch.export.Dim('edge_number', min=10, max=1500)
        output_names = ["single_out", "pair_out"]
        ordered_keys=['single', 'pair', 'target_feat',
                        'msa', 'msa_mask', 'deletion_matrix',
                        'asym_id',
                        'contact_matrix',
                        'template_aatype', 'template_atom_positions', 'template_atom_mask',
                        'rel_feat'
                        ]
        rel_feat = preprocess.get_rel_feat(token_index=batch.token_features.token_index,
                                           residue_index=batch.token_features.residue_index,
                                           asym_id=batch.token_features.asym_id,
                                           entity_id=batch.token_features.entity_id,
                                           sym_id=batch.token_features.sym_id, dtype=pair.dtype).contiguous()
        kwarg_inputs = {
            'single': single.to(dtype=torch.float32),
            'pair': pair.to(dtype=torch.float32),
            'target_feat': target_feat.to(dtype=torch.float32),
            'msa': batch.msa.rows.to(dtype=torch.float32),
            'msa_mask': batch.msa.mask,
            'deletion_matrix': batch.msa.deletion_matrix.to(dtype=torch.float32),

            'asym_id': batch.token_features.asym_id,

            # 'seq_mask': batch.token_features.mask,
            'contact_matrix': contact_matrix.to(dtype=torch.float32),

            'template_aatype': batch.templates.aatype,
            'template_atom_positions': batch.templates.atom_positions.to(dtype=torch.float32),
            'template_atom_mask': batch.templates.atom_mask,
            'rel_feat':rel_feat.to(dtype=torch.float32),
            # 'attn_mask_seq':attn_mask_seq,
        }
        ordered_inputs = tuple(kwarg_inputs[key] for key in ordered_keys)

        opset_version=22
        print("opset: ", opset_version)
        print("start to save")

        seq_len = torch.export.Dim('seq_len', min=10, max=1600)
        torch.onnx.export(self.evoformer,
                      ordered_inputs, f=save_path, dynamo=True,
                      input_names=ordered_keys,
                      output_names=output_names,
                      optimize=True,
                      opset_version=opset_version,
                      export_params=True,

                      dynamic_shapes={
                          'single': {0: seq_len},
                          'pair': {0: seq_len, 1: seq_len},
                          'target_feat': {0: seq_len},

                          'msa': {1: seq_len},
                          'msa_mask': {1: seq_len},
                          'deletion_matrix': {1: seq_len},

                          'asym_id': {0: seq_len},
                          # 'seq_mask': {0: seq_len},
                          'contact_matrix':{0: seq_len,1: seq_len},
                          # 模板相关
                          'template_aatype': {1: seq_len},
                          'template_atom_positions': {1: seq_len},
                          'template_atom_mask': {1: seq_len},
                          'rel_feat':{0:seq_len,1:seq_len}
                          # 'attn_mask_seq':{2:seq_len,3:seq_len}
                      },
                          )
        print("save done successfully save evoformer to onnx: ",save_path)
        # exit(0)



    def forward(self, batch,target_feat
                # pair_activations,pair_mask,

                ) -> dict[str, torch.Tensor]:
        num_res = batch.num_res
        pair= torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=target_feat.dtype,
            )
        single=torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=pair.dtype, device=target_feat.device,
            )
        target_feat = target_feat.to(dtype=pair.dtype,device=target_feat.device).contiguous()

        contact_matrix=preprocess.get_contact_matrix(
            gather_idxs_polymer_ligand=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
            tokens_to_polymer_ligand_bonds_gather_mask = batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
            gather_idxs_ligand_ligand = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
            tokens_to_ligand_ligand_bonds_gather_mask = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
            num_tokens=num_res,
        ).to(dtype=pair.dtype,device=target_feat.device).contiguous()
        rel_feat=preprocess.get_rel_feat( token_index=batch.token_features.token_index,
                residue_index = batch.token_features.residue_index,
                asym_id = batch.token_features.asym_id,
                entity_id = batch.token_features.entity_id,
                sym_id = batch.token_features.sym_id,dtype=pair.dtype,device=target_feat.device).contiguous()

        seq_mask = batch.token_features.mask
        num_tokens = seq_mask.shape[0]

        template_atom_positions=batch.templates.atom_positions.to(dtype=pair.dtype).contiguous()

        # time1=time.time()
        for recycle in range(self.num_recycles + 1):
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

                asym_id=batch.token_features.asym_id,
                # seq_mask=seq_mask,
                contact_matrix=contact_matrix,
                template_aatype = batch.templates.aatype,
                template_atom_positions = template_atom_positions,
                template_atom_mask = batch.templates.atom_mask,
                rel_feat=rel_feat,

            )
            print("evo one cost time:", time.time()-time1)

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

    def initOpenvinoModel(self,openvino_path,num_threads):
        self.core = ov.Core()
        cpu_optimization_capabilities =self.core.get_property("CPU","OPTIMIZATION_CAPABILITIES")
        print(cpu_optimization_capabilities)

        # 加载模型
        self.openvino = self.core.read_model(model=openvino_path)
        # 编译模型
        config = {
            properties.hint.performance_mode: properties.hint.PerformanceMode.LATENCY,
            properties.inference_num_threads: num_threads,
            properties.hint.inference_precision: 'bf16',
            properties.intel_cpu.denormals_optimization:True,
            # properties.num_streams:2,
            # "CPU_THREADS_NUM": "60",
            # "CPU_BIND_THREAD": "YES",
        }
        self.compiled_model = self.core.compile_model(
            model=self.openvino,
            device_name='CPU',
            config=config,
        )

    def forward(self, batch, target_feat_t) -> dict[str, torch.Tensor]:
        num_res = batch.num_res

        single=np.zeros(shape=[num_res, self.evoformer_seq_channel], dtype=np.float32)
        pair=np.zeros(shape=[num_res, num_res, self.evoformer_pair_channel], dtype=np.float32)
        target_feat=target_feat_t.numpy()

        contact_matrix = preprocess.get_contact_matrix(
            gather_idxs_polymer_ligand=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs,
            tokens_to_polymer_ligand_bonds_gather_mask=batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask,
            gather_idxs_ligand_ligand=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs,
            tokens_to_ligand_ligand_bonds_gather_mask=batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask,
            num_tokens=num_res,
        ).numpy()

        rel_feat = preprocess.get_rel_feat(token_index=batch.token_features.token_index,
                                           residue_index=batch.token_features.residue_index,
                                           asym_id=batch.token_features.asym_id,
                                           entity_id=batch.token_features.entity_id,
                                           sym_id=batch.token_features.sym_id, dtype=pair.dtype).contiguous()
        kwarg_inputs = {
            'target_feat': target_feat,
            'msa': batch.msa.rows.numpy(),
            'msa_mask': batch.msa.mask.numpy(),
            'deletion_matrix': batch.msa.deletion_matrix.numpy(),
            'asym_id': batch.token_features.asym_id.numpy(),

            'contact_matrix': contact_matrix,
            'template_aatype': batch.templates.aatype.numpy(),
            'template_atom_positions': batch.templates.atom_positions.numpy(),
            'template_atom_mask': batch.templates.atom_mask.numpy(),
            'rel_feat':rel_feat,
        }
        infer_request = self.compiled_model.create_infer_request()
        single=ov.Tensor(single)
        pair=ov.Tensor(pair)
        idx = 2
        # 将数据填充到输入张量
        for value in kwarg_inputs.values():
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

        output = {
            'single': torch.from_numpy(single.data),
            'pair': torch.from_numpy(pair.data),
            'target_feat': target_feat_t,
        }
        return output