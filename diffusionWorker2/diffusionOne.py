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
import pathlib
import diffusionWorker2.misc.params as params
import diffusionWorker2.misc.feat_batch as feat_batch
from diffusionWorker2.network import diffusion_head
from diffusionWorker2.network import atom_layout
from diffusionWorker2.premodel.pre_diffusion import DiffusionHead as pre_diffusion
import numpy as np


class diffusion():
    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(diffusion, self).__init__()
        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        # self.diffusion_steps = 2
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.diffusion_head = diffusion_head.DiffusionHead()
        self.diffusion_head.eval()
        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384
        self.pre_model=pre_diffusion()
        self.conversion_time=0

    def import_diffusion_head_params(self,model_path: pathlib.Path):
        params.import_diffusion_head_params(self,model_path)


    def getOnnxModel(self,batch,single, pair, target_feat,real_feat,save_path,):
        # batch = feat_batch.Batch.from_data_dict(batch)
        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask

        device = pred_dense_atom_mask.device
        # print("device:",device)

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        noise_level = noise_levels[0]
        positions = torch.randn(
            pred_dense_atom_mask.shape + (3,), device=device).contiguous()
        positions *= noise_level

        noise_level_prev = noise_level
        noise_level = noise_levels[1 + 1]

        acat_atoms_to_q_gather_idxs = batch.atom_cross_att.token_atoms_to_queries.gather_idxs
        acat_atoms_to_q_gather_mask = batch.atom_cross_att.token_atoms_to_queries.gather_mask

        queries_mask = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            pred_dense_atom_mask,
            layout_axes=(-2, -1),
        ).contiguous()

        target_feat = target_feat.to(dtype=positions.dtype)

        real_feat = real_feat.to(positions.dtype).contiguous()
        # noise_level = torch.tile(noise_levels[None, 0], (num_samples,))

        (trunk_single_cond, queries_single_cond,
         pair_act, keys_mask, keys_single_cond, pair_logits_cat) = self.pre_model(
            rel_features=real_feat,
            single=single, pair=pair, target_feat=target_feat,

            ref_ops=batch.ref_structure.positions,
            ref_mask=batch.ref_structure.mask,
            ref_element=batch.ref_structure.element,
            ref_charge=batch.ref_structure.charge,
            ref_atom_name_chars=batch.ref_structure.atom_name_chars,
            ref_space_uid=batch.ref_structure.ref_space_uid,

            queries_mask=queries_mask,

            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
            acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,

            acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
            acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,

            acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
            acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
            # use_conditioning=use_conditioning,
        )

        positions = diffusion_head.random_augmentation(
            positions=positions, mask=pred_dense_atom_mask
        ).contiguous()



        seq_len = torch.export.Dim('seq_len', min=10, max=10000)
        edge_number = torch.export.Dim('edge_number', min=10, max=10000)
        ordered_keys = [
            'queries_single_cond', 'pair_act', 'keys_mask',
            'keys_single_cond', 'trunk_single_cond', 'pair_logits_cat',
            # 'seq_mask',
            'pred_dense_atom_mask',
            'queries_mask',
            'acat_atoms_to_q_gather_idxs', 'acat_atoms_to_q_gather_mask',
            'acat_q_to_k_gather_idxs', 'acat_q_to_k_gather_mask',
            'acat_q_to_atom_gather_idxs', 'acat_q_to_atom_gather_mask',
            'positions','noise_level_prev','noise_level'
        ]
        output_names = ["positions_out"]

        kwarg_inputs = {
            'queries_single_cond': queries_single_cond,
            'pair_act': pair_act,
            'keys_mask': keys_mask,
            'keys_single_cond': keys_single_cond,
            'trunk_single_cond': trunk_single_cond,
            'pair_logits_cat': pair_logits_cat,

            # 'seq_mask': batch.token_features.mask,
            'pred_dense_atom_mask': batch.predicted_structure_info.atom_mask,

            'queries_mask': queries_mask,

            'acat_atoms_to_q_gather_idxs': batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
            'acat_atoms_to_q_gather_mask': batch.atom_cross_att.token_atoms_to_queries.gather_mask,

            'acat_q_to_k_gather_idxs': batch.atom_cross_att.queries_to_keys.gather_idxs,
            'acat_q_to_k_gather_mask': batch.atom_cross_att.queries_to_keys.gather_mask,

            'acat_q_to_atom_gather_idxs': batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
            'acat_q_to_atom_gather_mask': batch.atom_cross_att.queries_to_token_atoms.gather_mask,

            'positions': positions,
            'noise_level_prev':noise_level_prev,
            'noise_level': noise_level,

        }
        ordered_inputs = tuple(kwarg_inputs[key] for key in ordered_keys)
        opset_version=22
        # custom_translation_table = {
        #     torch.ops.pkg.onnxscript.torch_lib._aten_layer_norm_onnx: convert_aten_layer_norm,
        # }

        print("opset: ",opset_version)
        print("start to save")
        torch.onnx.export(self.diffusion_head,
                          ordered_inputs, f=save_path,
                          input_names=ordered_keys,
                          output_names=output_names,
                          # custom_translation_table=custom_translation_table,
                          optimize=True,
                          opset_version=opset_version,
                          dynamo=True,
                          export_params=True,
                          dynamic_shapes={
                              'queries_single_cond': {0:edge_number},
                              'pair_act': {0:edge_number},
                              'keys_mask': {0:edge_number},
                              'keys_single_cond': {0:edge_number},
                              'trunk_single_cond': {0:seq_len},
                              'pair_logits_cat': {3:seq_len,4:seq_len},

                              # 'seq_mask': {0: seq_len},
                              'pred_dense_atom_mask': {0: seq_len},
                              'queries_mask': {0: edge_number},

                              # 图谱注意力相关
                              'acat_atoms_to_q_gather_idxs': {0: edge_number},
                              'acat_atoms_to_q_gather_mask': {0: edge_number},

                              'acat_q_to_k_gather_idxs': {0: edge_number},
                              'acat_q_to_k_gather_mask': {0: edge_number},

                              'acat_q_to_atom_gather_idxs': {0: seq_len},
                              'acat_q_to_atom_gather_mask': {0: seq_len},

                              # 参考结构

                              'positions': {0: seq_len},
                              # 'positions_noisy':{0:seq_len},
                              'noise_level_prev':{},
                              'noise_level':{}
                          },
                          # training=torch.onnx.TrainingMode.EVAL
                          # dynamic_axes={'input': {}, 'output': {}},dynamo=True
                          )
        print("save onnx done:", save_path)
        # exit(0)


    def _apply_denoising_step(
            self,
            queries_single_cond,
            trunk_single_cond, pair_logits_cat,

            pair_act, keys_mask, keys_single_cond,

            # single,
            # token_index, residue_index, asym_id, entity_id, sym_id,
            pred_dense_atom_mask,

            queries_mask,

            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,

            acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask,

            acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask,

            positions: torch.Tensor,
            noise_level_prev: torch.Tensor,
            # mask: torch.Tensor,
            noise_level: torch.Tensor,
            # batch,
    ):
        # pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        positions = diffusion_head.random_augmentation(
            positions=positions, mask=pred_dense_atom_mask
        ).contiguous()

        positions_out = self.diffusion_head(
            queries_single_cond=queries_single_cond,
            pair_act=pair_act, keys_mask=keys_mask, keys_single_cond=keys_single_cond,

            trunk_single_cond=trunk_single_cond, pair_logits_cat=pair_logits_cat,

            pred_dense_atom_mask=pred_dense_atom_mask,


            queries_mask=queries_mask,
            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,

            acat_q_to_atom_gather_idxs=acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask=acat_q_to_atom_gather_mask,

            positions=positions,
            noise_level_prev=noise_level_prev,
            noise_level=noise_level,
            # batch=batch,
            # embeddings=embeddings,
            # use_conditioning=True
            )
        return positions_out


        # return positions_out, noise_level
    def hot_model(self,batch: feat_batch.Batch,target_feat,real_feat,seq_mask):
        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        device = pred_dense_atom_mask.device
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))
        num_res = batch.num_res

        noise_level = noise_levels[0]
        positions = torch.randn((self.num_samples,) +
                                pred_dense_atom_mask.shape + (3,), device=device, dtype=torch.bfloat16)[
            0].contiguous()
        positions *= noise_level

        acat_atoms_to_q_gather_idxs = batch.atom_cross_att.token_atoms_to_queries.gather_idxs
        acat_atoms_to_q_gather_mask = batch.atom_cross_att.token_atoms_to_queries.gather_mask

        queries_mask = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            pred_dense_atom_mask,
            layout_axes=(-2, -1),
        ).contiguous()
        pair = torch.zeros(
            [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
            dtype=torch.bfloat16,
        )
        single = torch.zeros(
            [num_res, self.evoformer_seq_channel], dtype=pair.dtype, device=target_feat.device,
        )
        (trunk_single_cond, queries_single_cond,
         pair_act, keys_mask, keys_single_cond, pair_logits_cat) = self.pre_model(
            rel_features=real_feat,
            single=single, pair=pair, target_feat=target_feat,
            ref_ops=batch.ref_structure.positions,
            ref_mask=batch.ref_structure.mask,
            ref_element=batch.ref_structure.element,
            ref_charge=batch.ref_structure.charge,
            ref_atom_name_chars=batch.ref_structure.atom_name_chars,
            ref_space_uid=batch.ref_structure.ref_space_uid,

            queries_mask=queries_mask,

            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
            acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,

            acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
            acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,

            acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
            acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
            # use_conditioning=use_conditioning,
        )

        trunk_single_cond = trunk_single_cond.to(dtype=positions.dtype).contiguous()
        queries_single_cond = queries_single_cond.to(dtype=positions.dtype).contiguous()
        pair_act = pair_act.to(dtype=positions.dtype).contiguous()
        keys_mask = keys_mask.contiguous()
        keys_single_cond = keys_single_cond.to(dtype=positions.dtype).contiguous()
        pair_logits_cat = pair_logits_cat.to(dtype=positions.dtype).contiguous()

        for step_idx in range(10):
            positions = self._apply_denoising_step(
                queries_single_cond=queries_single_cond,
                pair_act=pair_act, keys_mask=keys_mask, keys_single_cond=keys_single_cond,
                # single=embeddings['single'], pair=embeddings['pair'], target_feat=embeddings['target_feat'],
                trunk_single_cond=trunk_single_cond, pair_logits_cat=pair_logits_cat,
                pred_dense_atom_mask=pred_dense_atom_mask,

                queries_mask=queries_mask,

                acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
                acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
                acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,

                acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
                positions=positions, noise_level_prev=noise_levels[step_idx], noise_level=noise_levels[1 + step_idx],
            )

    def _sample_diffusion(
            self,
            batch: feat_batch.Batch,
            single, pair, target_feat,real_feat,
            index=0
            # embeddings: dict[str, torch.Tensor],
    ):
        """Sample using denoiser on batch."""

        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        device = pred_dense_atom_mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        noise_level = noise_levels[0]
        positions = torch.randn((self.num_samples,)+
            pred_dense_atom_mask.shape + (3,), device=device,dtype=pair.dtype)[index].contiguous()
        positions *= noise_level

        acat_atoms_to_q_gather_idxs = batch.atom_cross_att.token_atoms_to_queries.gather_idxs
        acat_atoms_to_q_gather_mask = batch.atom_cross_att.token_atoms_to_queries.gather_mask

        queries_mask = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            pred_dense_atom_mask,
            layout_axes=(-2, -1),
        ).contiguous()


        target_feat=target_feat.to(dtype=positions.dtype)


        real_feat=real_feat.to(positions.dtype).contiguous()
        # noise_level = torch.tile(noise_levels[None, 0], (num_samples,))

        (trunk_single_cond ,queries_single_cond,
         pair_act,keys_mask,keys_single_cond,pair_logits_cat)= self.pre_model(
            rel_features=real_feat,
            single=single, pair=pair, target_feat=target_feat,
            ref_ops=batch.ref_structure.positions,
            ref_mask=batch.ref_structure.mask,
            ref_element=batch.ref_structure.element,
            ref_charge=batch.ref_structure.charge,
            ref_atom_name_chars=batch.ref_structure.atom_name_chars,
            ref_space_uid=batch.ref_structure.ref_space_uid,

            queries_mask=queries_mask,

            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
            acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,

            acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
            acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,

            acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
            acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
            # use_conditioning=use_conditioning,
        )
        trunk_single_cond=trunk_single_cond.to(dtype=positions.dtype).contiguous()
        queries_single_cond=queries_single_cond.to(dtype=positions.dtype).contiguous()
        pair_act=pair_act.to(dtype=positions.dtype).contiguous()
        keys_mask=keys_mask.contiguous()
        keys_single_cond=keys_single_cond.to(dtype=positions.dtype).contiguous()
        pair_logits_cat=pair_logits_cat.to(dtype=positions.dtype).contiguous()

        for step_idx in range(self.diffusion_steps):
            positions = self._apply_denoising_step(
                queries_single_cond=queries_single_cond,
                pair_act=pair_act,keys_mask=keys_mask,keys_single_cond=keys_single_cond,

                trunk_single_cond=trunk_single_cond, pair_logits_cat=pair_logits_cat,
                pred_dense_atom_mask=pred_dense_atom_mask,
                queries_mask=queries_mask,
                acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
                acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
                acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,

                acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
                positions=positions, noise_level_prev=noise_levels[step_idx], noise_level=noise_levels[1 + step_idx],
            )

        return positions

    def forward(self, batch, single, pair, target_feat,real_feat,index=0):
        # self.conversion_time=0
        # batch = feat_batch.Batch.from_data_dict(batch)
        return self._sample_diffusion(batch,
            single, pair, target_feat,real_feat=real_feat,index=index
        )








