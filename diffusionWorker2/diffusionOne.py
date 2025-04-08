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
from openvino import properties
import openvino as ov
import torch
import torch.nn as nn
import time
import torch.distributed as dist
import onnxruntime as ort
from openvino import Core
import pathlib
import diffusionWorker2.misc.params as params
import diffusionWorker2.misc.feat_batch as feat_batch
from diffusionWorker2.network import diffusion_head
from diffusionWorker2.network.diffusion_step import DiffusionStep
from openvino import convert_model
import onnx
import numpy as np
from diffusionWorker2.network.onnxop import convert_aten_layer_norm
class diffusion():
    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(diffusion, self).__init__()
        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.diffusion_head = diffusion_head.DiffusionHead()
        self.diffusion_head.eval()
        self.conversion_time=0

    def import_diffusion_head_params(self,model_path: pathlib.Path):
        params.import_diffusion_head_params(self.diffusion_head,model_path)


    def getOnnxModel(self,batch,single, pair, target_feat,save_path,):
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


        positions = diffusion_head.random_augmentation(
            positions=positions, mask=pred_dense_atom_mask
        )
        # positions = positions.to(device)
        # gamma = self.gamma_0 * (noise_level > self.gamma_min)
        # t_hat = noise_level_prev * (1 + gamma)
        #
        # noise_scale = self.noise_scale * \
        #               torch.sqrt(t_hat ** 2 - noise_level_prev ** 2)
        # noise = noise_scale * torch.randn(size=positions.shape, device=noise_scale.device)
        # # noise = noise_scale
        #
        # positions_noisy = positions + noise
        #
        # print("positions:",positions_noisy.shape)
        #
        # positions_noisy = positions_noisy
        # noise_level = t_hat

        seq_len = torch.export.Dim('seq_len', min=10, max=1600)
        edge_number = torch.export.Dim('edge_number', min=10, max=1500)

        ordered_keys = [
            'single', 'pair', 'target_feat',
            'token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id',
            'seq_mask','pred_dense_atom_mask',
            'ref_ops', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars', 'ref_space_uid',
            'acat_atoms_to_q_gather_idxs', 'acat_atoms_to_q_gather_mask',
            'acat_q_to_k_gather_idxs', 'acat_q_to_k_gather_mask',
            'acat_t_to_q_gather_idxs', 'acat_t_to_q_gather_mask',
            'acat_q_to_atom_gather_idxs', 'acat_q_to_atom_gather_mask',
            'acat_t_to_k_gather_idxs', 'acat_t_to_k_gather_mask',
            'positions','noise_level_prev','noise_level'
        ]

        output_names = ["positions_denoised"]

        kwarg_inputs = {
            'single': single,
            'pair': pair,
            'target_feat': target_feat,

            'token_index': batch.token_features.token_index,
            'residue_index': batch.token_features.residue_index,
            'asym_id': batch.token_features.asym_id,
            'entity_id': batch.token_features.entity_id,
            'sym_id': batch.token_features.sym_id,

            'seq_mask': batch.token_features.mask,
            'pred_dense_atom_mask': batch.predicted_structure_info.atom_mask,

            'ref_ops': batch.ref_structure.positions,
            'ref_mask': batch.ref_structure.mask,
            'ref_element': batch.ref_structure.element,
            'ref_charge': batch.ref_structure.charge,
            'ref_atom_name_chars': batch.ref_structure.atom_name_chars,
            'ref_space_uid': batch.ref_structure.ref_space_uid,

            'acat_atoms_to_q_gather_idxs': batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
            'acat_atoms_to_q_gather_mask': batch.atom_cross_att.token_atoms_to_queries.gather_mask,

            'acat_q_to_k_gather_idxs': batch.atom_cross_att.queries_to_keys.gather_idxs,
            'acat_q_to_k_gather_mask': batch.atom_cross_att.queries_to_keys.gather_mask,

            'acat_t_to_q_gather_idxs': batch.atom_cross_att.tokens_to_queries.gather_idxs,
            'acat_t_to_q_gather_mask': batch.atom_cross_att.tokens_to_queries.gather_mask,

            'acat_q_to_atom_gather_idxs': batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
            'acat_q_to_atom_gather_mask': batch.atom_cross_att.queries_to_token_atoms.gather_mask,

            'acat_t_to_k_gather_idxs': batch.atom_cross_att.tokens_to_keys.gather_idxs,
            'acat_t_to_k_gather_mask': batch.atom_cross_att.tokens_to_keys.gather_mask,

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
                              # 一维序列数据
                              'single': {0: seq_len},
                              'pair': {0: seq_len, 1: seq_len},
                              'target_feat': {0: seq_len},

                              'token_index': {0: seq_len},
                              'residue_index': {0: seq_len},
                              'asym_id': {0: seq_len},
                              'entity_id': {0: seq_len},
                              'sym_id': {0: seq_len},

                              'seq_mask': {0: seq_len},
                              'pred_dense_atom_mask': {0: seq_len},

                              # 图谱注意力相关
                              'acat_atoms_to_q_gather_idxs': {0: edge_number},
                              'acat_atoms_to_q_gather_mask': {0: edge_number},

                              'acat_q_to_k_gather_idxs': {0: edge_number},
                              'acat_q_to_k_gather_mask': {0: edge_number},

                              'acat_t_to_q_gather_idxs': {0: edge_number},
                              'acat_t_to_q_gather_mask': {0: edge_number},

                              'acat_q_to_atom_gather_idxs': {0: seq_len},
                              'acat_q_to_atom_gather_mask': {0: seq_len},

                              'acat_t_to_k_gather_idxs': {0: edge_number},
                              'acat_t_to_k_gather_mask': {0: edge_number},

                              # 参考结构
                              'ref_ops': {0: seq_len},
                              'ref_mask': {0: seq_len},
                              'ref_element': {0: seq_len},
                              'ref_charge': {0: seq_len},
                              'ref_atom_name_chars': {0: seq_len},
                              'ref_space_uid': {0: seq_len},

                              'positions': {0: seq_len},
                              # 'positions_noisy':{0:seq_len},
                              'noise_level_prev':{},
                              'noise_level':{}
                          },
                          training=torch.onnx.TrainingMode.EVAL
                          # dynamic_axes={'input': {}, 'output': {}},dynamo=True
                          )
        # print("save onnx done:", save_path)
        # exit(0)

    def initOnnxModel(self,onnx_path):
        check_model=onnx.load(onnx_path,load_external_data=True)
        onnx.checker.check_model(check_model)
        print("ONNX DIFFUSION check success")
        print("available provider",ort.get_available_providers())
        sess_options = ort.SessionOptions()
        # device_type = "CPU_FP16"
        # sess_options.num_of_threads = 59
        # sess_options.spip install onnxruntime-openvino
        # sess_options.enable_profiling = True
        # sess_options.intra_op_num_threads = 60
        # sess_options.add_session_config_entry('session.intra_op_thread_affinities','1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;46;47;48;49;50;51;52;53;54;55;56;57;58;59;60')  # set affinities of all 7 threads to cores in the first NUMA node
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # sess_options.optimized_model_filepath = "/root/pycharm/diffusion_head_onnx_opt/diffusion_head.onnx"
        # vino_device='CPU_FP32'
        ort.set_default_logger_severity(3)
        # providers = [("OpenVINOExecutionProvider", {
        #     "device_type": "CPU",  # 自动检测AMX
        #     "precision": "FP32",  # 启用BF16
        #     # "num_threads": 59,  # 根据CPU核心数调整
        # })]
    # sess_options.AddConfigEntry(ort.SessionOptions.kOrtSessionOptionEpContextEnable, "1");
        session = ort.InferenceSession(onnx_path, sess_options=sess_options,provider_options=['CPUExecutionProvider'])
        # session = ort.InferenceSession(onnx_path,options=sess_options,providers=providers)
        # session.set_providers(['OpenVINOExecutionProvider'],[{'device_type':device_type,'num_of_threads':59}])
        self.onnx_model = session
    def initOpenvinoModel(self,openvino_path):
        self.core = ov.Core()
        # self.core.set_property(
        # "CPU",
        #     {   properties.hint.execution_mode: properties.hint.ExecutionMode.PERFORMANCE,
        #         },
        #     )
        # 加载模型
        self.openvino = self.core.read_model(model=openvino_path)
        # 编译模型
        config = {
            properties.hint.performance_mode: properties.hint.PerformanceMode.LATENCY,
            properties.inference_num_threads:60,
            properties.hint.inference_precision: 'bf16',
            # properties.hint.execution_mode: properties.hint.ExecutionMode.PERFORMANCE
            # properties.num_streams:1,
            # "CPU_THREADS_NUM": "60",
            # "CPU_BIND_THREAD": "YES",
        }
        self.compiled_model = self.core.compile_model(
            model=self.openvino,
            device_name='CPU',
            config=config,
        )
        # 获取输入/输出信息
        self.inputs_info = self.compiled_model.inputs
        self.outputs_info = self.compiled_model.outputs

    def _apply_denoising_step(
            self,
            single, pair, target_feat,
            token_index, residue_index, asym_id, entity_id, sym_id,
            seq_mask,
            pred_dense_atom_mask,
            ref_ops, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,
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
            # embeddings: dict[str, torch.Tensor],
            positions: torch.Tensor,
            noise_level_prev: torch.Tensor,
            # mask: torch.Tensor,
            noise_level: torch.Tensor,
            # batch,
            USE_ONNX=False,
    ):
        # pred_dense_atom_mask = batch.predicted_structure_info.atom_mask

        positions = diffusion_head.random_augmentation(
            positions=positions, mask=pred_dense_atom_mask
        )
        #step1
        # gamma = self.gamma_0 * (noise_level > self.gamma_min)
        # t_hat = noise_level_prev * (1 + gamma)
        #
        # noise_scale = self.noise_scale * \
        #               torch.sqrt(t_hat ** 2 - noise_level_prev ** 2)
        # noise = noise_scale * torch.randn(size=positions.shape, device=noise_scale.device)
        # # noise = noise_scale
        # positions_noisy = positions + noise

        # print("t_hat:",t_hat.shape)
        # print("positions_noisy:",positions_noisy.shape)
        # positions_denoised=None
        # print("that ",t_hat.dtype,"noise_level",noise_level.dtype)
        if not USE_ONNX:
            positions_out = self.diffusion_head(
            single=single, pair=pair, target_feat=target_feat,
            token_index=token_index, residue_index=residue_index,
            asym_id=asym_id, entity_id=entity_id, sym_id=sym_id,
            seq_mask=seq_mask,
            pred_dense_atom_mask=pred_dense_atom_mask,
            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            acat_t_to_q_gather_idxs=acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask=acat_t_to_q_gather_mask,
            acat_q_to_atom_gather_idxs=acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask=acat_q_to_atom_gather_mask,
            acat_t_to_k_gather_idxs=acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask=acat_t_to_k_gather_mask,
            positions=positions,
            noise_level_prev=noise_level_prev,
            noise_level=noise_level,
            # batch=batch,
            # embeddings=embeddings,
            # use_conditioning=True
            )
        else:
            time1=time.time()
            inputs = {
                "single":single.cpu().numpy().astype(np.float32),
                "pair": pair.cpu().numpy().astype(np.float32),
                "target_feat": target_feat.cpu().numpy().astype(np.float32),
                "token_index": token_index.numpy().astype(np.int32),
                "residue_index": residue_index.numpy().astype(np.int32),
                "asym_id": asym_id.numpy().astype(np.int32),
                "entity_id": entity_id.numpy().astype(np.int32),
                "sym_id": sym_id.numpy().astype(np.int32),
                "seq_mask": seq_mask.numpy().astype(np.bool),
                "pred_dense_atom_mask": pred_dense_atom_mask.numpy().astype(np.bool),

                "ref_ops": ref_ops.numpy().astype(np.float32),
                "ref_mask": ref_mask.numpy().astype(np.bool),
                "ref_element": ref_element.numpy().astype(np.int32),
                "ref_charge": ref_charge.numpy().astype(np.float32),
                "ref_atom_name_chars": ref_atom_name_chars.numpy().astype(np.int32),
                "ref_space_uid": ref_space_uid.numpy().astype(np.int32),
             # 交叉注意力相关输入（以 acat_ 开头的参数）
                "acat_atoms_to_q_gather_idxs": acat_atoms_to_q_gather_idxs.numpy().astype(
                np.int64),
                "acat_atoms_to_q_gather_mask": acat_atoms_to_q_gather_mask.numpy().astype(
                np.bool),
                "acat_q_to_k_gather_idxs": acat_q_to_k_gather_idxs.numpy().astype(
                np.int64),
                "acat_q_to_k_gather_mask": acat_q_to_k_gather_mask.numpy().astype(
                np.bool),
                "acat_t_to_q_gather_idxs": acat_t_to_q_gather_idxs.numpy().astype(
                np.int64),
                "acat_t_to_q_gather_mask": acat_t_to_q_gather_mask.numpy().astype(
                np.bool),
                "acat_q_to_atom_gather_idxs": acat_q_to_atom_gather_idxs.numpy().astype(
                np.int64),
                "acat_q_to_atom_gather_mask": acat_q_to_atom_gather_mask.numpy().astype(
                np.bool),
                "acat_t_to_k_gather_idxs": acat_t_to_k_gather_idxs.numpy().astype(
                np.int64),
                "acat_t_to_k_gather_mask": acat_t_to_k_gather_mask.numpy().astype(
                np.bool),
                # 'positions_noisy': positions_noisy.numpy(),
                # 'noise_level': t_hat.numpy(),
                'positions': positions.numpy(),
                'noise_level_prev': noise_level_prev.numpy(),
                'noise_level': noise_level.numpy(),
        }
            infer_request = self.compiled_model.create_infer_request()

            idx=0
            # 将数据填充到输入张量
            for value in inputs.values():
                # input_names = input_key.names
                # input_name = input_key.get_any_name()
                # print("input_name", input_name," index" ,input_key.get_index())
                ov_tensor = ov.Tensor(value)
                # if input_name in inputs:
                infer_request.set_input_tensor(idx, ov_tensor)
                idx+=1
            self.conversion_time+=time.time()-time1
            # 执行同步推理
            infer_request.infer()
            # --------------------------- 获取输出 ------------------------------
            # 获取输出张量 (假设第一个输出是 positions_denoised)
            output_tensor = infer_request.get_output_tensor(0)
            positions_out = torch.from_numpy(output_tensor.data)

        # if torch.allclose(positions_denoised,positions_denoised2,1e-4):
        #     print("no difference")
        # else:
        #     print("difference")
        #     print("torch:------------------")
        #     print(positions_denoised)
        #     print("onnx")
        #     print(positions_denoised2)
        #     exit(0)

        #step1
        # grad = (positions_noisy - positions_denoised) / t_hat
        #
        # d_t = noise_level - t_hat
        # positions_out = positions_noisy + self.step_scale * d_t * grad
        return positions_out
        # return positions_out, noise_level

    def _sample_diffusion(
            self,
            batch: feat_batch.Batch,
            single, pair, target_feat,
            # embeddings: dict[str, torch.Tensor],
            USE_ONNX=False,
    ):
        """Sample using denoiser on batch."""

        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        device = pred_dense_atom_mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        noise_level = noise_levels[0]
        positions = torch.randn(
            pred_dense_atom_mask.shape + (3,), device=device).contiguous()
        positions *= noise_level

        # noise_level = torch.tile(noise_levels[None, 0], (num_samples,))
        single_c = single
        pair_c = pair
        target_feat_c = target_feat

        print("diffusion2 start sample diffusion", positions.shape)

        for step_idx in range(self.diffusion_steps):
            positions = self._apply_denoising_step(
                # single=embeddings['single'], pair=embeddings['pair'], target_feat=embeddings['target_feat'],
                single=single_c, pair=pair_c, target_feat=target_feat_c,
                token_index=batch.token_features.token_index,
                residue_index=batch.token_features.residue_index,
                asym_id=batch.token_features.asym_id,
                entity_id=batch.token_features.entity_id,
                sym_id=batch.token_features.sym_id,
                seq_mask=batch.token_features.mask,
                pred_dense_atom_mask=pred_dense_atom_mask,
                ref_ops=batch.ref_structure.positions,
                ref_mask=batch.ref_structure.mask,
                ref_element=batch.ref_structure.element,
                ref_charge=batch.ref_structure.charge,
                ref_atom_name_chars=batch.ref_structure.atom_name_chars,
                ref_space_uid=batch.ref_structure.ref_space_uid,
                acat_atoms_to_q_gather_idxs=batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
                acat_atoms_to_q_gather_mask=batch.atom_cross_att.token_atoms_to_queries.gather_mask,
                acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
                acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,
                acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,
                acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
                acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
                acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
                positions=positions, noise_level_prev=noise_levels[step_idx], noise_level=noise_levels[1 + step_idx],
                USE_ONNX=USE_ONNX,
            )
            # if (step_idx % 200) == 0:
            # print("noise_level: ", noise_level)
        print("conversion cost time :",self.conversion_time)
        return positions




    def forward(self, batch: dict[str, torch.Tensor], single, pair, target_feat,USE_ONNX=False):
        self.conversion_time=0
        batch = feat_batch.Batch.from_data_dict(batch)

        return self._sample_diffusion(batch,
            single, pair, target_feat,USE_ONNX
        )





class diffusion2():
    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(diffusion2, self).__init__()
        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.apply_step=DiffusionStep()

    def getOnnx(self,batch,single,pair,target_feat,onnx_path):
        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device='cpu'))

        noise_level = noise_levels[0]
        positions = torch.randn(pred_dense_atom_mask.shape + (3,), device='cpu').contiguous()
        positions *= noise_level

        seq_len = torch.export.Dim('seq_len', min=10, max=1600)
        edge_number = torch.export.Dim('edge_number', min=10, max=1500)

        ordered_keys = [
            'single', 'pair', 'target_feat',
            'token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id',
            'seq_mask', 'pred_dense_atom_mask',
            'ref_ops', 'ref_mask', 'ref_element', 'ref_charge', 'ref_atom_name_chars', 'ref_space_uid',
            'acat_atoms_to_q_gather_idxs', 'acat_atoms_to_q_gather_mask',
            'acat_q_to_k_gather_idxs', 'acat_q_to_k_gather_mask',
            'acat_t_to_q_gather_idxs', 'acat_t_to_q_gather_mask',
            'acat_q_to_atom_gather_idxs', 'acat_q_to_atom_gather_mask',
            'acat_t_to_k_gather_idxs', 'acat_t_to_k_gather_mask',
            'positions', 'noise_level_prev','noise_level'
        ]

        output_names = ["positions_out"]

        kwarg_inputs = {
            'single': single,
            'pair': pair,
            'target_feat': target_feat,

            'token_index': batch.token_features.token_index,
            'residue_index': batch.token_features.residue_index,
            'asym_id': batch.token_features.asym_id,
            'entity_id': batch.token_features.entity_id,
            'sym_id': batch.token_features.sym_id,

            'seq_mask': batch.token_features.mask,
            'pred_dense_atom_mask': batch.predicted_structure_info.atom_mask,

            'ref_ops': batch.ref_structure.positions,
            'ref_mask': batch.ref_structure.mask,
            'ref_element': batch.ref_structure.element,
            'ref_charge': batch.ref_structure.charge,
            'ref_atom_name_chars': batch.ref_structure.atom_name_chars,
            'ref_space_uid': batch.ref_structure.ref_space_uid,

            'acat_atoms_to_q_gather_idxs': batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
            'acat_atoms_to_q_gather_mask': batch.atom_cross_att.token_atoms_to_queries.gather_mask,

            'acat_q_to_k_gather_idxs': batch.atom_cross_att.queries_to_keys.gather_idxs,
            'acat_q_to_k_gather_mask': batch.atom_cross_att.queries_to_keys.gather_mask,

            'acat_t_to_q_gather_idxs': batch.atom_cross_att.tokens_to_queries.gather_idxs,
            'acat_t_to_q_gather_mask': batch.atom_cross_att.tokens_to_queries.gather_mask,

            'acat_q_to_atom_gather_idxs': batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
            'acat_q_to_atom_gather_mask': batch.atom_cross_att.queries_to_token_atoms.gather_mask,

            'acat_t_to_k_gather_idxs': batch.atom_cross_att.tokens_to_keys.gather_idxs,
            'acat_t_to_k_gather_mask': batch.atom_cross_att.tokens_to_keys.gather_mask,

            'positions':positions,
            'noise_level_prev':noise_level,
            'noise_level':noise_levels[1]

        }
        ordered_inputs = tuple(kwarg_inputs[key] for key in ordered_keys)
        opset_version = 22
        # custom_translation_table = {
        #     torch.ops.pkg.onnxscript.torch_lib._aten_layer_norm_onnx: convert_aten_layer_norm,
        # }

        print("opset: ", opset_version)
        print("start to save")
        torch.onnx.export(self.apply_step,
                          ordered_inputs, f=onnx_path,
                          input_names=ordered_keys,
                          output_names=output_names,
                          # custom_translation_table=custom_translation_table,
                          optimize=True,
                          opset_version=opset_version,
                          dynamo=True,
                          export_params=True,

                          dynamic_shapes={
                              # 一维序列数据
                              'single': {0: seq_len},
                              'pair': {0: seq_len, 1: seq_len},
                              'target_feat': {0: seq_len},

                              'token_index': {0: seq_len},
                              'residue_index': {0: seq_len},
                              'asym_id': {0: seq_len},
                              'entity_id': {0: seq_len},
                              'sym_id': {0: seq_len},

                              'seq_mask': {0: seq_len},
                              'pred_dense_atom_mask': {0: seq_len},

                              # 图谱注意力相关
                              'acat_atoms_to_q_gather_idxs': {0: edge_number},
                              'acat_atoms_to_q_gather_mask': {0: edge_number},

                              'acat_q_to_k_gather_idxs': {0: edge_number},
                              'acat_q_to_k_gather_mask': {0: edge_number},

                              'acat_t_to_q_gather_idxs': {0: edge_number},
                              'acat_t_to_q_gather_mask': {0: edge_number},

                              'acat_q_to_atom_gather_idxs': {0: seq_len},
                              'acat_q_to_atom_gather_mask': {0: seq_len},

                              'acat_t_to_k_gather_idxs': {0: edge_number},
                              'acat_t_to_k_gather_mask': {0: edge_number},

                              # 参考结构
                              'ref_ops': {0: seq_len},
                              'ref_mask': {0: seq_len},
                              'ref_element': {0: seq_len},
                              'ref_charge': {0: seq_len},
                              'ref_atom_name_chars': {0: seq_len},
                              'ref_space_uid': {0: seq_len},

                              'positions': {0: seq_len},
                              'noise_level_prev':{},
                              'noise_level': {}
                          },
                          training=torch.onnx.TrainingMode.EVAL
                          # dynamic_axes={'input': {}, 'output': {}},dynamo=True
                          )
        print("save onnx done:", onnx_path)
        # exit(0)



    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        single,pair,target_feat,
        # embeddings: dict[str, torch.Tensor],
    ) :
        """Sample using denoiser on batch."""
        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        device = pred_dense_atom_mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        noise_level = noise_levels[0]
        positions = torch.randn(
            pred_dense_atom_mask.shape + (3,), device=device).contiguous()
        positions *= noise_level
        # noise_level = torch.tile(noise_levels[None, 0], (num_samples,))
        single_c=single
        pair_c=pair
        target_feat_c=target_feat
        print("diffusion2 start sample diffusion",positions.shape)

        for step_idx in range(self.diffusion_steps):
            positions = self.apply_step(
                # single=embeddings['single'], pair=embeddings['pair'], target_feat=embeddings['target_feat'],
                single=single_c, pair=pair_c, target_feat=target_feat_c,
                token_index=batch.token_features.token_index,
                residue_index=batch.token_features.residue_index,
                asym_id=batch.token_features.asym_id,
                entity_id=batch.token_features.entity_id,
                sym_id=batch.token_features.sym_id,
                seq_mask=batch.token_features.mask,
                pred_dense_atom_mask=pred_dense_atom_mask,
                ref_ops=batch.ref_structure.positions,
                ref_mask=batch.ref_structure.mask,
                ref_element=batch.ref_structure.element,
                ref_charge=batch.ref_structure.charge,
                ref_atom_name_chars=batch.ref_structure.atom_name_chars,
                ref_space_uid=batch.ref_structure.ref_space_uid,
                acat_atoms_to_q_gather_idxs=batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
                acat_atoms_to_q_gather_mask=batch.atom_cross_att.token_atoms_to_queries.gather_mask,
                acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
                acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,
                acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,
                acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
                acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
                acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
                positions= positions,noise_level_prev= noise_levels[step_idx],noise_level= noise_levels[1 + step_idx]
            )
            # print(positions[0])
        return positions

    def forward(self, batch: dict[str, torch.Tensor], single, pair, target_feat,USE_ONNX=False):
        # self.conversion_time=0
        batch = feat_batch.Batch.from_data_dict(batch)
        return self._sample_diffusion(batch,
            single, pair, target_feat
        )



class DiffusionOne(nn.Module):

    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(DiffusionOne, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps


        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.diffusion_head = diffusion_head.DiffusionHead()

    def _apply_denoising_step(
        self,
        single,pair,target_feat,
        token_index, residue_index, asym_id, entity_id, sym_id,
        seq_mask,
        pred_dense_atom_mask,
        ref_ops, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,
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
        # embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_level_prev: torch.Tensor,
        # mask: torch.Tensor,
        noise_level: torch.Tensor
    ) :
        # pred_dense_atom_mask = batch.predicted_structure_info.atom_mask

        positions = diffusion_head.random_augmentation(
            positions=positions, mask=pred_dense_atom_mask
        )

        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        noise = noise_scale * \
            torch.randn(size=positions.shape, device=noise_scale.device)
        # noise = noise_scale
        positions_noisy = positions + noise

        positions_denoised = self.diffusion_head(
            single=single, pair=pair, target_feat=target_feat,
            token_index=token_index, residue_index=residue_index,
            asym_id=asym_id,entity_id=entity_id, sym_id=sym_id,

            seq_mask=seq_mask,

            pred_dense_atom_mask = pred_dense_atom_mask,

            ref_ops=ref_ops, ref_mask=ref_mask, ref_element=ref_element, ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,


            acat_atoms_to_q_gather_idxs=acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask=acat_atoms_to_q_gather_mask,
            acat_q_to_k_gather_idxs=acat_q_to_k_gather_idxs,
            acat_q_to_k_gather_mask=acat_q_to_k_gather_mask,
            acat_t_to_q_gather_idxs=acat_t_to_q_gather_idxs,
            acat_t_to_q_gather_mask=acat_t_to_q_gather_mask,
            acat_q_to_atom_gather_idxs=acat_q_to_atom_gather_idxs,
            acat_q_to_atom_gather_mask=acat_q_to_atom_gather_mask,
            acat_t_to_k_gather_idxs=acat_t_to_k_gather_idxs,
            acat_t_to_k_gather_mask=acat_t_to_k_gather_mask,

            positions_noisy=positions_noisy,
            noise_level=t_hat,
            # batch=batch,
            # embeddings=embeddings,
            # use_conditioning=True
            )
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad

        return positions_out, noise_level


    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        single,pair,target_feat,
        # embeddings: dict[str, torch.Tensor],
    ) :
        """Sample using denoiser on batch."""

        pred_dense_atom_mask = batch.predicted_structure_info.atom_mask
        device = pred_dense_atom_mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        noise_level = noise_levels[0]
        positions = torch.randn(
            pred_dense_atom_mask.shape + (3,), device=device).contiguous()
        positions *= noise_level

        # noise_level = torch.tile(noise_levels[None, 0], (num_samples,))
        single_c=single
        pair_c=pair
        target_feat_c=target_feat

        print("diffusion2 start sample diffusion",positions.shape)

        for step_idx in range(self.diffusion_steps):
            positions, noise_level = self._apply_denoising_step(
                # single=embeddings['single'], pair=embeddings['pair'], target_feat=embeddings['target_feat'],
                single=single_c, pair=pair_c, target_feat=target_feat_c,
                token_index=batch.token_features.token_index,
                residue_index=batch.token_features.residue_index,
                asym_id=batch.token_features.asym_id,
                entity_id=batch.token_features.entity_id,
                sym_id=batch.token_features.sym_id,
                seq_mask=batch.token_features.mask,
                pred_dense_atom_mask=pred_dense_atom_mask,
                ref_ops=batch.ref_structure.positions,
                ref_mask=batch.ref_structure.mask,
                ref_element=batch.ref_structure.element,
                ref_charge=batch.ref_structure.charge,
                ref_atom_name_chars=batch.ref_structure.atom_name_chars,
                ref_space_uid=batch.ref_structure.ref_space_uid,
                acat_atoms_to_q_gather_idxs=batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
                acat_atoms_to_q_gather_mask=batch.atom_cross_att.token_atoms_to_queries.gather_mask,
                acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
                acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,

                acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,

                acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
                acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
                acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
                positions= positions,noise_level_prev= noise_level,noise_level= noise_levels[1 + step_idx]

            )
        return positions




    def forward(self,batch: dict[str, torch.Tensor],single,pair,target_feat):
        batch = feat_batch.Batch.from_data_dict(batch)
        # embeddings = {
        #     'pair': pair,
        #     'single': single,
        #     'target_feat': target_feat,  # type: ignore
        # }
        return self._sample_diffusion(batch,
                                      single,pair,target_feat
        )







