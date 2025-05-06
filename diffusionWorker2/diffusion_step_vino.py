
from openvino import properties
import openvino as ov
import torch
import torch.nn as nn
import time
import torch.distributed as dist
from openvino import Core
import pathlib
import diffusionWorker2.misc.params as params
import diffusionWorker2.misc.feat_batch as feat_batch
from diffusionWorker2.network import diffusion_head
from diffusionWorker2.network.diffusion_step import DiffusionStep
from diffusionWorker2.premodel.pre_diffusion import DiffusionHead as pre_diffusion
from diffusionWorker2.network import atom_layout

import time
import numpy as np


def mask_mean(mask, value, dim=None, keepdim=False, eps=1e-10):
    broadcast_factor = 1.0
    numerator = np.sum(mask * value, axis=dim, keepdims=keepdim)
    denominator = np.clip(
        np.sum(mask, axis=dim, keepdims=keepdim) * broadcast_factor,
        a_min=eps,
        a_max=None
    )
    return numerator / denominator


def random_rotation(dtype):
    # 生成两个随机向量并正交化
    v0, v1 = np.random.randn(2, 3).astype(dtype)
    e0 = v0 / np.clip(np.linalg.norm(v0), a_min=1e-10, a_max=None)
    v1 = v1 - e0 * np.dot(v1, e0)
    e1 = v1 / np.clip(np.linalg.norm(v1), a_min=1e-10, a_max=None)
    e2 = np.cross(e0, e1)
    return np.stack([e0, e1, e2], axis=0)


def random_augmentation(
        positions: np.ndarray,
        mask: np.ndarray,
) -> np.ndarray:
    """Apply random rigid augmentation (NumPy version)."""

    # 计算掩码加权中心点
    center = mask_mean(
        mask[..., None], positions, dim=(-2, -3), keepdim=True, eps=1e-6
    ).astype(positions.dtype)

    # 生成随机旋转矩阵
    rot = random_rotation(dtype=positions.dtype)

    # 生成随机平移向量
    translation = np.random.randn(3).astype(positions.dtype)

    # 应用刚体变换
    augmented_positions = np.matmul(positions - center, rot) + translation

    # 保持掩码形状并返回
    return augmented_positions * mask[..., None]


class diffusion_vino():
    def __init__(self, num_recycles: int = 10, num_samples: int = 5,diffusion_steps: int = 200):
        super(diffusion_vino, self).__init__()
        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.pre_model=pre_diffusion()
        self.SIGMA_DATA = 16.0

    def import_diffusion_head_params(self,model_path: pathlib.Path):
        params.import_pre_model_params(self,model_path)


    def initOpenvinoModel(self,openvino_path):
        self.core = ov.Core()

        # 加载模型
        self.openvino = self.core.read_model(model=openvino_path)
        # 编译模型
        config = {
            properties.hint.performance_mode: properties.hint.PerformanceMode.LATENCY,
            properties.inference_num_threads:60,
            properties.hint.inference_precision: 'bf16',
            properties.intel_cpu.denormals_optimization: True,
        }
        self.compiled_model = self.core.compile_model(
            model=self.openvino,
            device_name='CPU',
            config=config,
        )


    def noise_schedule(self,t, smin=0.0004, smax=160.0, p=7):
        return (
                self.SIGMA_DATA
                * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
        )

    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        single,pair,target_feat,real_feat,index=0
        # embeddings: dict[str, torch.Tensor],
    ) :
        """Sample using denoiser on batch."""
        pred_dense_atom_mask_t=batch.predicted_structure_info.atom_mask
        pred_dense_atom_mask = pred_dense_atom_mask_t.numpy()
        device = pred_dense_atom_mask.device
        # print("device:",device)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device)).numpy()


        len=pred_dense_atom_mask.shape[0]
        positions = np.random.randn(5,len,24,3)[index].astype(np.float32)
        # positions = np.random.randn(len,24,3).astype(np.float32)

        # print(noise_levels)
        # positions = torch.randn(pred_dense_atom_mask_t.shape + (3,), device=device).numpy()

        positions *= noise_levels[0]

        acat_atoms_to_q_gather_idxs = batch.atom_cross_att.token_atoms_to_queries.gather_idxs
        acat_atoms_to_q_gather_mask = batch.atom_cross_att.token_atoms_to_queries.gather_mask

        queries_mask = atom_layout.convertV2(
            acat_atoms_to_q_gather_idxs,
            acat_atoms_to_q_gather_mask,
            torch.from_numpy(pred_dense_atom_mask),
            layout_axes=(-2, -1),
        ).contiguous()

        (trunk_single_cond, trunk_pair_cond, queries_single_cond,
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


        single_c = single
        pair_c = pair
        target_feat_c = target_feat
        inputs = {
            "single": single_c.numpy().astype(np.float32),
            "pair": pair_c.numpy().astype(np.float32),
            "target_feat": target_feat_c.cpu().numpy().astype(np.float32),
            "token_index": batch.token_features.token_index.cpu().numpy().astype(np.int32),
            "residue_index": batch.token_features.residue_index.cpu().numpy().astype(np.int32),
            "asym_id": batch.token_features.asym_id.cpu().numpy().astype(np.int32),
            "entity_id": batch.token_features.entity_id.cpu().numpy().astype(np.int32),
            "sym_id": batch.token_features.sym_id.cpu().numpy().astype(np.int32),

            "seq_mask": batch.token_features.mask.cpu().numpy().astype(np.bool),
            "pred_dense_atom_mask": pred_dense_atom_mask,


            "ref_ops": batch.ref_structure.positions.cpu().numpy().astype(np.float32),
            "ref_mask": batch.ref_structure.mask.cpu().numpy().astype(np.bool),
            "ref_element": batch.ref_structure.element.cpu().numpy().astype(np.int32),
            "ref_charge": batch.ref_structure.charge.cpu().numpy().astype(np.float32),
            "ref_atom_name_chars": batch.ref_structure.atom_name_chars.cpu().numpy().astype(np.int32),
            "ref_space_uid": batch.ref_structure.ref_space_uid.cpu().numpy().astype(np.int32),

            # 交叉注意力相关输入（以 acat_ 开头的参数）
            "acat_atoms_to_q_gather_idxs": batch.atom_cross_att.token_atoms_to_queries.gather_idxs.cpu().numpy().astype(
                np.int64),
            "acat_atoms_to_q_gather_mask": batch.atom_cross_att.token_atoms_to_queries.gather_mask.cpu().numpy().astype(
                np.bool),
            "acat_q_to_k_gather_idxs": batch.atom_cross_att.queries_to_keys.gather_idxs.cpu().numpy().astype(
                np.int64),
            "acat_q_to_k_gather_mask": batch.atom_cross_att.queries_to_keys.gather_mask.cpu().numpy().astype(
                np.bool),
            "acat_t_to_q_gather_idxs": batch.atom_cross_att.tokens_to_queries.gather_idxs.cpu().numpy().astype(
                np.int64),
            "acat_t_to_q_gather_mask": batch.atom_cross_att.tokens_to_queries.gather_mask.cpu().numpy().astype(
                np.bool),
            "acat_q_to_atom_gather_idxs": batch.atom_cross_att.queries_to_token_atoms.gather_idxs.cpu().numpy().astype(
                np.int64),
            "acat_q_to_atom_gather_mask": batch.atom_cross_att.queries_to_token_atoms.gather_mask.cpu().numpy().astype(
                np.bool),
            "acat_t_to_k_gather_idxs": batch.atom_cross_att.tokens_to_keys.gather_idxs.cpu().numpy().astype(
                np.int64),
            "acat_t_to_k_gather_mask": batch.atom_cross_att.tokens_to_keys.gather_mask.cpu().numpy().astype(
                np.bool),
            # 参考结构相关输入
            # 'positions': positions,
            # 'noise_level_prev': np.array(noise_levels[0],dtype=np.float32),
            # 'noise_level': np.array(noise_levels[1],dtype=np.float32),
        }

        print("diffusion2 start sample diffusion",positions.shape)
        infer_request = self.compiled_model.create_infer_request()
        idx = 0
        for key in inputs.keys():
            ov_tensor = ov.Tensor(inputs[key])
            # if input_name in inputs:
            infer_request.set_input_tensor(idx, ov_tensor)
            idx += 1
        # output_tensor=None
        # print(idx)
        sum_time=0
        for step_idx in range(self.diffusion_steps):
            time1 = time.time()
            # positions=diffusion_head.random_augmentation(torch.from_numpy(positions),pred_dense_atom_mask_t).numpy()
            positions=random_augmentation(positions,mask=pred_dense_atom_mask)

            noise_level_prev=noise_levels[step_idx]
            noise_level=noise_levels[1+step_idx]
            # # print(ov.Tensor(noise_level_prev).data)
            # # 将数据填充到输入张量
            infer_request.set_input_tensor(26,ov.Tensor(positions))
            infer_request.set_input_tensor(27,ov.Tensor(np.array(noise_level_prev)))
            infer_request.set_input_tensor(28,ov.Tensor(np.array(noise_level)))
            sum_time+=time.time()-time1
            infer_request.infer()
            positions = infer_request.get_output_tensor(0).data
        print('conversion time:',sum_time)
        return torch.from_numpy(positions)

    def forward(self, batch, single, pair, target_feat,real_feat,index,seq_mask=None,):
        batch = feat_batch.Batch.from_data_dict(batch)
        print("start:" ,index)
        return self._sample_diffusion(batch,
            single, pair, target_feat,seq_mask,real_feat=real_feat,index=index
        )
