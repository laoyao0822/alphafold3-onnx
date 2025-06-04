import openvino as ov
import openvino.properties as properties
import torch

class ConfidenceVino():
    def __init__(self):
        super(ConfidenceVino, self).__init__()

    def initOpenvinoModel(self, openvino_path,num_threads):
        self.core = ov.Core()
        cpu_optimization_capabilities = self.core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
        print(cpu_optimization_capabilities)
        # 加载模型
        self.openvino = self.core.read_model(model=openvino_path)
        # 编译模型
        config = {
            properties.hint.performance_mode: properties.hint.PerformanceMode.LATENCY,
            properties.inference_num_threads: num_threads,
            properties.hint.inference_precision: 'bf16',
            properties.intel_cpu.denormals_optimization: True,
            # properties.num_streams:2,
            # "CPU_THREADS_NUM": "60",
            # "CPU_BIND_THREAD": "YES",
        }
        self.compiled_model = self.core.compile_model(
            model=self.openvino,
            device_name='CPU',
            config=config,
        )
    def forward(self, batch,embeddings,positions,):
        infer_request = self.compiled_model.create_infer_request()
        kwarg_inputs = {
            'dense_atom_positions': positions.to(torch.float32).numpy(),
            'single': embeddings['single'].to(dtype=torch.float32).numpy(),
            'pair': embeddings['pair'].to(dtype=torch.float32).numpy(),
            'target_feat': embeddings['target_feat'].to(dtype=torch.float32).numpy(),
            'seq_mask': batch.token_features.mask.numpy(),
            'ta_to_pb_gather_idxs': batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_idxs.numpy(),
            'ta_to_pb_gather_mask ': batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_mask.numpy(),
            'asym_id': batch.token_features.asym_id.numpy(),
        }
        idx=0
        for value in kwarg_inputs.values():
            ov_tensor = ov.Tensor(value)
            # if input_name in inputs:
            infer_request.set_input_tensor(idx, ov_tensor)
            idx += 1
        infer_request.infer()

        predicted_lddt=torch.from_numpy(infer_request.get_input_tensor(0).data)
        predicted_experimentally_resolved=torch.from_numpy(infer_request.get_input_tensor(1).data)
        full_pde=torch.from_numpy(infer_request.get_input_tensor(2).data)
        average_pde=torch.from_numpy(infer_request.get_input_tensor(3).data)
        full_pae=torch.from_numpy(infer_request.get_input_tensor(4).data)
        tmscore_adjusted_pae_global=torch.from_numpy(infer_request.get_input_tensor(5).data)
        tmscore_adjusted_pae_interface=torch.from_numpy(infer_request.get_input_tensor(6).data)
        return predicted_lddt,predicted_experimentally_resolved,full_pde,average_pde,full_pae,tmscore_adjusted_pae_global,tmscore_adjusted_pae_global,tmscore_adjusted_pae_interface
