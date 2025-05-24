

import datetime
import random
from collections.abc import Sequence
import csv
import dataclasses
import os
import pathlib

import textwrap
from typing import overload

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
import alphafold3.cpp # type: ignore
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.model import features
from alphafold3.model import post_processing
from alphafold3.model import model
from alphafold3.model.components import utils
from evoformer.network.dot_product_attention import get_attn_mask
import fan_intergate
import numpy as np
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
import torch.utils._pytree as pytree

from torch.profiler import profile, record_function, ProfilerActivity

from torchfold3.alphafold3 import AlphaFold3
from torchfold3.misc.params import import_jax_weights_
from target_feat.TargetFeat import  TargetFeat
from torchfold3.config import *
from torchfold3.misc import feat_batch
from target_feat.misc import params as target_feat_params
from evoformer.evoformerOne import EvoFormerOne
from evoformer.evoformerOne import EvoformerVino
from evoformer.misc import params as evoformer_params

from distogram.network.head import DistogramHead
from distogram.misc import params as distogram_params

from confidenceWorker.confidence import ConfidenceOne
from confidenceWorker.misc import params as confidence_params
import time

from diffusionWorker2.diffusionOne import diffusion

from diffusionWorker2.misc import params as diffusion_params
from diffusionWorker2.diffusion_step_vino import diffusion_vino
from evoformer import preprocess

DIFFUSION_ONNX=False
SAVE_ONNX=False
# UseVino=False
SAVE_EVO_ONNX= False
USE_EVO_VINO= False
SAVE_CONFIDENCE_ONNX=False
USE_IPEX=False

_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = _HOME_DIR / 'models/model_103275239_1'
DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'
ONNX_PATH = '/root/asc25'
OPENVINO_PATH = '/root/asc25'


ONNX_PATH_DIFFUSION_PATH=ONNX_PATH+'/diffusion_head_onnx_2/diffusion_head.onnx'
EVO_ONNX_PATH = ONNX_PATH+'/evo_onnx/evoformer.onnx'
EVO_VINO_PATH=OPENVINO_PATH+'/evo_vino/model.xml'
DIFFUSION_OPENVINO_PATH=OPENVINO_PATH+'/diffusion_head_openvino/model.xml'

# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)

_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    False,
    'Whether to run the data pipeline on the fold inputs.',
)

_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)
_USE_DIFFUSION_VINO = flags.DEFINE_bool(
    'use_diffusion_vino',
    False,
    'Whether to run inference on the fold inputs.',
)
_DIFFUSION_VINO_PATH = flags.DEFINE_string(
    'diffusion_vino_path',
    DIFFUSION_OPENVINO_PATH,
    'Path to the model to use for inference.',
)

_CPU_INFERENCE = flags.DEFINE_bool(
    'cpu_inference',
    True,
    'Whether to run inference on the cpu.',
)

# control the number of threads used by the data pipeline.
_NUM_THREADS = flags.DEFINE_integer(
    'num_cpu_threads',
    8,
    'Number of threads to use for the data pipeline.',
)

#can not use buckets because it will make seq_mask have 0
# _BUCKETS = flags.DEFINE_list(
#     'buckets',
#     # pyformat: disable
#     ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
#      '3584', '4096', '4608', '5120'],
#     # pyformat: enable
#     'Strictly increasing order of token sizes for which to cache compilations.'
#     ' For any input with more tokens than the largest bucket size, a new bucket'
#     ' is created for exactly that number of tokens.',
# )

_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    24,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
)

_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.',
)

_RANK_ = flags.DEFINE_integer(
    'rank',
    0,
    'Number of the pipeline Chunk.',
)
_WOLRD_SIZE = flags.DEFINE_integer(
    'world_size',
    1,
    'Number of THE world Size.',
)
_CONFIDENCE_DP = flags.DEFINE_bool(
    'confidence_dp',
    True,
    'Whether to run inference on the cpu.',
)


import torch.distributed as dist
def setup(rank, world_size,master_addr='192.168.10.1', master_port='8082'):

    if _CPU_INFERENCE.value:
        print("start to set up multi cpu","rank:",rank,"world_size:",world_size)
        dist.init_process_group(
            backend='gloo',
            init_method='tcp://10.0.0.1:11499',
            rank=rank,
            world_size=world_size,
        )
    else:
        print("start to set up multi gpu", "rank:", rank, "world_size:", world_size)
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:8802',
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{rank}")
        )
        torch.cuda.set_device(rank)

class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        model_dir: pathlib.Path,
        device: torch.device,
    ):
        self._model_dir = model_dir
        self._device = device
        rank = _RANK_.value
        if _WOLRD_SIZE.value>1:
            setup(_RANK_.value, _WOLRD_SIZE.value)

        #import target feat
        print('import target feat')
        self.target_feat=TargetFeat()
        self.target_feat.eval()
        target_feat_params.import_jax_weights_(self.target_feat,model_dir)

        if USE_EVO_VINO:
            print('import evoformer vino')
            self.evoformer=EvoformerVino()
            self.evoformer.initOpenvinoModel(EVO_VINO_PATH)
        else:
            print('import evoformer torch')
            self.evoformer=EvoFormerOne()
            self.evoformer.evoformer.eval()
            evoformer_params.import_evoformer_jax_weights_(self.evoformer.evoformer,model_dir)

        if _USE_DIFFUSION_VINO.value:
            # self.diffusion=diffusion()
            self.diffusion=diffusion_vino()
            self.diffusion.initOpenvinoModel(_DIFFUSION_VINO_PATH.value)
            self.diffusion.import_diffusion_head_params(model_dir)

            # self.diffusion.initOnnxModel(ONNX_PATH)
        else:
            print("select diffusion")
            self.diffusion=diffusion()
            self.diffusion.diffusion_head.eval()
            self.diffusion.pre_model.eval()
            # self.diffusion.eval()
            diffusion_params.import_diffusion_head_params(self.diffusion,model_dir)

        self.confidence=ConfidenceOne()
        confidence_params.import_jax_weights_(self.confidence,model_dir)
        self.confidence.confidence_head.eval()
        self.distogram=DistogramHead()
        self.distogram.eval()
        distogram_params.import_jax_weights_(self.distogram,model_dir)

        # Apply IPEX optimization for CPU if device is CPU
        if _CPU_INFERENCE.value:
            print("mkl",torch.backends.mkl.is_available(),"onednn",torch.backends.mkldnn.is_available())
            if  USE_IPEX:
                # pass
                import intel_extension_for_pytorch as ipex
                # import openvino as ov
                print("Applying Intel Extension for PyTorch optimizations...")
                self.target_feat = ipex.optimize(self.target_feat,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                self.evoformer.evoformer = ipex.optimize(self.evoformer.evoformer,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)

                # self.evoformer.evoformer = torch.compile(self.evoformer.evoformer, backend="ipex")
                if not _USE_DIFFUSION_VINO.value:
                    self.diffusion.pre_model = ipex.optimize(self.diffusion.pre_model,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                    self.diffusion.diffusion_head = ipex.optimize(self.diffusion.diffusion_head,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                # self.diffusion.diffusion_head=torch.compile(self.diffusion.diffusion_head,backend="ipex")
                self.confidence.confidence_head=ipex.optimize(self.confidence.confidence_head,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                self.distogram=ipex.optimize(self.distogram,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)

            if _CPU_FLUSH_DENORM_OPT:
                torch.set_flush_denormal(True)
            # self._model = torch.compile(self._model,backend="ipex")

        #将模型迁移到cuda
        if _CPU_INFERENCE.value == False:
            if rank>0:
                torch.cuda.set_device(rank)
            device=f"cuda:{rank}"
            self.target_feat = self.target_feat.to(device=device)
            self.confidence.confidence_head=self.confidence.confidence_head.to(device=device)
            self.diffusion.pre_model=self.diffusion.pre_model.to(device=device)
            self.diffusion.diffusion_head=self.diffusion.diffusion_head.to(device=device)
            self.evoformer.evoformer=self.evoformer.evoformer.to(device=device)
            self.distogram=self.distogram.to(device=device)
            print("Applying CUDA optimizations...")
            # print(torch._dynamo.list_backends())
            # self._model = torch.compile(self._model,backend="inductor",dynamic=False)

    @torch.inference_mode()
    def run_inference(
        self, featurised_example: features.BatchDict
    ) -> model.ModelResult:
        """Computes a forward pass of the model on a featurised example."""
        featurised_example = pytree.tree_map(
            torch.from_numpy, utils.remove_invalidly_typed_feats(
                featurised_example)
        )

        featurised_example = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(device=self._device).contiguous(),
            featurised_example,
        )

        featurised_example['deletion_mean'] = featurised_example['deletion_mean'].to(
            dtype=torch.float32)

        amp_type="cpu"
        device="cpu"
        if _CPU_INFERENCE.value==False:
            amp_type="cuda"
            device="cuda:0"
        #     if True:
        #         with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        #             print("Running inference with AMP on GPU...")
        #             result = self._model(featurised_example)
        #             result['__identifier__'] = self._model.__identifier__.numpy()
        #
        # else: # CPU Inference
        if _CPU_AMP_OPT:
                with torch.amp.autocast(amp_type, dtype=torch.bfloat16):
                    # print("Running inference with AMP on CPU...")
                    batch = feat_batch.Batch.from_data_dict(featurised_example)
                    time1=time.time()
                    seq_mask = batch.token_features.mask.contiguous()
                    num_tokens = seq_mask.shape[0]
                    print("start to process lenghth: ",num_tokens)
                    target_feat = self.target_feat(
                        aatype=batch.token_features.aatype,
                        profile=batch.msa.profile,
                        deletion_mean=batch.msa.deletion_mean,

                        pred_dense_atom_mask=batch.predicted_structure_info.atom_mask,

                        acat_atoms_to_q_gather_idxs=batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
                        acat_atoms_to_q_gather_mask=batch.atom_cross_att.token_atoms_to_queries.gather_mask,

                        acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                        acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,
                        acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
                        acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,
                        acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                        acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,

                        acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
                        acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
                        ref_ops=batch.ref_structure.positions,
                        ref_mask=batch.ref_structure.mask,
                        ref_element=batch.ref_structure.element,
                        ref_charge=batch.ref_structure.charge,
                        ref_atom_name_chars=batch.ref_structure.atom_name_chars,
                        ref_space_uid=batch.ref_structure.ref_space_uid
                    ).contiguous()
                    # print('target_feat: ',target_feat.device)


                    if(seq_mask==False).sum().item() !=0:
                        print('zero count of seq_mask is not zero,please cancel bucket:',(seq_mask==False).sum().item() !=0)
                        exit(0)
                    # attn_mask_4=pair_mask.to(torch.bool)[:, None, None, :].expand(-1,4,-1,-1).contiguous()
                    # pair_mask = pair_mask.to(dtype=torch.bool,device=device).contiguous()

                    rel_feat = preprocess.get_rel_feat(token_index=batch.token_features.token_index,
                                                       residue_index=batch.token_features.residue_index,
                                                       asym_id=batch.token_features.asym_id,
                                                       entity_id=batch.token_features.entity_id,
                                                       sym_id=batch.token_features.sym_id, dtype=target_feat.dtype,device=device)
                    if SAVE_EVO_ONNX:
                        self.evoformer.getOnnxModel(batch,target_feat,EVO_ONNX_PATH)

                    time1=time.time()
                    # with profile(activities=[ProfilerActivity.CPU],
                                 # profile_memory=False, record_shapes=False) as prof:
                    embeddings=self.evoformer.forward(batch,target_feat)
                    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=500))
                    # exit(0)
                    print("Evoformer took %.2f seconds" % (time.time() - time1))

                    # time1=time.time()
                    if _WOLRD_SIZE.value>1:
                        single_s=embeddings['single'].to(dtype=torch.bfloat16).contiguous()
                        pair_s=embeddings['pair'].to(dtype=torch.bfloat16).contiguous()
                        dist.broadcast(single_s, src=0,async_op=True)
                        dist.broadcast(pair_s, src=0,async_op=True)

                    pred_dense_atom_mask = batch.predicted_structure_info.atom_mask

                    positions = torch.zeros((_NUM_DIFFUSION_SAMPLES.value,) + pred_dense_atom_mask.shape + (3,),
                                            device=amp_type, dtype=torch.float32).contiguous()

                    if SAVE_ONNX:
                        self.diffusion.getOnnxModel(batch=batch,
                                            single=embeddings['single'], pair=embeddings['pair'],
                                            target_feat=target_feat,real_feat=rel_feat,
                                            save_path=DIFFUSION_OPENVINO_PATH)


                    sample_mask = batch.predicted_structure_info.atom_mask
                    # samples = self._sample_diffusion(batch, embeddings)

                    num_samples=_NUM_DIFFUSION_SAMPLES.value
                    #准备接收数据
                    predicted_lddt_all=torch.zeros((num_samples,num_tokens,24),dtype=torch.float32).contiguous()
                    predicted_experimentally_resolved_all=torch.zeros((num_samples,num_tokens,24),dtype=torch.float32).contiguous()
                    full_pde_all=torch.zeros((num_samples,num_tokens,num_tokens),dtype=torch.float32).contiguous()
                    full_pae_all=torch.zeros((num_samples,num_tokens,num_tokens),dtype=torch.float32).contiguous()
                    tmscore_adjusted_pae_global_all=torch.zeros((num_samples,num_tokens,num_tokens),dtype=torch.float32).contiguous()
                    tmscore_adjusted_pae_interface_all=torch.zeros((num_samples,num_tokens,num_tokens),dtype=torch.float32).contiguous()
                    average_pde_all=torch.zeros((num_samples,),dtype=torch.float32).contiguous()

                    #本地需要执行的diffusion数量
                    num_execute=_NUM_DIFFUSION_SAMPLES.value-_WOLRD_SIZE.value+1
                    # print('num to execute:',num_execute)
                    for i in range(num_execute):
                        # print("diffusion sample %d" % i)
                        time1 = time.time()
                        if not _USE_DIFFUSION_VINO.value:
                            positions[i] = self.diffusion.forward(batch,single=embeddings['single'], pair=embeddings['pair'],
                                                      target_feat=target_feat,real_feat=rel_feat,index=i,seq_mask=seq_mask,
                                                      )
                        else:
                            positions[i] = self.diffusion.forward(batch,single=embeddings['single'], pair=embeddings['pair'],
                                                      target_feat=target_feat,real_feat=rel_feat,seq_mask=seq_mask,
                                                                  index=i
                                                      )

                        print("diffusion cost time: ", time.time() - time1)

                    pos_waits=[]
                    for i in range(1,_WOLRD_SIZE.value):
                        pos_waits.append(dist.irecv(positions[num_samples-i],src=i))

                    #如果不使用confidence dp则需要执行完成所有的confidence
                    if not _CONFIDENCE_DP.value:
                        num_execute=5

                    for i in range(num_execute):
                        time1 = time.time()
                        (predicted_lddt, predicted_experimentally_resolved, full_pde, average_pde,
                         full_pae, tmscore_adjusted_pae_global,
                         tmscore_adjusted_pae_interface) = self.confidence.forward(batch=batch,
                                                                                   embeddings=embeddings,
                                                                                   positions=positions[i],
                                                                                   )
                        predicted_experimentally_resolved = predicted_experimentally_resolved.to(
                            dtype=torch.float32).contiguous()

                        predicted_lddt_all[i] = predicted_lddt
                        predicted_experimentally_resolved_all[i] = predicted_experimentally_resolved
                        full_pde_all[i] = full_pde
                        full_pae_all[i] = full_pae
                        average_pde_all[i] = average_pde
                        tmscore_adjusted_pae_global_all[i] = tmscore_adjusted_pae_global
                        tmscore_adjusted_pae_interface_all[i] = tmscore_adjusted_pae_interface

                        # confidence_output_per_sample.append(confidence_output)
                        print("confidence output time: ", time.time() - time1)

                    bin_edges,contact_probs=self.distogram(embeddings['pair'].clone())
                    distogram={
                            'bin_edges': bin_edges,
                            'contact_probs': contact_probs,
                    }
                    # time1=time.time()
                    for pos_wait in pos_waits:
                        pos_wait.wait()
                    # print('scuess recv positions cost  time: ', time.time() - time1)
                    final_dense_atom_mask = torch.tile(sample_mask[None], (_NUM_DIFFUSION_SAMPLES.value, 1, 1))
                    samples = {'atom_positions': positions, 'mask': final_dense_atom_mask}

                    #confidence的分布式结果收集
                    if _CONFIDENCE_DP.value and _WOLRD_SIZE.value>1:
                        slice_sizes = [
                            num_tokens * 24,  # predicted_lddt
                            num_tokens * 24,  # predicted_experimentally_resolved
                            num_tokens * num_tokens,  # full_pde
                            1,  # average_pde
                            num_tokens * num_tokens,  # full_pae
                            num_tokens * num_tokens,  # tmscore_adjusted_pae_global
                            num_tokens * num_tokens  # tmscore_adjusted_pae_interface
                        ]
                        packed_tensor=torch.zeros((num_tokens*(num_tokens*4+48)+1,),dtype=torch.float32).contiguous()

                        gather_list = [torch.empty_like(packed_tensor) for _ in range(_WOLRD_SIZE.value)]
                        dist.gather(packed_tensor, gather_list, dst=0)
                        for node_idx, node_data in enumerate(gather_list):
                            if (node_idx==0):
                                continue
                            else:
                                node_idx=num_samples-node_idx
                            # print('process node_idx',node_idx)
                            ptr = 0
                            # predicted_lddt
                            predicted_lddt_all[node_idx] = node_data[ptr:ptr + slice_sizes[0]].reshape(num_tokens,24)
                            ptr += slice_sizes[0]
                            # predicted_experimentally_resolved
                            predicted_experimentally_resolved_all[node_idx] = node_data[ptr:ptr + slice_sizes[1]].reshape(num_tokens, 24)
                            ptr += slice_sizes[1]
                            # 后续张量同理
                            full_pde_all[node_idx] = node_data[ptr:ptr + slice_sizes[2]].reshape(num_tokens, num_tokens)
                            ptr += slice_sizes[2]
                            average_pde_all[node_idx] = node_data[ptr:ptr + slice_sizes[3]].item()  # 标量直接取数值
                            ptr += slice_sizes[3]
                            full_pae_all[node_idx] = node_data[ptr:ptr + slice_sizes[4]].reshape(num_tokens, num_tokens)
                            ptr += slice_sizes[4]
                            tmscore_adjusted_pae_global_all[node_idx] = node_data[ptr:ptr + slice_sizes[5]].reshape(
                                num_tokens, num_tokens)
                            ptr += slice_sizes[5]
                            tmscore_adjusted_pae_interface_all[node_idx] = node_data[ptr:ptr + slice_sizes[6]].reshape(
                                num_tokens, num_tokens)

                    confidence_output={
                                'predicted_lddt': predicted_lddt_all,
                                'predicted_experimentally_resolved': predicted_experimentally_resolved_all,
                                'full_pde': full_pde_all,
                                'average_pde': average_pde_all,
                                'full_pae':full_pae_all,
                                'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global_all,
                                'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface_all,
                    }
                    result={
                            'diffusion_samples': samples,
                            'distogram': distogram,
                            **confidence_output,
                    }
                    result['__identifier__'] = self.target_feat.__identifier__.numpy()

        result = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(
                dtype=torch.float32) if x.dtype == torch.bfloat16 else x,
            result,
        )
        result = pytree.tree_map_only(
            torch.Tensor, lambda x: x.cpu().detach().numpy(), result)
        result['__identifier__'] = result['__identifier__'].tobytes()

        return result

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        """Generates structures from model outputs."""
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )
    def extract_embeddings(
        self,
        result: model.ModelResult,
        ) -> dict[str, np.ndarray] | None:
        """Extracts embeddings from model outputs."""
        embeddings = {}
        if 'single_embeddings' in result:
            embeddings['single_embeddings'] = result['single_embeddings']
        if 'pair_embeddings' in result:
            embeddings['pair_embeddings'] = result['pair_embeddings']
        return embeddings or None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
    """
    seed: int
    inference_results: Sequence[model.InferenceResult]
    full_fold_input: folding_input.Input
    embeddings: dict[str, np.ndarray] | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    print(f'Featurising data for seeds {fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, 
        buckets=buckets, 
        ccd=ccd, 
        verbose=True,
        conformer_max_iterations=conformer_max_iterations,
    )
    print(
        f'Featurising data for seeds {fold_input.rng_seeds} took '
        f' {time.time() - featurisation_start_time:.2f} seconds.'
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        print(f'Running model inference for seed {seed}...')
        if _CPU_INFERENCE.value==False:
            torch.cuda.synchronize()
        inference_start_time = time.time()

        # set the random seed for the model.
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = model_runner.run_inference(example)
        if _CPU_INFERENCE.value==False:
            torch.cuda.synchronize()
        print(
            f'Running model inference for seed {seed} took '
            f' {time.time() - inference_start_time:.2f} seconds.'
        )
        print(f'Extracting output structures (one per sample) for seed {seed}...')
        extract_structures = time.time()
        inference_results = model_runner.extract_structures(
            batch=example, result=result, target_name=fold_input.name
        )
        print(
            f'Extracting output structures (one per sample) for seed {seed} took '
            f' {time.time() - extract_structures:.2f} seconds.'
        )
        
        embeddings = model_runner.extract_embeddings(result)
        
        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
                embeddings=embeddings,
            )
        )
        print(
            'Running model inference and extracting output structures for seed'
            f' {seed} took  {time.time() - inference_start_time:.2f} seconds.'
        )
    print(
        'Running model inference and extracting output structures for seeds'
        f' {fold_input.rng_seeds} took '
        f' {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
    ) as f:
        f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    output_terms = (
        pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    ).read_text(errors='replace')

    os.makedirs(output_dir, exist_ok=True)
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            post_processing.write_output(
                inference_result=result, output_dir=sample_dir
            )
            ranking_score = float(result.metadata['ranking_score'])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result
        if embeddings := results_for_seed.embeddings:
            embeddings_dir = os.path.join(output_dir, f'seed-{seed}_embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            post_processing.write_embeddings(
                embeddings=embeddings, output_dir=embeddings_dir
        )
    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=output_terms,
            name=job_name,
        )
        # Save csv of ranking scores with seeds and sample indices, to allow easier
        # comparison of ranking scores across different runs.
        with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)
            
            if _SCORE_TABLE_TIMESTAMP:
                import datetime
                # Add the current time as the last row
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow(['Current Time', current_time])
            if _SCORE_TABLE_DISPALY_TOP:
                # Add the top ranking score
                writer.writerow(['Top Ranking Score:', max_ranking_score])
            if _SCORE_TABLE_DISPALY_AVG:
                # Add the average ranking score
                writer.writerow(['Average Ranking Score:', np.mean([score[2] for score in ranking_scores])])


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
    ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    conformer_max_iterations: int | None = None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """
    print(f'Processing fold input {fold_input.name}')

    if not fold_input.chains:
        raise ValueError('Fold input has no chains.')

    if os.path.exists(output_dir) and os.listdir(output_dir):
        new_output_dir = (
            f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        print(
            f'Output directory {output_dir} exists and non-empty, using instead '
            f' {new_output_dir}.'
        )
        output_dir = new_output_dir

    # if model_runner is not None:
    #     # If we're running inference, check we can load the model parameters before
    #     # (possibly) launching the data pipeline.
    #     print('Checking we can load the model parameters...')
    #     _ = model_runner.model_params
    
    if data_pipeline_config is None:
        print('Skipping data pipeline...')
    else:
        print('Running data pipeline...')
        fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

    print(f'Output directory: {output_dir}')
    print(f'Writing model input JSON to {output_dir}')
    write_fold_input_json(fold_input, output_dir)
    if model_runner is None:
        print('Skipping inference...')
        output = fold_input
    else:
        print(
            f'Predicting 3D structure for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
            conformer_max_iterations=conformer_max_iterations,
        )
        print(
            f'Writing outputs for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        write_outputs(
            all_inference_results=all_inference_results,
            output_dir=output_dir,
            job_name=fold_input.sanitised_name(),
        )
        output = all_inference_results

    print(f'Done processing fold input {fold_input.name}.')
    return output


def main(_):
    if _JSON_PATH.value is None == _INPUT_DIR.value is None:
        raise ValueError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )

    if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
        raise ValueError(
            'At least one of --run_inference or --run_data_pipeline must be'
            ' set to true.'
        )

    if _INPUT_DIR.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_dir(
            pathlib.Path(_INPUT_DIR.value)
        )
    elif _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    else:
        raise AssertionError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )
    pid = os.getpid()
    print(f"Main Thread PID: {pid}")
    # Make sure we can create the output directory before running anything.
    try:
        os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
        raise

    notice = textwrap.wrap(
        'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
        ' parameters are only available under terms of use provided at'
        ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
        ' If you do not agree to these terms and are using AlphaFold 3 derived'
        ' model parameters, cancel execution of AlphaFold 3 inference with'
        ' CTRL-C, and do not use the model parameters.',
        break_long_words=False,
        break_on_hyphens=False,
        width=80,
    )
    print('\n'.join(notice))

    if False:
       pass
    else:
        print('Skipping running the data pipeline.')
        data_pipeline_config = None

    if _RUN_INFERENCE.value:
        if _CPU_INFERENCE.value:
            device = torch.device('cpu')
        else:
            # 使用第二张显卡
            device = torch.device('cuda')
        
        print(f'Found local device: {device}')
        
        if _CPU_INFERENCE.value:
            # Set the number of threads to use for the data pipeline.
            torch.set_num_threads(_NUM_THREADS.value)
            print(f'Number of threads: {torch.get_num_threads()}')

        print('Building model from scratch...')
        model_runner = ModelRunner(
            model_dir=pathlib.Path(_MODEL_DIR.value),
            device=device,
        )
    else:
        print('Skipping running model inference.')
        model_runner = None

    print(f'Processing fold inputs.')
    num_fold_inputs = 0   
    for fold_input in fold_inputs:
        print(f'Processing fold input #{num_fold_inputs + 1}')
        process_fold_input(
            fold_input=fold_input,
            data_pipeline_config=data_pipeline_config,
            model_runner=model_runner,
            output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
            # buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
            conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
        )
        num_fold_inputs += 1

    print(f'Done processing {num_fold_inputs} fold inputs.')
    if _WOLRD_SIZE.value > 1:
        print("Destroying the process group")
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    flags.mark_flags_as_required([
        'output_dir',
    ])
    app.run(main)
