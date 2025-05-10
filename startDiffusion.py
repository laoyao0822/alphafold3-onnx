import datetime
import random
from collections.abc import Sequence

import time
from typing import overload
import torch.utils._pytree as pytree
import numpy as np
from absl import flags, app
import os
import pathlib
import torch.distributed as dist

import torch

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


from target_feat.TargetFeat import  TargetFeat
from torchfold3.config import *
from torchfold3.misc import feat_batch
from target_feat.misc import params as target_feat_params
from diffusionWorker2.diffusionOne import diffusion

from diffusionWorker2.misc import params as diffusion_params
from diffusionWorker2.diffusion_step_vino import diffusion_vino
from evoformer import preprocess
from confidenceWorker.confidence import ConfidenceOne
from confidenceWorker.misc import params as confidence_params



_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = _HOME_DIR / 'models/model_103275239_1'
OPENVINO_PATH = '/root/ASC25F/AF3/diffusion_head_openvino/model.xml'



_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
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

_CPU_INFERENCE = flags.DEFINE_bool(
    'cpu_inference',
    True,
    'Whether to run inference on the cpu.',
)

# control the number of threads used by the data pipeline.
_NUM_THREADS = flags.DEFINE_integer(
    'num_cpu_threads',
    48,
    'Number of threads to use for the data pipeline.',
)


_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
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
_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
)
_CONFIDENCE_DP = flags.DEFINE_bool(
    'confidence_dp',
    True,
    'Whether to run inference on the cpu.',
)
_USE_DIFFUSION_VINO = flags.DEFINE_bool(
    'use_diffusion_vino',
    False,
    'Whether to run inference on the fold inputs.',
)
_DIFFUSION_VINO_PATH = flags.DEFINE_string(
    'diffusion_vino_path',
    OPENVINO_PATH,
    'Path to the model to use for inference.',
)

_CPU_IPEX_OPT=True
_CPU_FLUSH_DENORM_OPT=True
_CPU_AMP_OPT=True


def setup(rank, world_size,master_addr='192.168.10.1', master_port='8082'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # 配置Gloo使用IB传输
    # os.environ['GLOO_DEVICE_TRANSPORT'] = 'ibverbs'  # 使用IB Verbs API
    # os.environ['GLOO_SOCKET_IFNAME'] = 'ib0'  # 指定InfiniBand网络接口


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
        if _WOLRD_SIZE.value > 1:
            setup(_RANK_.value, _WOLRD_SIZE.value)
        else:
            print('only support multi cpu')
            exit(0)

        self.target_feat = TargetFeat()
        self.target_feat.eval()
        target_feat_params.import_jax_weights_(self.target_feat, model_dir)

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


        print('loading the model parameters...')


        # self._model = self._model.to(device=self._device)

        # Apply IPEX optimization for CPU if device is CPU
        if _CPU_INFERENCE.value:
            import intel_extension_for_pytorch as ipex
            print("Applying Intel Extension for PyTorch optimizations...")
            self.target_feat = ipex.optimize(self.target_feat, weights_prepack=False, optimize_lstm=True,
                                             auto_kernel_selection=True, dtype=torch.bfloat16)
            if not _USE_DIFFUSION_VINO.value:
                    self.diffusion.pre_model = ipex.optimize(self.diffusion.pre_model,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                    self.diffusion.diffusion_head = ipex.optimize(self.diffusion.diffusion_head,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                # self.diffusion.diffusion_head=torch.compile(self.diffusion.diffusion_head,backend="ipex")
            self.confidence.confidence_head=ipex.optimize(self.confidence.confidence_head,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)

            if _CPU_FLUSH_DENORM_OPT:
                torch.set_flush_denormal(True)
            # self._model = torch.compile(self._model,backend="ipex")
        if _CPU_INFERENCE.value == False:
            exit(0)
            # torch.cuda.set_device(rank)
            # self._model = self._model.to(f"cuda:{rank}")
            # print("Applying CUDA optimizations...")
            # print(torch._dynamo.list_backends())
            # self._model = torch.compile(self._model,backend="inductor",dynamic=False)

    @torch.inference_mode()
    def run_inference(
            self, featurised_example: features.BatchDict
    ) :
        """Computes a forward pass of the model on a featurised example."""
        featurised_example = pytree.tree_map(
            torch.from_numpy, utils.remove_invalidly_typed_feats(
                featurised_example)
        )
        featurised_example = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(device=self._device),
            featurised_example,
        )
        featurised_example['deletion_mean'] = featurised_example['deletion_mean'].to(
            dtype=torch.float32)



        if _CPU_INFERENCE.value == False:
            if True:
                exit(0)


        else:  # CPU Inference
            if _CPU_AMP_OPT:
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                    print("Running inference with AMP on CPU...")
                    batch = feat_batch.Batch.from_data_dict(featurised_example)
                    time1 = time.time()
                    seq_mask = batch.token_features.mask.contiguous()
                    num_tokens = seq_mask.shape[0]
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

                    if (seq_mask == False).sum().item() != 0:
                        print('zero count of seq_mask is not zero,please cancel bucket:',
                              (seq_mask == False).sum().item() != 0)
                        exit(0)
                    rel_feat = preprocess.get_rel_feat(token_index=batch.token_features.token_index,
                                                       residue_index=batch.token_features.residue_index,
                                                       asym_id=batch.token_features.asym_id,
                                                       entity_id=batch.token_features.entity_id,
                                                       sym_id=batch.token_features.sym_id, dtype=target_feat.dtype)

                    attn_mask_seq = get_attn_mask(mask=seq_mask, dtype=torch.bfloat16, device='cpu', num_heads=16,
                                                  seq_len=num_tokens, batch_size=1).contiguous()
                    pair_mask = seq_mask[:, None] * seq_mask[None, :]

                    attn_mask_4 = get_attn_mask(mask=pair_mask, dtype=torch.bfloat16,
                                                device='cpu',
                                                batch_size=num_tokens,
                                                num_heads=4, seq_len=num_tokens).contiguous()
                    # print("zero count",(seq_mask == False).sum().item())

                    need_to_process=_NUM_DIFFUSION_SAMPLES.value -_RANK_.value

                    pair = torch.zeros([num_tokens, num_tokens, 128], device=target_feat.device,
                                       dtype=torch.bfloat16).contiguous()
                    single = torch.zeros(
                        [num_tokens,384 ], device=target_feat.device, dtype=torch.bfloat16
                    ).contiguous()
                    print("recv diffusion data")
                    dist.broadcast(tensor=single, src=0)
                    dist.broadcast(tensor=pair, src=0)

                    print('recv embeddings ok')
                    embeddings = {
                        'pair': pair,
                        'single': single,
                        'target_feat': target_feat,  # type: ignore
                    }
                    time1 = time.time()
                    if not _USE_DIFFUSION_VINO.value:
                        positions= self.diffusion.forward(batch, single=embeddings['single'],
                                                              pair=embeddings['pair'],
                                                              target_feat=target_feat, real_feat=rel_feat, index=need_to_process,
                                                              seq_mask=seq_mask,
                                                              )
                    else:
                        positions = self.diffusion.forward(featurised_example, single=embeddings['single'],
                                                              pair=embeddings['pair'],
                                                              target_feat=target_feat, real_feat=rel_feat,
                                                              seq_mask=seq_mask,
                                                              index=need_to_process
                                                              )
                    positions=positions.to(dtype=torch.float32).contiguous()
                    pos_wait=dist.isend(tensor=positions, dst=0)

                    if _CONFIDENCE_DP.value:
                        (predicted_lddt, predicted_experimentally_resolved, full_pde, average_pde,
                         full_pae, tmscore_adjusted_pae_global,
                         tmscore_adjusted_pae_interface) = self.confidence.forward(batch=batch,
                                                                                   embeddings=embeddings,
                                                                                   positions=positions,
                                                                                   attn_seq_mask=attn_mask_seq,
                                                                                   pair_mask=pair_mask,
                                                                                   attn_pair_mask=attn_mask_4)
                        packed_tensor = torch.cat([
                            predicted_lddt.flatten(),
                            predicted_experimentally_resolved.flatten().to(torch.float32),  # 统一数据类型
                            full_pde.flatten(),
                            average_pde.flatten(),
                            full_pae.flatten(),
                            tmscore_adjusted_pae_global.flatten(),
                            tmscore_adjusted_pae_interface.flatten()
                        ]).contiguous()


                    print("diffusion cost time: ", time.time() - time1)
                    pos_wait.wait()
                    print('pos send ok')
                    if _CONFIDENCE_DP.value:
                        dist.gather(packed_tensor, dst=0)
                        print('send packed_tensor')

            else:
                print('not suppose without amp')
                exit(0)



def predict_structure(
        fold_input: folding_input.Input,
        model_runner: ModelRunner,
        buckets: Sequence[int] | None = None,
        conformer_max_iterations: int | None = None,
):
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
        if _CPU_INFERENCE.value == False:
            torch.cuda.synchronize()
        inference_start_time = time.time()
        # set the random seed for the model.
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_runner.run_inference(example)


def process_fold_input(
        fold_input: folding_input.Input,
        model_runner: ModelRunner | None,
        buckets: Sequence[int] | None = None,
        conformer_max_iterations: int | None = None,
) :

    print(f'Processing fold input {fold_input.name}')

    if not fold_input.chains:
        raise ValueError('Fold input has no chains.')


    if model_runner is None:
        print('Skipping inference...')
    else:
        print(
            f'Predicting 3D structure for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
            conformer_max_iterations=conformer_max_iterations,
        )


def main(_):
    if _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    else:
        raise ValueError('No input JSON file provided.')
    if True:
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
    print(f'Processing fold inputs.')
    num_fold_inputs = 0
    for fold_input in fold_inputs:
        print(f'Processing fold input #{num_fold_inputs + 1}')
        process_fold_input(
            fold_input=fold_input,
            model_runner=model_runner,
            # buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
            conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
        )
        num_fold_inputs += 1
        dist.barrier()
    # dist.destroy_process_group()
if __name__ == '__main__':
    app.run(main)