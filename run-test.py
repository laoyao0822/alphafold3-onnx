

import datetime
import random
from collections.abc import Sequence
import csv
import dataclasses
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
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

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.utils._pytree as pytree
import onnxruntime as ort

import openvino as ov
from torchfold3.alphafold3 import AlphaFold3
from torchfold3.misc.params import import_jax_weights_
from target_feat.TargetFeat import  TargetFeat
from torchfold3.config import *
from torchfold3.misc import feat_batch
from target_feat.misc import params as target_feat_params
from evoformer.evoformer import EvoFormerOne
from evoformer.misc import params as evoformer_params
import time

from diffusionWorker2.diffusionOne import DiffusionOne
from diffusionWorker2.diffusionOne import diffusion

from diffusionWorker2.misc import params as diffusion_params
DIFFUSION_ONNX=False
SAVE_ONNX=True
_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = _HOME_DIR / 'models/model_103275239_1'
DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'
ONNX_PATH = '/root/pycharm/diffusion_head_onnx_base2/diffusion_head.onnx'
# ONNX_PATH='/root/pycharm/diffusion_head_onnx_base_fp16/diffusion_head.onnx'
OPENVINO_PATH = '/root/pycharm/diffusion_head_openvino'

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

_CPU_INFERENCE = flags.DEFINE_bool(
    'cpu_inference',
    True,
    'Whether to run inference on the cpu.',
)

# control the number of threads used by the data pipeline.
_NUM_THREADS = flags.DEFINE_integer(
    'num_cpu_threads',
    59,
    'Number of threads to use for the data pipeline.',
)

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
    10,
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
import torch.distributed as dist
def setup(rank, world_size,init_method='tcp://127.0.0.1:8802'):
    if _CPU_INFERENCE.value:
        print("start to set up multi cpu","rank:",rank,"world_size:",world_size)
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
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
import onnx

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
        self._model = AlphaFold3(num_samples=_NUM_DIFFUSION_SAMPLES.value)
        self._model.eval()

        print('loading the model parameters...')
        import_jax_weights_(self._model, model_dir)

        #import target feat
        print('import target feat')
        self.target_feat=TargetFeat()
        self.target_feat.eval()
        target_feat_params.import_jax_weights_(self.target_feat,model_dir)
        print('import evoformer')
        self.evoformer=EvoFormerOne()
        self.evoformer.eval()
        evoformer_params.import_jax_weights_(self.evoformer,model_dir)

        # diffusion=onnx.load('/root/pycharm/diffusion_onnx5/diffusion.onnx',load_external_data=True)
        # onnx.checker.check_model('/root/pycharm/diffusion_onnx5/diffusion.onnx')
        # print("check success")
        # sess_options = ort.SessionOptions()
        # sess_options.enable_profiling = True
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # ort.set_default_logger_severity(1)
        # # sess_options.AddConfigEntry(ort.SessionOptions.kOrtSessionOptionEpContextEnable, "1");
        # session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options,provider_options=['OpenVINO_CPU'])
        # self.diffusion=session
        if not DIFFUSION_ONNX:
            self.diffusion=diffusion()
            self.diffusion.diffusion_head.eval()
        # diffusion_params.import_jax_weights_(self.diffusion,model_dir)
            self.diffusion.import_diffusion_head_params(model_dir)
        #
        # self._model = self._model.to(device=self._device)
        else:
            self.diffusion=diffusion()
            self.diffusion.initOnnxModel(ONNX_PATH)
            self.diffusion.import_diffusion_head_params(model_dir)

        # Apply IPEX optimization for CPU if device is CPU
        if _CPU_INFERENCE.value:
            print("mkl",torch.backends.mkl.is_available(),"onednn",torch.backends.mkldnn.is_available())
            if not SAVE_ONNX and not DIFFUSION_ONNX:
                import intel_extension_for_pytorch as ipex
                import openvino as ov
                print("Applying Intel Extension for PyTorch optimizations...")
                self._model = ipex.optimize(self._model,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)

                self.target_feat = ipex.optimize(self.target_feat,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                self.evoformer = ipex.optimize(self.evoformer,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                self.diffusion.diffusion_head = ipex.optimize(self.diffusion.diffusion_head,weights_prepack=False,optimize_lstm=True,auto_kernel_selection=True,dtype=torch.bfloat16)
                # opts = {"device": "CPU", "config": {"PERFORMANCE_HINT": "LATENCY"}, "model_caching" : True,"cache_dir": "./model_cache"}
                # self.diffusion.diffusion_head=torch.compile(self.diffusion.diffusion_head,backend="openvino",options=opts)
                # self.evoformer.evoformer=torch.compile(self.evoformer.evoformer, backend="openvino",dynamic=True)
                # self.diffusion.diffusion_head=torch.compile(self.diffusion.diffusion_head, backend="openvino",dynamic=False,options=opts)
                # self.evoformer
                # ov_model=ov.convert_model()
            if _CPU_FLUSH_DENORM_OPT:
                torch.set_flush_denormal(True)
                
            # self._model = torch.compile(self._model,backend="ipex")
            
        if _CPU_INFERENCE.value == False:
            torch.cuda.set_device(rank)
            self._model = self._model.to(f"cuda:{rank}")
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

        if _CPU_INFERENCE.value==False:
            if True:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    print("Running inference with AMP on GPU...")
                    result = self._model(featurised_example)
                    result['__identifier__'] = self._model.__identifier__.numpy()

        else: # CPU Inference
            if _CPU_AMP_OPT:
                # with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                    print("Running inference with AMP on CPU...")
                    # self._model=torch.jit.trace(self._model,featurised_example)
                    # result = self._model(featurised_example)
                    # exit(0)
                    batch = feat_batch.Batch.from_data_dict(featurised_example)
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
                    )
                    time1=time.time()
                    embeddings=self.evoformer(featurised_example,target_feat=target_feat)
                    print("Evoformer took %.2f seconds" % (time.time()-time1))
                    pred_dense_atom_mask = batch.predicted_structure_info.atom_mask

                    positions = torch.zeros((_NUM_DIFFUSION_SAMPLES.value,) + pred_dense_atom_mask.shape + (3,), device='cpu', dtype=torch.float32)

                    # self.diffusion.getOpenvinoModel(batch=featurised_example,
                    #                             single=embeddings['single'],pair=embeddings['pair'],target_feat=target_feat,
                    #                             save_path=OPENVINO_PATH)
                    if SAVE_ONNX:
                        self.diffusion.getOnnxModel(batch=featurised_example,
                                            single=embeddings['single'], pair=embeddings['pair'],
                                            target_feat=target_feat,
                                            save_path=ONNX_PATH)

                    for i in range(_NUM_DIFFUSION_SAMPLES.value):
                        # print("diffusion sample %d" % i)
                        time1 = time.time()
                        # outputs = self.diffusion.run(
                        #     output_names=output_names,
                        #     input_feed=inputs
                        # )

                        # 假设输出名为 "positions"，根据模型实际情况调整
                        # positions[i] = torch.from_numpy(outputs[0])
                        # positions[i]=self.diffusion(single=embeddings['single'],pair=embeddings['pair'],target_feat=target_feat,
                        #             seq_mask=batch.token_features.mask,
                        #             token_index=batch.token_features.token_index,
                        #             residue_index=batch.token_features.residue_index,
                        #             asym_id=batch.token_features.asym_id,
                        #             entity_id=batch.token_features.entity_id,
                        #             sym_id=batch.token_features.sym_id,
                        #
                        #             pred_dense_atom_mask=batch.predicted_structure_info.atom_mask,
                        #
                        #             acat_atoms_to_q_gather_idxs=batch.atom_cross_att.token_atoms_to_queries.gather_idxs,
                        #             acat_atoms_to_q_gather_mask=batch.atom_cross_att.token_atoms_to_queries.gather_mask,
                        #
                        #             acat_q_to_k_gather_idxs=batch.atom_cross_att.queries_to_keys.gather_idxs,
                        #             acat_q_to_k_gather_mask=batch.atom_cross_att.queries_to_keys.gather_mask,
                        #
                        #             acat_t_to_q_gather_idxs=batch.atom_cross_att.tokens_to_queries.gather_idxs,
                        #             acat_t_to_q_gather_mask=batch.atom_cross_att.tokens_to_queries.gather_mask,
                        #
                        #             acat_q_to_atom_gather_idxs=batch.atom_cross_att.queries_to_token_atoms.gather_idxs,
                        #             acat_q_to_atom_gather_mask=batch.atom_cross_att.queries_to_token_atoms.gather_mask,
                        #
                        #             acat_t_to_k_gather_idxs=batch.atom_cross_att.tokens_to_keys.gather_idxs,
                        #             acat_t_to_k_gather_mask=batch.atom_cross_att.tokens_to_keys.gather_mask,
                        #
                        #             ref_ops=batch.ref_structure.positions,
                        #             ref_mask=batch.ref_structure.mask,
                        #             ref_element=batch.ref_structure.element,
                        #             ref_charge=batch.ref_structure.charge,
                        #             ref_atom_name_chars=batch.ref_structure.atom_name_chars,
                        #             ref_space_uid=batch.ref_structure.ref_space_uid
                        #                             )

                        positions[i] = self.diffusion.forward(featurised_example,single=embeddings['single'], pair=embeddings['pair'],
                                                      target_feat=target_feat,USE_ONNX=DIFFUSION_ONNX
                                                      )
                        print("diffusion cost time: ", time.time() - time1)
                # with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                    result=self._model(featurised_example,embeddings,positions)

                    # if torch.allclose(result, target_feat, rtol=1e-5):
                    #     print("target_feat 张量没有变化")
                    # else:
                    #     print("target_feat 张量发生了变化")
                    # exit(0)
                    result['__identifier__'] = self._model.__identifier__.numpy()
            else:
                print("Running inference without AMP on CPU...")
                result = self._model(featurised_example)
                result['__identifier__'] = self._model.__identifier__.numpy()

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
