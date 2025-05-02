import torchWorker.alphafold3
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
import torch.utils._pytree as pytree
import numpy as np
from absl import flags, app
import os
import pathlib
import torchWorker.misc.params as params
import torch.distributed as dist
from torchWorker.alphafold3 import AlphaFold3
from torchWorker.misc.params import import_jax_weights_

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

from torchfold3.config import *

_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = _HOME_DIR / 'models/model_103275239_1'


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
    False,
    'Whether to run inference on the cpu.',
)

# control the number of threads used by the data pipeline.
_NUM_THREADS = flags.DEFINE_integer(
    'num_cpu_threads',
    20,
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
        self._model = torchWorker.alphafold3.AlphaFold3(num_samples=_NUM_DIFFUSION_SAMPLES.value)
        self._model.eval()
        print('loading the model parameters...')
        import_jax_weights_(self._model, model_dir)


        # self._model = self._model.to(device=self._device)

        # Apply IPEX optimization for CPU if device is CPU
        if _CPU_INFERENCE.value:
            if _CPU_IPEX_OPT:
                import intel_extension_for_pytorch as ipex
                print("Applying Intel Extension for PyTorch optimizations...")
                self._model = ipex.optimize(self._model, weights_prepack=False, optimize_lstm=True,
                                            auto_kernel_selection=True, dtype=torch.bfloat16)

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
            if _CUDA_AMP_OPT:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    print("Running inference with AMP on GPU...")
                    self._model(featurised_example)
            else:
                print("Running inference without AMP on GPU...")
                self._model(featurised_example)
        else:  # CPU Inference
            if _CPU_AMP_OPT:
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                    print("Running inference with AMP on CPU...")
                    self._model(featurised_example)
            else:
                print("Running inference without AMP on CPU...")
                self._model(featurised_example)



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