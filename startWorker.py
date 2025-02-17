import torchWorker.alphafold3
from absl import flags, app
import os
import pathlib
import torchWorker.misc.params as params
import torch.distributed as dist
import torch
_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = _HOME_DIR / 'models/model_103275239_1'


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

def setup(rank, world_size):
    print("start to set up multi gpu","rank:",rank,"world_size:",world_size)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:8802',
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

def main(argv):
    # 这里可以安全地访问标志
    rank=_RANK_.value
    torch.cuda.set_device(rank)
    with torch.no_grad():
        worker_model = torchWorker.alphafold3.AlphaFold3(num_samples=5).to(f"cuda:{rank}").eval()
        params.import_jax_weights_(worker_model, pathlib.Path(_MODEL_DIR.value))
        setup(_RANK_.value, _WOLRD_SIZE.value)

    print("worker_model import params done")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            worker_model()
    dist.destroy_process_group()
if __name__ == '__main__':
    app.run(main)