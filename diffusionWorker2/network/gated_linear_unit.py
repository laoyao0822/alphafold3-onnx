import torch

# import triton
from torchWorker.config import *
# if _CUDA_GLU_OPT:
#     import gated_linear_unit_cuda
# import triton.language as tl

def gated_linear_unit_torch(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out


def gated_linear_unit(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out
