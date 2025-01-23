from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # 根据你的 GPU 架构设置，A100 是 8.0
os.environ["MAX_JOBS"] = "4" 

setup(
    name='gated_linear_unit',
    ext_modules=[
        CUDAExtension(
            name='gated_linear_unit_cuda',
            sources=[
                'glu.cpp',
                'gated_linear_unit_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_80'],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)