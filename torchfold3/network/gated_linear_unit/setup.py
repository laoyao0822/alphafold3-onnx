# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gated_linear_unit_cuda',
    ext_modules=[
        CUDAExtension('gated_linear_unit_cuda', [
            'glu_cuda_wrapper.cu',
            'glu_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)