from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nn_weighted_match',
    ext_modules=[
        CUDAExtension('nn_weighted_match', [
            './src/nn_weighted_match.cpp',
            './src/nn_weighted_match_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
