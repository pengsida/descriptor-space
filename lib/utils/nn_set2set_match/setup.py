from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nn_set2set_match',
    ext_modules=[
        CUDAExtension('nn_set2set_match', [
            './src/nn_set2set_match.cpp',
            './src/nn_set2set_match_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
