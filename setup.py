from setuptools import setup
from torch.utils import cpp_extension

build_cuda = cpp_extension.CUDA_HOME != None

CXX_FLAGS = ['-fopenmp']
NVCC_FLAGS = ['--expt-extended-lambda', '-Xcompiler', '-fopenmp']

if build_cuda:
    csprng_ext = cpp_extension.CUDAExtension(
        'torch_csprng', ['csprng.cu'],
        extra_compile_args={'cxx': [],
                            'nvcc': NVCC_FLAGS}
    )
else:
    csprng_ext = cpp_extension.CppExtension(
        'torch_csprng', ['csprng.cpp'],
        extra_compile_args={'cxx': CXX_FLAGS}
    )

setup(name='pytorch_csprng',
      ext_modules=[csprng_ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
