from setuptools import setup
from torch.utils import cpp_extension

build_cuda = cpp_extension.CUDA_HOME != None

if build_cuda:
    csprng_ext = cpp_extension.CUDAExtension(
        'torch_csprng', ['csprng.cu'],
        extra_compile_args={'cxx': [],
                            'nvcc': ['-O2', '--expt-extended-lambda']}
    )
else:
    csprng_ext = cpp_extension.CppExtension(
        'torch_csprng', ['csprng.cpp']
    )

setup(name='pytorch_csprng',
      ext_modules=[csprng_ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
