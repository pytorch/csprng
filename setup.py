import sys
from setuptools import setup, Extension
from torch.utils import cpp_extension

if sys.platform == 'win32':
    vc_version = os.getenv('VCToolsVersion', '')
    if vc_version.startswith('14.16.'):
        CXX_FLAGS = ['/sdl']
    else:
        CXX_FLAGS = ['/sdl', '/permissive-']
else:
    CXX_FLAGS = ['-g']

build_cuda = cpp_extension.CUDA_HOME != None

if build_cuda:
    csprng_ext = cpp_extension.CUDAExtension(
        'torch_csprng', ['csprng.cu'],
        extra_compile_args={'cxx': CXX_FLAGS,
                            'nvcc': ['-O2', '--expt-extended-lambda', '-DAT_PARALLEL_OPENMP=1']}
    )
else:
    csprng_ext = cpp_extension.CppExtension(
        'torch_csprng', ['csprng.cpp'],
        extra_compile_args={'cxx': CXX_FLAGS + ['-DAT_PARALLEL_OPENMP=1']}
    )

setup(name='pytorch_csprng',
      ext_modules=[csprng_ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
