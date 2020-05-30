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

setup(name='pytorch_csprng',
      ext_modules=[cpp_extension.CUDAExtension(
        'torch_csprng', [
            'csprng.cu'
        ],
        extra_compile_args={'cxx': CXX_FLAGS,
                            'nvcc': ['-O2', '--expt-extended-lambda', '-DAT_PARALLEL_NATIVE=1']})],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
