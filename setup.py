import os
import subprocess
from setuptools import setup
from torch.utils import cpp_extension

build_cuda = cpp_extension.CUDA_HOME != None

CXX_FLAGS = ['-fopenmp']
NVCC_FLAGS = ['--expt-extended-lambda', '-Xcompiler', '-fopenmp']

module_name = 'torch_csprng'

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, module_name, 'csrc')

if build_cuda:
    csprng_ext = cpp_extension.CUDAExtension(
        module_name, [os.path.join(extensions_dir, 'csprng.cu')],
        extra_compile_args={'cxx': [],
                            'nvcc': NVCC_FLAGS}
    )
else:
    csprng_ext = cpp_extension.CppExtension(
        module_name, [os.path.join(extensions_dir, 'csprng.cpp')],
        extra_compile_args={'cxx': CXX_FLAGS}
    )

version = open('version.txt', 'r').read().strip()
sha = 'Unknown'
package_name = 'pytorch_csprng'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=this_dir).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))

# Doesn't work yet :(
# version_path = os.path.join(this_dir, module_name, 'version.py')
# with open(version_path, 'w') as f:
#     f.write("__version__ = '{}'\n".format(version))
#     f.write("git_version = {}\n".format(repr(sha)))

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name=package_name,
    version=version,
    author="Pavel Belevich",
    author_email="pbelevich@fb.com",
    description="Cryptographically secure pseudorandom number generators for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD-3',
    url="https://github.com/pytorch/csprng",
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
    ext_modules=[csprng_ext],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
