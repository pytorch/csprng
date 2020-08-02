import os
from sys import platform
import subprocess
from setuptools import setup, find_packages
from torch.utils import cpp_extension

build_cuda = cpp_extension.CUDA_HOME is not None or os.getenv('FORCE_CUDA', '0') == '1'

CXX_FLAGS = os.getenv('CXX_FLAGS', '')
if CXX_FLAGS == '':
    CXX_FLAGS = []
else:
    CXX_FLAGS = CXX_FLAGS.split(' ')
if platform == 'linux':
    CXX_FLAGS.append('-fopenmp')

NVCC_FLAGS = os.getenv('NVCC_FLAGS', '')
if NVCC_FLAGS == '':
    NVCC_FLAGS = []
else:
    NVCC_FLAGS = NVCC_FLAGS.split(' ')

for flag in ['--expt-extended-lambda', '-Xcompiler', '-fopenmp']:
    if not flag in NVCC_FLAGS:
        NVCC_FLAGS.append(flag)

module_name = 'torchcsprng'

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
package_name = 'torchcsprng'

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

pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

requirements = [
    pytorch_dep,
]

setup(
    # Metadata
    name=package_name,
    version=version,
    author="Pavel Belevich",
    author_email="pbelevich@fb.com",
    url="https://github.com/pytorch/csprng",
    description="Cryptographically secure pseudorandom number generators for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD-3',

    # Package info
    # packages=find_packages(exclude=('test',)), # doesn't work
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    ext_modules=[csprng_ext],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
