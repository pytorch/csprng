import os
import sys
import subprocess
from setuptools import setup, find_packages
import distutils.command.clean
import glob
import shutil

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, \
    CUDAExtension, CUDA_HOME

version = open('version.txt', 'r').read().strip()
sha = 'Unknown'
package_name = 'torchcsprng'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode(
        'ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'torchcsprng', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))
        # f.write("from torchcsprng.extension import _check_cuda_version\n")
        # f.write("if _check_cuda_version() > 0:\n")
        # f.write("    cuda = _check_cuda_version()\n")


write_version_file()

with open("README.md", "r") as fh:
    long_description = fh.read()

pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

requirements = [
    pytorch_dep,
]


def append_flags(flags, flags_to_append):
    for flag in flags_to_append:
        if not flag in flags:
            flags.append(flag)
    return flags


def get_extensions():
    build_cuda = torch.cuda.is_available() or os.getenv('FORCE_CUDA',
                                                        '0') == '1'

    module_name = 'torchcsprng'

    extensions_dir = os.path.join(cwd, module_name, 'csrc')

    openmp = 'ATen parallel backend: OpenMP' in torch.__config__.parallel_info()

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    cxx_flags = os.getenv('CXX_FLAGS', '')
    if cxx_flags == '':
        cxx_flags = []
    else:
        cxx_flags = cxx_flags.split(' ')
    if openmp:
        if sys.platform == 'linux':
            cxx_flags = append_flags(cxx_flags, ['-fopenmp'])
        elif sys.platform == 'win32':
            cxx_flags = append_flags(cxx_flags, ['/openmp'])
        # elif sys.platform == 'darwin':
        #     cxx_flags = append_flags(cxx_flags, ['-Xpreprocessor', '-fopenmp'])

    if build_cuda:
        extension = CUDAExtension
        source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))
        sources += source_cuda

        define_macros += [('WITH_CUDA', None)]

        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        nvcc_flags = append_flags(nvcc_flags, ['--expt-extended-lambda', '-Xcompiler'])
        extra_compile_args = {
            'cxx': cxx_flags,
            'nvcc': nvcc_flags,
        }
    else:
        extra_compile_args = {
            'cxx': cxx_flags,
        }

    ext_modules = [
        extension(
            module_name + '._C',
            sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            start_deleting = False
            for wildcard in filter(None, ignores.split('\n')):
                if wildcard == '# do not change or delete this comment - `python setup.py clean` deletes everything after this line':
                    start_deleting = True
                if not start_deleting:
                    continue
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


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
    packages=find_packages(exclude=('test',)),
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
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    ext_modules=get_extensions(),
    test_suite='test',
    cmdclass={
        'build_ext': BuildExtension,
        'clean': clean,
    }
)
