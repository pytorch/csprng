# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from functools import lru_cache
from secrets import token_bytes
from torch.utils.cpp_extension import load
from torchcsprng._C import *
from typing import Optional, Union
from . import _modinfo


@lru_cache(maxsize=1)
def load_module():
    build_cuda = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"
    openmp = "ATen parallel backend: OpenMP" in torch.__config__.parallel_info()

    cflags = ["-Wall", "-Wextra", "-Wno-unused"]
    cuda_cflags = [
        "-std=c++14",
        f"--compiler-options={' '.join(cflags)!r}",
        "--expt-extended-lambda",
        "-Xcompiler",
    ]
    define_macros = []
    ldflags = []

    if openmp:
        ldflags += ["-fopenmp"]

    if (nvcc_flags := os.getenv("NVCC_FLAGS", "")) != "":
        cuda_cflags += nvcc_flags.split(" ")

    sources = _modinfo.SOURCES.copy()
    if build_cuda:
        sources += _modinfo.CUDA_SOURCES
        define_macros += ["WITH_CUDA"]

    cflags += [f"-D{macro}" for macro in define_macros]
    cuda_cflags += [f"-D{macro}" for macro in define_macros]

    _modinfo.BUILD_DIR.mkdir(exist_ok=True)

    return load(
        name="torchcsprng",
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        build_directory=_modinfo.BUILD_DIR,
        verbose=True,
    )


def create_stream_generator(
    key: Optional[torch.Tensor] = None, device: Union[torch.device, str] = "cpu"
) -> torch.Generator:
    module = load_module()
    if isinstance(device, str):
        device = torch.device(device)
    if key is None:
        key = torch.Tensor([b for b in token_bytes(12)])
        key = key.to(dtype=torch.uint8, device=device)
    return module.create_stream_generator_(key)


try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass
