# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from secrets import token_bytes
from torchcsprng._C import *
from typing import Optional, Union


def create_stream_generator(
    key: Optional[torch.Tensor] = None, device: Union[torch.device, str] = "cpu"
) -> torch.Generator:
    if isinstance(device, str):
        device = torch.device(device)
    if key is None:
        key = torch.Tensor([b for b in token_bytes(12)])
        key = key.to(dtype=torch.uint8, device=device)
    return create_stream_generator_(key)


try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass
