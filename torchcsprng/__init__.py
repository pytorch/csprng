# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchcsprng._C import *


try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass
