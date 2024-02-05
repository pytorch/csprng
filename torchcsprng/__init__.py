# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchcsprng._C import *


try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass
