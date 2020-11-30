/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TORCH_CSPRNG_HOST __host__
#define TORCH_CSPRNG_HOST_DEVICE __host__ __device__
#define TORCH_CSPRNG_CONSTANT __constant__
#else
#define TORCH_CSPRNG_HOST
#define TORCH_CSPRNG_HOST_DEVICE
#define TORCH_CSPRNG_CONSTANT
#endif
