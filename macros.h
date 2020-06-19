#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TORCH_CSPRNG_HOST_DEVICE __host__ __device__
#define TORCH_CSPRNG_CONSTANT __constant__
#else
#define TORCH_CSPRNG_HOST_DEVICE
#define TORCH_CSPRNG_CONSTANT
#endif
