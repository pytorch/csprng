#pragma once

#include "macros.h"
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include "OffsetCalculator.cuh"
#include <ATen/Parallel.h>
#include <cstdint>
#include <mutex>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define UNROLL_IF_CUDA #pragma unroll
#else
#define UNROLL_IF_CUDA
#endif

namespace torch {
namespace csprng {

// Generates `block_t_size`-bytes random key Tensor on CPU 
// using `generator`, which must be an instance of `at::CPUGeneratorImpl`
// and passes it to the `device`.
template<typename RNG>
at::Tensor key_tensor(size_t block_t_size, c10::optional<at::Generator> generator) {
  std::lock_guard<std::mutex> lock(generator->mutex());
  auto gen = at::check_generator<RNG>(generator);
  auto t = torch::empty({static_cast<signed long>(block_t_size)}, torch::kUInt8);
  using random_t = uint32_t;
  constexpr size_t random_t_size = sizeof(random_t);
  for (size_t i = 0; i < block_t_size / random_t_size; i++) {
    const auto rand = gen->random();
    for (size_t j = 0; j < random_t_size; j++) {
      size_t k = i * random_t_size + j;
      t[k] = static_cast<uint8_t>((rand >> (j * 8)) & 0xff);
    }
  }
  return t;
}

// A simple container for random state sub-blocks that implements RNG interface 
// with random() and random64() methods, that are used by transformation function
template<size_t size>
struct RNGValues {
  TORCH_CSPRNG_HOST_DEVICE RNGValues(uint64_t* vals) {
    memcpy(&vals_, vals, size * sizeof(uint64_t));
  }
  uint32_t TORCH_CSPRNG_HOST_DEVICE random() { auto res = static_cast<uint32_t>(vals_[index]); index++; return res; }
  uint64_t TORCH_CSPRNG_HOST_DEVICE random64() { auto res = vals_[index]; index++; return res; }
private:
  uint64_t vals_[size];
  int index = 0;
};

// Runs a block cipher in a counter mode in approximately `numel / (block_t_size / sizeof(uint_t) / N)` CUDA threads,
// without any assumption about target tensor layout. It uses `index_calc` to find memory locations of
// the tensor elements.
// `scalar_t`       is a scalar type equivalent of target tensor dtype
// `uint_t`         is an unsigned integral type of sub-blocks that random state is divided to
//                  (e.g, 16 bytes random state block can be divided into 16 uint8_t sub-blocks 
//                  or 8 uint16_t sub-block or 4 uint32_t sub-block or 2 uint64_t sub-blocks)
// `N`              is a number of sub-block which is used by `transform_func` 
//                  to generate a random value of specific distribution (e.g. `normal` uses 2)
// `numel`          is a number of elements in target tensor
// `block_t_size`   is a number of bytes in cipher's block (e.g. 16 for AES128)
// `cipher`         is a callable that receives a counter `idx` and returns an encrypted block
// `transform_func` is a callable that converts N `uint_t` random state sub-blocks passed in RNGValues into target dtype `scalar_t`
template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t, typename index_calc_t>
TORCH_CSPRNG_HOST_DEVICE static void block_cipher_kernel_helper(int idx, scalar_t* data, int64_t numel, size_t block_t_size, cipher_t cipher, transform_t transform_func, index_calc_t index_calc) {
  const int unroll_factor = block_t_size / sizeof(uint_t) / N;
  if (unroll_factor * idx < numel) {
    auto block = cipher(idx);
    UNROLL_IF_CUDA
    for (auto i = 0; i < unroll_factor; ++i) {
      const auto li = unroll_factor * idx + i;
      if (li < numel) {
        uint64_t vals[N];
        UNROLL_IF_CUDA
        for (size_t j = 0; j < N; j++) {
          vals[j] = (reinterpret_cast<uint_t*>(&block))[N * i + j];
        }
        RNGValues<N> rng(vals);
        data[index_calc(li)] = transform_func(&rng);
      }
    }
  }
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t, typename index_calc_t>
__global__ static void block_cipher_kernel_cuda(scalar_t* data, int64_t numel, int block_t_size, cipher_t cipher, transform_t transform_func, index_calc_t index_calc) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  block_cipher_kernel_helper<scalar_t, uint_t, N>(idx, data, numel, block_t_size, cipher, transform_func, index_calc);
}
#endif

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t, typename index_calc_t>
static void block_cipher_kernel_cpu_serial(int64_t begin, int64_t end, scalar_t* data, int64_t numel, int block_t_size, cipher_t cipher, transform_t transform_func, index_calc_t index_calc) {
  for (auto idx = begin; idx < end; ++idx) {
    block_cipher_kernel_helper<scalar_t, uint_t, N>(idx, data, numel, block_t_size, cipher, transform_func, index_calc);
  }
}

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t, typename index_calc_t>
static void block_cipher_kernel_cpu(int64_t total, scalar_t* data, int64_t numel, int block_t_size, cipher_t cipher, transform_t transform_func, index_calc_t index_calc) {
  if (total < at::internal::GRAIN_SIZE || at::get_num_threads() == 1) {
    block_cipher_kernel_cpu_serial<scalar_t, uint_t, N>(0, total, data, numel, block_t_size, cipher, transform_func, index_calc);
  } else {
    at::parallel_for(0, total, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      block_cipher_kernel_cpu_serial<scalar_t, uint_t, N>(begin, end, data, numel, block_t_size, cipher, transform_func, index_calc);
    });
  }
}

// Runs a block cipher in a counter mode in approximately `numel / (block_t_size / sizeof(uint_t) / N)` CUDA threads.
// Each CUDA thread generates `block_t_size`-bytes random state and divides it into `block_t_size / sizeof(uint_t)` sub-blocks.
// Then `transform_func` transforms `N` random state sub-blocks passed in a `RNGValues` to final random values of type `scalar_t`.
template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t>
void block_cipher_ctr_mode(at::TensorIterator& iter, int block_t_size, cipher_t cipher, transform_t transform_func) {
  const auto numel = iter.numel();
  if (numel == 0) {
    return;
  }
  const int unroll_factor = block_t_size / sizeof(uint_t) / N;
  const auto block = 256;
  const auto grid = (numel + (block * unroll_factor) - 1) / (block * unroll_factor);
  scalar_t* data = (scalar_t*)iter.data_ptr(0);
  auto offset_calc = make_offset_calculator<1>(iter);
  auto index_calc_identity = [] TORCH_CSPRNG_HOST_DEVICE (int li) -> int { return li; };
  auto index_calc_offset = [offset_calc] TORCH_CSPRNG_HOST_DEVICE (int li) -> int { return offset_calc.get(li)[0] / sizeof(scalar_t); };
  if (iter.device_type() == at::kCPU) {
    if (iter.output(0).is_contiguous()) {
      block_cipher_kernel_cpu<scalar_t, uint_t, N, cipher_t, transform_t>(
        grid * block, data, numel, block_t_size, cipher, transform_func, index_calc_identity);
    } else {
      block_cipher_kernel_cpu<scalar_t, uint_t, N, cipher_t, transform_t>(
        grid * block, data, numel, block_t_size, cipher, transform_func, index_calc_offset);
    }
  } else if (iter.device_type() == at::kCUDA) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    auto stream = at::cuda::getCurrentCUDAStream();
    if (iter.output(0).is_contiguous()) {
      block_cipher_kernel_cuda<scalar_t, uint_t, N, cipher_t, transform_t><<<grid, block, 0, stream>>>(
        data, numel, block_t_size, cipher, transform_func, index_calc_identity);
    } else {
      block_cipher_kernel_cuda<scalar_t, uint_t, N, cipher_t, transform_t><<<grid, block, 0, stream>>>(
        data, numel, block_t_size, cipher, transform_func, index_calc_offset);
    }
    AT_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "csprng was compiled without CUDA support");
#endif
  } else {
    TORCH_CHECK(false, "block_cipher_ctr_mode supports only CPU and CUDA devices");
  }
}

}}
