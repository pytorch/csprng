#pragma once

#include "macros.h"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
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
namespace custom_prng {

// Generates `block_t_size`-bytes random key Tensor on CPU 
// using `generator`, which must be an instance of `at::CPUGeneratorImpl`
// and passes it to the `device`.
template<typename RNG>
at::Tensor _fill_random_key_tensor(Tensor& t, at::Generator generator) {
  auto gen = at::check_generator<RNG>(generator);
  if (gen->key().defined()) {
    return gen->key().clone();
  }

  TORCH_CHECK(t.device().type() == torch::kCPU);
  TORCH_CHECK(t.is_contiguous(), "key_tensor must be contiguous");
  const auto scalarType = t.scalar_type();
  TORCH_CHECK(isIntegralType(scalarType, /*includeBool=*/true), "key_tensor must be integral");
  const auto elem_size = elementSize(scalarType);
  const auto tensor_size = t.numel();

  if (elem_size <= 4) {
    using random_t = uint32_t;
    TORCH_CHECK(sizeof(random_t) == 4);
    random_t mask = static_cast<random_t>((static_cast<uint64_t>(1) << (8 * elem_size)) - static_cast<uint64_t>(1));
    const auto random_size = sizeof(random_t) / elem_size;

    std::lock_guard<std::mutex> lock(generator.mutex());
    auto gen = at::check_generator<RNG>(generator);

    for (size_t i = 0; i < (tensor_size + random_size - 1) / random_size; ++i) {
      random_t random = gen->random();
      for (size_t j = 0; j < random_size; ++j) {
        int k = i * random_size + j;
        if (k < tensor_size) {
          AT_DISPATCH_INTEGRAL_TYPES(scalarType, "key_tensor_assign", [&]() {
            t[k] = static_cast<scalar_t>((random >> (j * elem_size)) & mask);
          });
        }
      }
    }
  } else if (elem_size == 8) {
    using random_t = uint64_t;
    TORCH_CHECK(sizeof(random_t) == 8);
    random_t mask = 0xffffffffffffffffUL;
    const auto random_size = sizeof(random_t) / elem_size;

    std::lock_guard<std::mutex> lock(generator.mutex());
    auto gen = at::check_generator<RNG>(generator);

    for (size_t i = 0; i < (tensor_size + random_size - 1) / random_size; ++i) {
      random_t random = gen->random();
      for (size_t j = 0; j < random_size; ++j) {
        int k = i * random_size + j;
        if (k < tensor_size) {
          AT_DISPATCH_INTEGRAL_TYPES(scalarType, "key_tensor_assign", [&]() {
            t[k] = static_cast<scalar_t>((random >> (j * elem_size)) & mask);
          });
        }
      }
    }
  } else {
    TORCH_CHECK(false, "_fill_random_key_tensor does supports only integral dtypes less then or equal to 8 bytes");
  }
  return t;
}

template<typename RNG>
at::Tensor _random_key_tensor(size_t size, ScalarType scalar_type, at::Device device, at::Generator generator) {
  auto gen = at::check_generator<RNG>(generator);
  if (gen->key().defined()) {
    return gen->key().clone();
  }

  auto t = torch::empty({static_cast<signed long>(size)}, torch::TensorOptions(scalar_type).device(torch::kCPU));
  return _fill_random_key_tensor<RNG>(t, generator).to(device);
}

uint8_t* raw_uint8_t_pointer(const Tensor& t) {
  TORCH_CHECK(t.is_contiguous(), "key_tensor must be contiguous");
  const auto scalarType = t.scalar_type();
  TORCH_CHECK(isIntegralType(scalarType, /*includeBool=*/true), "key_tensor must be integral");
  return AT_DISPATCH_INTEGRAL_TYPES(scalarType, "raw", [&]() {
    return reinterpret_cast<uint8_t*>(t.data_ptr<scalar_t>());
  });
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
