/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

template<typename input_index_calc_t>
TORCH_CSPRNG_HOST_DEVICE static void copy_input_to_block(int64_t idx, uint8_t* block, int block_size,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc) {
  for (auto i = 0; i < block_size / input_type_size; ++i) {
    const auto linear_index = idx * (block_size / input_type_size) + i;
    if (linear_index < input_numel) {
      std::memcpy(
          &(block[i * input_type_size]),
          &(reinterpret_cast<uint8_t*>(input_ptr)[input_index_calc(linear_index)]),
          input_type_size
      );
    }
  }
}

template<typename output_index_calc_t>
TORCH_CSPRNG_HOST_DEVICE static void copy_block_to_output(int64_t idx, uint8_t* block, int block_size,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc) {
  for (auto i = 0; i < block_size / output_type_size; ++i) {
    const auto linear_index = idx * block_size / output_type_size + i;
    if (linear_index < output_numel) {
      std::memcpy(
          &(reinterpret_cast<uint8_t*>(output_ptr)[output_index_calc(linear_index)]),
          &(block[i * output_type_size]),
          output_type_size
      );
    }
  }
}

template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t>
TORCH_CSPRNG_HOST_DEVICE static void block_cipher_kernel_helper_2(int64_t idx, cipher_t cipher, int block_size,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc) {
  uint8_t block[block_size];
  memset(&block, 0, block_size); // is it ok to use zeros as padding?
  copy_input_to_block(idx, block, block_size, input_ptr, input_numel, input_type_size, input_index_calc);
  cipher(idx, block);
  copy_block_to_output(idx, block, block_size, output_ptr, output_numel, output_type_size, output_index_calc);
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t>
__global__ static void block_cipher_kernel_cuda_2(cipher_t cipher, int block_size,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  block_cipher_kernel_helper_2(idx, cipher, block_size,
    input_ptr, input_numel, input_type_size, input_index_calc,
    output_ptr, output_numel, output_type_size, output_index_calc);
}
#endif

template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t>
static void block_cipher_kernel_cpu_serial_2(int64_t begin, int64_t end, cipher_t cipher, int block_size,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc) {
  for (auto idx = begin; idx < end; ++idx) {
    block_cipher_kernel_helper_2(idx, cipher, block_size,
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc);
  }
}

template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t>
static void block_cipher_kernel_cpu_2(int64_t total, cipher_t cipher, int block_size,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc) {
  if (total < at::internal::GRAIN_SIZE || at::get_num_threads() == 1) {
    block_cipher_kernel_cpu_serial_2(0, total, cipher, block_size,
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc);
  } else {
    at::parallel_for(0, total, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      block_cipher_kernel_cpu_serial_2(begin, end, cipher, block_size,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc);
    });
  }
}

template<typename cipher_t>
void block_cipher_2(Tensor input, Tensor output, cipher_t cipher, int block_size) {
  if (input.numel() == 0) {
    return;
  }

  const auto input_ptr = input.data_ptr();
  const auto input_numel = input.numel();
  const auto input_type_size = input.element_size();
  const auto input_iter = TensorIterator::nullary_op(input);
  const auto input_offset_calc = make_offset_calculator<1>(input_iter);
  const std::function<int(int)> input_index_calc_contiguous = [input_type_size] TORCH_CSPRNG_HOST_DEVICE (int li) -> int { // TODO: int or uint32_t?
    return li * input_type_size;
  };
  const std::function<int(int)> input_index_calc_non_contiguous = [input_offset_calc] TORCH_CSPRNG_HOST_DEVICE (int li) -> int {  // TODO: int or uint32_t?
    return input_offset_calc.get(li)[0];
  };
  const auto input_index_calc = input.is_contiguous() ? input_index_calc_contiguous : input_index_calc_non_contiguous;

  const auto output_ptr = output.data_ptr();
  const auto output_numel = output.numel();
  const auto output_type_size = output.element_size();
  const auto output_iter = TensorIterator::nullary_op(output);
  const auto output_offset_calc = make_offset_calculator<1>(output_iter);
  const std::function<int(int)> output_index_calc_contiguous = [output_type_size] TORCH_CSPRNG_HOST_DEVICE (int li) -> int { // TODO: int or uint32_t?
    return li * output_type_size;
  };
  const std::function<int(int)> output_index_calc_non_contiguous = [output_offset_calc] TORCH_CSPRNG_HOST_DEVICE (int li) -> int {  // TODO: int or uint32_t?
    return output_offset_calc.get(li)[0];
  };
  const auto output_index_calc = output.is_contiguous() ? output_index_calc_contiguous : output_index_calc_non_contiguous;

  TORCH_CHECK((input_numel * input_type_size + block_size - 1) / block_size * block_size == output_numel * output_type_size, "wrong size");

  const auto size_in_bytes = input_numel * input_type_size;

  if (input.device().type() == at::kCPU) {
    const auto total = (size_in_bytes + block_size - 1) / block_size;
    block_cipher_kernel_cpu_2(total, cipher, block_size,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc
    );
  } else if (input.device().type() == at::kCUDA) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    const auto block = 256;
    const auto grid = (size_in_bytes + (block * block_size) - 1) / (block * block_size);
    auto stream = at::cuda::getCurrentCUDAStream();
    block_cipher_kernel_cuda_2<<<grid, block, 0, stream>>>(cipher, block_size,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc
    );
    AT_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "torchcsprng was compiled without CUDA support");
#endif
  } else {
    TORCH_CHECK(false, "block_cipher supports only CPU and CUDA devices");
  }
}

}}
