/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
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
          block + i * input_type_size,
          &(reinterpret_cast<uint8_t*>(input_ptr)[input_index_calc(linear_index)]),
          input_type_size
      );
    }
  }
}

template<typename output_index_calc_t>
TORCH_CSPRNG_HOST_DEVICE static void copy_block_to_output(int64_t idx, uint8_t* block, int output_elem_per_block,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc) {
  for (auto i = 0; i < output_elem_per_block; ++i) {
    const auto linear_index = idx * output_elem_per_block + i;
    if (linear_index < output_numel) {
      std::memcpy(
          &(reinterpret_cast<uint8_t*>(output_ptr)[output_index_calc(linear_index)]),
          block + i * output_type_size,
          output_type_size
      );
    }
  }
}

template<int block_size, typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
TORCH_CSPRNG_HOST_DEVICE static void block_cipher_kernel_helper(
    int64_t idx, cipher_t cipher, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform) {
  uint8_t block[block_size];
  std::memset(&block, 0, block_size); // is it ok to use zeros as padding?
  if (input_ptr != nullptr) {
    copy_input_to_block(idx, block, block_size, input_ptr, input_numel, input_type_size, input_index_calc);
  }
  cipher(idx, block);
  transform(block);
  copy_block_to_output(idx, block, output_elem_per_block, output_ptr, output_numel, output_type_size, output_index_calc);
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template<int block_size, typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
__global__ static void block_cipher_kernel_cuda(cipher_t cipher, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  block_cipher_kernel_helper<block_size>(idx, cipher, output_elem_per_block,
    input_ptr, input_numel, input_type_size, input_index_calc,
    output_ptr, output_numel, output_type_size, output_index_calc,
    transform);
}
#endif

template<int block_size, typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
static void block_cipher_kernel_cpu_serial(int64_t begin, int64_t end, cipher_t cipher, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform) {
  for (auto idx = begin; idx < end; ++idx) {
    block_cipher_kernel_helper<block_size>(idx, cipher, output_elem_per_block,
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc,
      transform);
  }
}

template<int block_size, typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
static void block_cipher_kernel_cpu(int64_t total, cipher_t cipher, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform_func) {
  if (total < at::internal::GRAIN_SIZE || at::get_num_threads() == 1) {
    block_cipher_kernel_cpu_serial<block_size>(0, total, cipher, output_elem_per_block,
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc,
      transform_func);
  } else {
    at::parallel_for(0, total, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      block_cipher_kernel_cpu_serial<block_size>(begin, end, cipher, output_elem_per_block,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc,
        transform_func);
    });
  }
}

template<int block_size, typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
void block_cipher(
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    at::Device device, cipher_t cipher, int output_elem_per_block, transform_t transform_func) {
  if (output_ptr == nullptr || output_numel == 0) {
    return;
  }

  if (device.type() == at::kCPU) {
    const auto total = (output_numel + output_elem_per_block - 1) / output_elem_per_block;
    block_cipher_kernel_cpu<block_size>(total,
        cipher, output_elem_per_block,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc,
        transform_func
    );
  } else if (device.type() == at::kCUDA) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    const auto threads = 256;
    const auto grid = (output_numel + (threads * output_elem_per_block) - 1) / (threads * output_elem_per_block);
    auto stream = at::cuda::getCurrentCUDAStream();
    block_cipher_kernel_cuda<block_size><<<grid, threads, 0, stream>>>(
        cipher, output_elem_per_block,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc,
        transform_func
    );
    AT_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "torchcsprng was compiled without CUDA support");
#endif
  } else {
    TORCH_CHECK(false, "block_cipher supports only CPU and CUDA devices");
  }
}

template<int block_size, typename cipher_t>
void block_cipher(at::Tensor input, at::Tensor output, cipher_t cipher) {
  const auto input_ptr = input.data_ptr();
  const auto input_numel = input.numel();

  // Otherwise OffsetCalculator/IntDivider crashes with integer division by zero
  if (input_ptr == nullptr || input_numel == 0) {
    return;
  }

  const auto input_type_size = input.element_size();
  const auto input_offset_calc = make_offset_calculator<1>(at::TensorIterator::nullary_op(input));
  const auto input_index_calc = [input_offset_calc] TORCH_CSPRNG_HOST_DEVICE (uint32_t li) -> uint32_t {
    return input_offset_calc.get(li)[0];
  };

  const auto output_ptr = output.data_ptr();
  const auto output_numel = output.numel();

  // Otherwise OffsetCalculator/IntDivider crashes with integer division by zero
  if (output_ptr == nullptr || output_numel == 0) {
    return;
  }

  const auto output_type_size = output.element_size();
  const auto output_offset_calc = make_offset_calculator<1>(at::TensorIterator::nullary_op(output));
  const auto output_index_calc = [output_offset_calc] TORCH_CSPRNG_HOST_DEVICE (uint32_t li) -> uint32_t {
    return output_offset_calc.get(li)[0];
  };

  const auto device = output.device();

  torch::csprng::block_cipher<block_size>(
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc,
      device, cipher, block_size / output_type_size,
      [] TORCH_CSPRNG_HOST_DEVICE (uint8_t* x) {});
}

}}
