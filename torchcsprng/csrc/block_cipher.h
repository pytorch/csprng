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
TORCH_CSPRNG_HOST_DEVICE static void copy_block_to_output(int64_t idx, uint8_t* block, int block_size, int output_elem_per_block,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc) {
//  std::cout << "output_elem_per_block = " << output_elem_per_block << std::endl;
//  std::cout << "block_size = " << block_size << std::endl;
//  std::cout << "output_type_size = " << output_type_size << std::endl;
  for (auto i = 0; i < output_elem_per_block; ++i) {
    const auto linear_index = idx * output_elem_per_block + i;
    if (linear_index < output_numel) {
      std::memcpy(
          &(reinterpret_cast<uint8_t*>(output_ptr)[output_index_calc(linear_index)]),
          &(block[i * output_type_size]),
          output_type_size
      );
    }
  }
}

template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
TORCH_CSPRNG_HOST_DEVICE static void block_cipher_kernel_helper(
    int64_t idx, cipher_t cipher, int block_size, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform) {
  uint8_t block[block_size];
  std::memset(&block, 0, block_size); // is it ok to use zeros as padding?
  if (input_ptr != nullptr) {
    copy_input_to_block(idx, block, block_size, input_ptr, input_numel, input_type_size, input_index_calc);
  }
  cipher(idx, block);
  const auto new_block_size = transform(block);
  copy_block_to_output(idx, block, new_block_size, output_elem_per_block, output_ptr, output_numel, output_type_size, output_index_calc);
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
__global__ static void block_cipher_kernel_cuda(cipher_t cipher, int block_size, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  block_cipher_kernel_helper(idx, cipher, block_size, output_elem_per_block
    input_ptr, input_numel, input_type_size, input_index_calc,
    output_ptr, output_numel, output_type_size, output_index_calc,
    transform);
}
#endif

template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
static void block_cipher_kernel_cpu_serial(int64_t begin, int64_t end, cipher_t cipher, int block_size, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform) {
  for (auto idx = begin; idx < end; ++idx) {
    block_cipher_kernel_helper(idx, cipher, block_size, output_elem_per_block,
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc,
      transform);
  }
}

template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
static void block_cipher_kernel_cpu(int64_t total, cipher_t cipher, int block_size, int output_elem_per_block,
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    transform_t transform_func) {
  if (total < at::internal::GRAIN_SIZE || at::get_num_threads() == 1) {
    block_cipher_kernel_cpu_serial(0, total, cipher, block_size, output_elem_per_block,
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc,
      transform_func);
  } else {
    at::parallel_for(0, total, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      block_cipher_kernel_cpu_serial(begin, end, cipher, block_size, output_elem_per_block,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc,
        transform_func);
    });
  }
}

template<typename cipher_t, typename input_index_calc_t, typename output_index_calc_t, typename transform_t>
void block_cipher(
    void* input_ptr, int64_t input_numel, int input_type_size, input_index_calc_t input_index_calc,
    void* output_ptr, int64_t output_numel, int output_type_size, output_index_calc_t output_index_calc,
    Device device, cipher_t cipher, int block_size, int output_elem_per_block, transform_t transform_func) {
//  if (input.numel() == 0) {
//    return;
//  }
//  TORCH_CHECK((input_numel * input_type_size + block_size - 1) / block_size * block_size == output_numel * output_type_size, "wrong size");

//  const auto size_in_bytes = input_numel * input_type_size;
//  const auto size_in_bytes = output_numel * output_type_size;

  if (device.type() == at::kCPU) {
//    const auto total = (size_in_bytes + block_size - 1) / block_size;
//    const auto total = (size_in_bytes + block_size / N - 1) / block_size * N;
    const auto total = (output_numel + output_elem_per_block - 1) / output_elem_per_block;
    block_cipher_kernel_cpu(total,
        cipher, block_size, output_elem_per_block,
        input_ptr, input_numel, input_type_size, input_index_calc,
        output_ptr, output_numel, output_type_size, output_index_calc,
        transform_func
    );
  } else if (device.type() == at::kCUDA) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    const auto threads = 256;
    const auto grid = (output_numel + (threads * output_elem_per_block) - 1) / (threads * output_elem_per_block);
    auto stream = at::cuda::getCurrentCUDAStream();
    block_cipher_kernel_cuda<<<grid, threads, 0, stream>>>(
        cipher, block_size, output_elem_per_block
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

std::function<int(int)> create_index_calc(Tensor input) {
  if (input.is_contiguous()) {
    const auto input_type_size = input.element_size();
    return [input_type_size] TORCH_CSPRNG_HOST_DEVICE (uint32_t li) -> uint32_t {
      return li * input_type_size;
    };
  } else {
    const auto input_iter = TensorIterator::nullary_op(input);
    const auto input_offset_calc = make_offset_calculator<1>(input_iter);
    return [input_offset_calc] TORCH_CSPRNG_HOST_DEVICE (uint32_t li) -> uint32_t {
      return input_offset_calc.get(li)[0];
    };
  }
}

template<typename cipher_t>
void block_cipher(Tensor input, Tensor output,
                  cipher_t cipher, int block_size) {

  const auto input_ptr = input.data_ptr();
  const auto input_numel = input.numel();
  const auto input_type_size = input.element_size();
  const auto input_index_calc = create_index_calc(input);

  const auto output_ptr = output.data_ptr();
  const auto output_numel = output.numel();
  const auto output_type_size = output.element_size();
  const auto output_index_calc = create_index_calc(output);

  const auto device = output.device();

  block_cipher(
      input_ptr, input_numel, input_type_size, input_index_calc,
      output_ptr, output_numel, output_type_size, output_index_calc,
      device, cipher, block_size, block_size / output_type_size,
      [block_size] (auto x) { return block_size; });
}

}}
