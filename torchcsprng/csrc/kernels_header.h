/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/DistributionTemplates.h>
#include <torch/torch.h>
#include "macros.h"
#include "block_cipher.h"
#include "aes.h"

// Generates `block_t_size`-bytes random key Tensor on CPU
// using `generator`, which must be an instance of `at::CPUGeneratorImpl`
// and passes it to the `device`.
template<typename RNG>
at::Tensor key_tensor(size_t block_t_size, c10::optional<at::Generator> generator) {
  std::lock_guard<std::mutex> lock(generator->mutex());
  auto gen = at::check_generator<RNG>(generator);
  if (gen->key().defined()) {
    return gen->key().clone();
  }
  auto key = torch::empty({static_cast<signed long>(block_t_size)}, torch::kUInt8);
  using random_t = typename std::result_of<decltype(&RNG::random)(RNG)>::type;
  constexpr size_t random_t_size = sizeof(random_t);
  for (size_t i = 0; i < block_t_size / random_t_size; i++) {
    const auto rand = gen->random();
    for (size_t j = 0; j < random_t_size; j++) {
      size_t k = i * random_t_size + j;
      key[k] = static_cast<uint8_t>((rand >> (j * 8)) & 0xff);
    }
  }
  return key;
}

template<typename RNG>
at::Tensor aes128_key_tensor(at::Generator generator) {
  return key_tensor<RNG>(torch::csprng::aes::block_t_size, generator);
}

// ====================================================================================================================

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

// Applies AES in CTR mode with the `key` for passed TensorIterator iter.
// `scalar_t`       is a scalar type equivalent of target tensor dtype
// `uint_t`         is an unsigned integral type of sub-blocks that random state is divided to
//                  (e.g, 16 bytes random state block can be divided into 16 uint8_t sub-blocks
//                  or 8 uint16_t sub-block or 4 uint32_t sub-block or 2 uint64_t sub-blocks)
// `N`              is a number of sub-block which is used by `transform_func`
//                  to generate a random value of specific distribution (e.g. `normal` uses 2)
// `key`            is a CUDA pointer to random key memory block
// `transform_func` is a callable that converts N `uint_t` random state sub-blocks passed in RNGValues into target dtype `scalar_t`
template<typename scalar_t, typename uint_t, size_t N = 1, typename transform_t>
void aes_helper(at::TensorIterator& iter, const uint8_t* key_bytes, transform_t transform_func) {
  auto output = iter.tensor(0);
  const auto output_offset_calc = make_offset_calculator<1>(at::TensorIterator::nullary_op(output));
  const auto output_index_calc = [output_offset_calc] TORCH_CSPRNG_HOST_DEVICE (uint32_t li) -> uint32_t {
      return output_offset_calc.get(li)[0];
  };
  torch::csprng::block_cipher<torch::csprng::aes::block_t_size>(
      nullptr, 0, 0, output_index_calc,
      output.data_ptr(), output.numel(), output.element_size(), output_index_calc,
      iter.device_type(),
      [key_bytes] TORCH_CSPRNG_HOST_DEVICE (int64_t idx, uint8_t* block) -> void {
          uint8_t idx_block[torch::csprng::aes::block_t_size];
          std::memset(&idx_block, 0, torch::csprng::aes::block_t_size);
          *(reinterpret_cast<int64_t*>(idx_block)) = idx;
          torch::csprng::aes::encrypt(idx_block, key_bytes);
          for (size_t i = 0; i < torch::csprng::aes::block_t_size; i++) {
            block[i] ^= idx_block[i];
          }
      },
      torch::csprng::aes::block_t_size / (N * sizeof(uint_t)),
  [transform_func] TORCH_CSPRNG_HOST_DEVICE (uint8_t* block) {
    const auto n = torch::csprng::aes::block_t_size / (N * sizeof(uint_t));
    for (size_t i = 0; i < n; ++i) {
      uint64_t vals[N];
      for (size_t j = 0; j < N; ++j) {
        vals[j] = (reinterpret_cast<uint_t*>(block))[N * i + j];
      }
      RNGValues<N> rng(vals);
      reinterpret_cast<scalar_t*>(block)[i] = transform_func(&rng);
    }
  }
  );
}

// ====================================================================================================================

// A mapping between scalar type and corresponding unsigned integer type of random state sub-block.
// uint64_t for double and long, uint32_t for the rest
template <typename T>
struct UIntType {};

template <> struct UIntType<double> { using type = uint64_t; };
template <> struct UIntType<float> { using type = uint32_t; };
template <> struct UIntType<int64_t> { using type = uint64_t; };
template <> struct UIntType<int32_t> { using type = uint32_t; };
template <> struct UIntType<int16_t> { using type = uint32_t; };
template <> struct UIntType<int8_t> { using type = uint32_t; };
template <> struct UIntType<uint8_t> { using type = uint32_t; };
template <> struct UIntType<bool> { using type = uint32_t; };

// ==================================================== Random ========================================================

template<typename RNG>
struct RandomKernel {
  void operator()(at::TensorIterator& iter, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "random_kernel", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type>(iter, key,
                                                     [] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
        at::uniform_int_distribution<scalar_t> random;
        return random(generator);
      });
    });
  }
};

template<typename scalar_t, typename uint_t>
void random_from_to_kernel_helper(at::TensorIterator& iter, uint64_t range, int64_t base, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
                               [range, base] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
      at::uniform_int_from_to_distribution<scalar_t> random(range, base);
      return random(generator);
    });
}

template<typename scalar_t, typename uint_t>
void random_full_range_kernel_helper(at::TensorIterator& iter, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
                               [] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
      at::uniform_int_full_range_distribution<scalar_t> random;
      return random(generator);
    });
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(at::TensorIterator& iter, uint64_t range, int64_t base, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel", [&] {
      if ((
          std::is_same<scalar_t, int64_t>::value ||
          std::is_same<scalar_t, double>::value ||
          std::is_same<scalar_t, float>::value ||
          std::is_same<scalar_t, at::BFloat16>::value)/* TODO: && range >= 1ULL << 32*/)
      {
        random_from_to_kernel_helper<scalar_t, uint64_t>(iter, range, base, key);
      } else {
        random_from_to_kernel_helper<scalar_t, uint32_t>(iter, range, base, key);
      }
    });
  }
  void operator()(at::TensorIterator& iter, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel", [&] {
      if (std::is_same<scalar_t, int64_t>::value ||
          std::is_same<scalar_t, double>::value ||
          std::is_same<scalar_t, float>::value ||
          std::is_same<scalar_t, at::BFloat16>::value)
      {
        random_full_range_kernel_helper<scalar_t, uint64_t>(iter, key);
      } else {
        TORCH_CHECK(false, "random_full_64_bits_range_kernel_cuda handles only int64, double, float and bfloat16");
      }
    });
  }
};

// ==================================================== Uniform =======================================================

template<typename RNG>
struct UniformKernel {
  void operator()(at::TensorIterator& iter, double from, double to, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "uniform_kernel", [&] {
      aes_helper<scalar_t, uint64_t>(iter, key,
                                     [from, to] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
        at::uniform_real_distribution<double> uniform(from, to);
        return static_cast<scalar_t>(uniform(generator));
      });
    });
  }
};

// ==================================================== Normal ========================================================

template<typename RNG>
struct NormalKernel {
  void operator()(at::Tensor& self, double mean, double std, c10::optional<at::Generator> generator) {
    auto iter = at::TensorIterator::nullary_op(self);
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "normal_kernel", [&] {
      aes_helper<scalar_t, uint64_t, 2>(iter, key,
                                        [mean, std] TORCH_CSPRNG_HOST_DEVICE (RNGValues<2>* gen) -> scalar_t {
        at::normal_distribution<double> normal(mean, std);
        return static_cast<scalar_t>(normal(gen));
      });
    });
  }
};

// ==================================================== Cauchy ========================================================

template<typename RNG>
struct CauchyKernel {
  void operator()(at::TensorIterator& iter, double median, double sigma, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cauchy_kernel", [&] {
      aes_helper<scalar_t, uint64_t, 1>(iter, key,
                                        [median, sigma] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* gen) -> scalar_t {
        at::cauchy_distribution<double> cauchy(median, sigma);
        return static_cast<scalar_t>(cauchy(gen));
      });
    });
  }
};

// ================================================== LogNormal =======================================================

template<typename RNG>
struct LogNormalKernel {
  void operator()(at::TensorIterator& iter, double mean, double std, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_normal", [&] {
      aes_helper<scalar_t, uint64_t, 2>(iter, key,
                                        [mean, std] TORCH_CSPRNG_HOST_DEVICE (RNGValues<2>* gen) -> scalar_t {
        at::lognormal_distribution<double> logNormal(mean, std);
        return static_cast<scalar_t>(logNormal(gen));
      });
    });
  }
};

// ================================================== Geometric =======================================================

template<typename RNG>
struct GeometricKernel {
  void operator()(at::TensorIterator& iter, double p, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "geometric_kernel", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type, 1>(iter, key,
        [p] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* gen) -> scalar_t {
        at::geometric_distribution<scalar_t> geometric(p);
        return geometric(gen);
      });
    });
  }
};

// ================================================== Exponential =====================================================

template<typename RNG>
struct ExponentialKernel {
  void operator()(at::TensorIterator& iter, double lambda, c10::optional<at::Generator> generator) {
    const at::Tensor key_t = aes128_key_tensor<RNG>(*generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "exponential_kernel", [&] {
      aes_helper<scalar_t, uint64_t, 1>(iter, key,
        [lambda] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* gen) -> scalar_t {
        at::exponential_distribution<double> exponential(lambda);
        return static_cast<scalar_t>(exponential(gen));
      });
    });
  }
};
