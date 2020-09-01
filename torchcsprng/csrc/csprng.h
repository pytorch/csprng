/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/DistributionsHelper.h>
#include <memory>
#include <random>
#include "macros.h"
#include "block_cipher.h"
#include "aes.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#endif

using namespace at;
using namespace at::native::templates;
using namespace torch::csprng;

inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

// CUDA CSPRNG is actually CPU generator which is used only to generate a random key on CPU for AES running in a block mode on CUDA 
struct CSPRNGGeneratorImpl : public c10::GeneratorImpl {
  CSPRNGGeneratorImpl(bool use_rd)              : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, use_rd_{use_rd} {}
  CSPRNGGeneratorImpl(const std::string& token) : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, use_rd_{true}, rd_{token} {}
  CSPRNGGeneratorImpl(uint64_t seed)            : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, use_rd_{false}, mt_{static_cast<unsigned int>(seed)} { }
  ~CSPRNGGeneratorImpl() = default;
  uint32_t random() { return use_rd_ ? rd_() : mt_(); }
  uint64_t random64() { return use_rd_ ? make64BitsFrom32Bits(rd_(), rd_()) : make64BitsFrom32Bits(mt_(), mt_()); }

  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  CSPRNGGeneratorImpl* clone_impl() const override { throw std::runtime_error("not implemented"); }

  static DeviceType device_type() { return DeviceType::CPU; }

  bool use_rd_;
  std::random_device rd_;
  std::mt19937 mt_;
};

// ====================================================================================================================

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
void aes_helper(TensorIterator& iter, const uint8_t* key, transform_t transform_func) {
  block_cipher_ctr_mode<scalar_t, uint_t, N>(iter, aes::block_t_size,
    [key] TORCH_CSPRNG_HOST_DEVICE (unsigned int idx) -> aes::block_t {
      aes::block_t block;
      memset(&block, 0, aes::block_t_size);
      block.x = idx;
      aes::encrypt(reinterpret_cast<uint8_t*>(&block), key);
      return block;
    },
    transform_func
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
  void operator()(TensorIterator& iter, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, iter.dtype(), "random_kernel", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type>(iter, key,
        [] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
          uniform_int_distribution<scalar_t> random;
          return random(generator);
        }
      );
    });
  }
};

template<typename scalar_t, typename uint_t>
void random_from_to_kernel_helper(TensorIterator& iter, uint64_t range, int64_t base, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [range, base] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
      uniform_int_from_to_distribution<scalar_t> random(range, base);
      return random(generator);
    }
  );
}

template<typename scalar_t, typename uint_t>
void random_full_range_kernel_helper(TensorIterator& iter, const uint8_t* key) {
  aes_helper<scalar_t, uint_t>(iter, key,
    [] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
      uniform_int_full_range_distribution<scalar_t> random;
      return random(generator);
    }
  );
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIterator& iter, uint64_t range, int64_t base, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel", [&] {
      if ((
        std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32)
      {
        random_from_to_kernel_helper<scalar_t, uint64_t>(iter, range, base, key);
      } else {
        random_from_to_kernel_helper<scalar_t, uint32_t>(iter, range, base, key);
      }
    });
  }
  void operator()(TensorIterator& iter, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
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

Tensor& random_(Tensor& self, c10::optional<Generator> generator) {
  return random_impl<RandomKernel, CSPRNGGeneratorImpl>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, c10::optional<Generator> generator) {
  return random_from_to_impl<RandomFromToKernel, CSPRNGGeneratorImpl>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, c10::optional<Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

// ==================================================== Uniform =======================================================

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIterator& iter, double from, double to, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "uniform_kernel", [&] {
      aes_helper<scalar_t, uint64_t>(iter, key,
        [from, to] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* generator) -> scalar_t {
          uniform_real_distribution<double> uniform(from, to);
          return static_cast<scalar_t>(uniform(generator));
        }
      );
    });
  }
};

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> generator) {
  return uniform_impl_<UniformKernel, CSPRNGGeneratorImpl>(self, from, to, generator);
}

// ==================================================== Normal ========================================================

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, c10::optional<Generator> generator) {
    auto iter = TensorIterator::nullary_op(self);
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "normal_kernel", [&] {
      aes_helper<scalar_t, uint64_t, 2>(iter, key,
        [mean, std] TORCH_CSPRNG_HOST_DEVICE (RNGValues<2>* gen) -> scalar_t {
          normal_distribution<double> normal(mean, std);
          return static_cast<scalar_t>(normal(gen));
        }
      );
    });
  }
};

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> generator) {
  return normal_impl_<NormalKernel, CSPRNGGeneratorImpl>(self, mean, std, generator);
}

Tensor& normal_Tensor_float_out(Tensor& output, const Tensor& mean, double std, c10::optional<Generator> gen) {
  return normal_out_impl<NormalKernel, CSPRNGGeneratorImpl>(output, mean, std, gen);
}

Tensor& normal_float_Tensor_out(Tensor& output, double mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_out_impl<NormalKernel, CSPRNGGeneratorImpl>(output, mean, std, gen);
}

Tensor& normal_Tensor_Tensor_out(Tensor& output, const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_out_impl<NormalKernel, CSPRNGGeneratorImpl>(output, mean, std, gen);
}

Tensor normal_Tensor_float(const Tensor& mean, double std, c10::optional<Generator> gen) {
  return normal_impl<NormalKernel, CSPRNGGeneratorImpl>(mean, std, gen);
}

Tensor normal_float_Tensor(double mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_impl<NormalKernel, CSPRNGGeneratorImpl>(mean, std, gen);
}

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  return normal_impl<NormalKernel, CSPRNGGeneratorImpl>(mean, std, gen);
}

// ==================================================== Cauchy ========================================================

template<typename RNG>
struct CauchyKernel {
  void operator()(TensorIterator& iter, double median, double sigma, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cauchy_kernel", [&] {
      aes_helper<scalar_t, uint64_t, 1>(iter, key,
        [median, sigma] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* gen) -> scalar_t {
          cauchy_distribution<double> cauchy(median, sigma);
          return static_cast<scalar_t>(cauchy(gen));
        }
      );
    });
  }
};

Tensor& cauchy_(Tensor& self, double median, double sigma, c10::optional<Generator> generator) {
  return cauchy_impl_<CauchyKernel, CSPRNGGeneratorImpl>(self, median, sigma, generator);
}

// ================================================== LogNormal =======================================================

template<typename RNG>
struct LogNormalKernel {
  void operator()(TensorIterator& iter, double mean, double std, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_normal", [&] {
      aes_helper<scalar_t, uint64_t, 2>(iter, key,
        [mean, std] TORCH_CSPRNG_HOST_DEVICE (RNGValues<2>* gen) -> scalar_t {
          lognormal_distribution<double> logNormal(mean, std);
          return static_cast<scalar_t>(logNormal(gen));
        }
      );
    });
  }
};

Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return log_normal_impl_<LogNormalKernel, CSPRNGGeneratorImpl>(self, mean, std, gen);
}

// ================================================== Geometric =======================================================

template<typename RNG>
struct GeometricKernel {
  void operator()(TensorIterator& iter, double p, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "geometric_kernel", [&] {
      aes_helper<scalar_t, UIntType<scalar_t>::type, 1>(iter, key,
        [p] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* gen) -> scalar_t {
          geometric_distribution<scalar_t> geometric(p);
          return geometric(gen);
        }
      );
    });
  }
};

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen) {
  return geometric_impl_<GeometricKernel, CSPRNGGeneratorImpl>(self, p, gen);
}

// ================================================== Exponential =====================================================

template<typename RNG>
struct ExponentialKernel {
  void operator()(TensorIterator& iter, double lambda, c10::optional<Generator> generator) {
    const Tensor key_t = key_tensor<RNG>(aes::block_t_size, generator).to(iter.device());
    const auto key = key_t.data_ptr<uint8_t>();
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "exponential_kernel", [&] {
      aes_helper<scalar_t, uint64_t, 1>(iter, key,
        [lambda] TORCH_CSPRNG_HOST_DEVICE (RNGValues<1>* gen) -> scalar_t {
          exponential_distribution<double> exponential(lambda);
          return static_cast<scalar_t>(exponential(gen));
        }
      );
    });
  }
};

Tensor& exponential_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  return exponential_impl_<ExponentialKernel, CSPRNGGeneratorImpl>(self, lambda, gen);
}

// =============================================== Random permutation =================================================

// randperm implementation was copied from PyTorch to unblock CSPRNG users, but ultimately CSPRNG must reuse
// refactored randperm from PyTorch, see https://github.com/pytorch/pytorch/issues/43816

namespace {

  inline void check_supported_max_int_with_precision(int64_t n, const Tensor& tensor) {
    TORCH_CHECK(at::scalar_tensor(n, tensor.options()).defined(),
                "n is too large for result tensor type: '", tensor.toString(), "'");

    // Ensure sufficient precision for floating point representation.
    switch (tensor.scalar_type()) {
      case at::ScalarType::Half:
        TORCH_CHECK(n <= (int64_t(1) << 11) + 1, "n cannot be greater than 2049 for Half type.");
        break;
      case at::ScalarType::Float:
        TORCH_CHECK(n <= (int64_t(1) << 24) + 1, "n cannot be greater than 2^24+1 for Float type.");
        break;
      case at::ScalarType::Double:  // Unlikely to happen, but doesn't hurt to check
        TORCH_CHECK(n <= (int64_t(1) << 53) + 1, "n cannot be greater than 2^53+1 for Double type.");
        break;
      default:
        break;
    }
  }

  template <typename scalar_t, typename RNG>
  void randperm(Tensor& result, int64_t n, c10::optional<at::Generator> generator) {
    auto gen = at::check_generator<RNG>(generator);
    scalar_t *r__data = result.data_ptr<scalar_t>();

    result.resize_({n});
    int64_t r__stride_0 = result.stride(0);

    at::parallel_for(0, n, internal::GRAIN_SIZE,
                     [&r__data, &r__stride_0](int64_t p_begin, int64_t p_end) {
                       for(int64_t i = p_begin; i < p_end; i++)
                         r__data[i*r__stride_0] = static_cast<scalar_t>(i);
                     });

    for(int64_t i = 0; i < n - 1; i++)
    {
      int64_t z = gen->random() % (n-i);
      scalar_t sav = r__data[i*r__stride_0];
      r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
      r__data[(z+i)*r__stride_0] = sav;
    }
  }
} // namespace

Tensor& randperm_generator_out(Tensor& result, int64_t n, c10::optional<Generator> generator) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  check_supported_max_int_with_precision(n, result);
  if (result.device().type() == at::kCUDA) {
    auto result_cpu = at::empty({n}, result.options().device(kCPU));
    randperm_generator_out(result_cpu, n, generator);
    result.resize_({n});
    return result.copy_(result_cpu);
  }
  result.resize_({n});
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(generator->mutex());
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, result.scalar_type(), "randperm", [&]() -> void {
    randperm<scalar_t, CSPRNGGeneratorImpl>(result, n, generator);
  });
  return result;
}

// ====================================================================================================================

Generator create_random_device_generator(c10::optional<std::string> token = c10::nullopt) {
  if (token.has_value()) {
    return make_generator<CSPRNGGeneratorImpl>(*token);
  } else {
    return make_generator<CSPRNGGeneratorImpl>(true);
  }
}

Generator create_mt19937_generator(c10::optional<uint64_t> seed = c10::nullopt) {
  if (seed.has_value()) {
    return make_generator<CSPRNGGeneratorImpl>(*seed);
  } else {
    return make_generator<CSPRNGGeneratorImpl>(false);
  }
}

bool supports_cuda() {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return true;
#else
  return false;
#endif
}

TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
  // Random
  m.impl_UNBOXED("random_.from",             random_from_to);
  m.impl_UNBOXED("random_.to",               random_to);
  m.impl_UNBOXED("random_",                  random_);
  // Uniform
  m.impl_UNBOXED("uniform_",                 uniform_);
  // Normal
  m.impl_UNBOXED("normal_",                  normal_);
  m.impl_UNBOXED("normal.Tensor_float_out",  normal_Tensor_float_out);
  m.impl_UNBOXED("normal.float_Tensor_out",  normal_float_Tensor_out);
  m.impl_UNBOXED("normal.Tensor_Tensor_out", normal_Tensor_Tensor_out);
  m.impl_UNBOXED("normal.Tensor_float",      normal_Tensor_float);
  m.impl_UNBOXED("normal.float_Tensor",      normal_float_Tensor);
  m.impl_UNBOXED("normal.Tensor_Tensor",     normal_Tensor_Tensor);
  // Cauchy
  m.impl_UNBOXED("cauchy_",                  cauchy_);
  // LogNormal
  m.impl_UNBOXED("log_normal_",              log_normal_);
  // Geometric
  m.impl_UNBOXED("geometric_",               geometric_);
  // Exponential
  m.impl_UNBOXED("exponential_",             exponential_);
  // Random permutation
  m.impl_UNBOXED("randperm.generator_out",   randperm_generator_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("supports_cuda", &supports_cuda);
  m.def("create_random_device_generator", &create_random_device_generator, py::arg("token") = nullptr);
  m.def("create_mt19937_generator", &create_mt19937_generator, py::arg("seed") = nullptr);
}
