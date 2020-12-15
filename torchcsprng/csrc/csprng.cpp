/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/core/op_registration/op_registration.h>

#include "kernels_commons.h"
#include "cpu/kernels.h"
#ifdef WITH_CUDA
#include "cuda/kernels.cuh"
#endif

using namespace at;
using namespace torch::csprng;

static const auto GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE = "generator does not support tensor device type";
static const auto TENSOR_DEVICE_TYPE_IS_NOT_SUPPORTED = "tensor device type is not supported";

// ==================================================== Random ========================================================

Tensor& random_(Tensor& self, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::random_(self, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::random_(self, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to,
                       c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::random_from_to(self, from, to, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::random_from_to(self, from, to, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor& random_to(Tensor& self, int64_t to,
                  c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::random_to(self, to, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::random_to(self, to, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

// ==================================================== Uniform =======================================================

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::uniform_(self, from, to, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::uniform_(self, from, to, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

// ==================================================== Normal ========================================================

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::normal_(self, mean, std, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::normal_(self, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor& normal_Tensor_float_out(const Tensor& mean, double std, c10::optional<Generator> gen, Tensor& output) {
  if (output.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_float_out(output, mean, std, gen);
#ifdef WITH_CUDA
  } else if (output.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::normal_Tensor_float_out(output, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor& normal_float_Tensor_out(double mean, const Tensor& std, c10::optional<Generator> gen, Tensor& output) {
  if (output.device().type() == DeviceType::CPU) {
    return cpu::normal_float_Tensor_out(output, mean, std, gen);
#ifdef WITH_CUDA
  } else if (output.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::normal_float_Tensor_out(output, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor& normal_Tensor_Tensor_out(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen, Tensor& output) {
  if (output.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_Tensor_out(output, mean, std, gen);
#ifdef WITH_CUDA
  } else if (output.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::normal_Tensor_Tensor_out(output, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor normal_Tensor_float(const Tensor& mean, double std, c10::optional<Generator> gen) {
  if (mean.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_float(mean, std, gen);
#ifdef WITH_CUDA
  } else if (mean.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::normal_Tensor_float(mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor normal_float_Tensor(double mean, const Tensor& std, c10::optional<Generator> gen) {
  if (std.device().type() == DeviceType::CPU) {
    return cpu::normal_float_Tensor(mean, std, gen);
#ifdef WITH_CUDA
  } else if (std.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::normal_float_Tensor(mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  if (mean.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_Tensor(mean, std, gen);
#ifdef WITH_CUDA
  } else if (mean.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::normal_Tensor_Tensor(mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

// ==================================================== Cauchy ========================================================

Tensor& cauchy_(Tensor& self, double median, double sigma, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::cauchy_(self, median, sigma, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::cauchy_(self, median, sigma, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

// ================================================== LogNormal =======================================================

Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::log_normal_(self, mean, std, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::log_normal_(self, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

// ================================================== Geometric =======================================================

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::geometric_(self, p, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::geometric_(self, p, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
}

// ================================================== Exponential =====================================================

Tensor& exponential_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::exponential_(self, lambda, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::exponential_(self, lambda, gen);
#endif
  } else {
    TORCH_CHECK(false, GENERATOR_DOES_NOT_SUPPORT_TENSOR_DEVICE_TYPE);
  }
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

Tensor& randperm_generator_out(int64_t n, c10::optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  check_supported_max_int_with_precision(n, result);
  if (result.device().type() == at::kCUDA) {
    auto result_cpu = at::empty({n}, result.options().device(kCPU));
    randperm_generator_out(n, generator, result_cpu);
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

// ================================================Encrypt/Decrypt=====================================================

Tensor encrypt_pybind(Tensor input, Tensor output, Tensor key, const std::string& cipher, const std::string& mode) {
  if (input.device().type() == DeviceType::CPU) {
    return cpu::encrypt(input, output, key, cipher, mode);
#ifdef WITH_CUDA
  } else if (input.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::encrypt(input, output, key, cipher, mode);
#endif
  } else {
    TORCH_CHECK(false, TENSOR_DEVICE_TYPE_IS_NOT_SUPPORTED);
  }
}

Tensor decrypt_pybind(Tensor input, Tensor output, Tensor key, const std::string& cipher, const std::string& mode) {
  if (input.device().type() == DeviceType::CPU) {
    return cpu::decrypt(input, output, key, cipher, mode);
#ifdef WITH_CUDA
  } else if (input.device().type() == DeviceType::CUDA) {
    return torch::csprng::cuda::decrypt(input, output, key, cipher, mode);
#endif
  } else {
    TORCH_CHECK(false, TENSOR_DEVICE_TYPE_IS_NOT_SUPPORTED);
  }
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
#ifdef WITH_CUDA
  return true;
#else
  return false;
#endif
}

TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
  // Random
  m.impl("random_.from",             random_from_to);
  m.impl("random_.to",               random_to);
  m.impl("random_",                  random_);
  // Uniform
  m.impl("uniform_",                 uniform_);
  // Normal
  m.impl("normal_",                  normal_);
  m.impl("normal.Tensor_float_out",  normal_Tensor_float_out);
  m.impl("normal.float_Tensor_out",  normal_float_Tensor_out);
  m.impl("normal.Tensor_Tensor_out", normal_Tensor_Tensor_out);
  m.impl("normal.Tensor_float",      normal_Tensor_float);
  m.impl("normal.float_Tensor",      normal_float_Tensor);
  m.impl("normal.Tensor_Tensor",     normal_Tensor_Tensor);
  // Cauchy
  m.impl("cauchy_",                  cauchy_);
  // LogNormal
  m.impl("log_normal_",              log_normal_);
  // Geometric
  m.impl("geometric_",               geometric_);
  // Exponential
  m.impl("exponential_",             exponential_);
  // Random permutation
  m.impl("randperm.generator_out",   randperm_generator_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("supports_cuda", &supports_cuda);
  m.def("create_random_device_generator", &create_random_device_generator, py::arg("token") = nullptr);
  m.def("create_mt19937_generator", &create_mt19937_generator, py::arg("seed") = nullptr);
  m.def("encrypt", &encrypt_pybind);
  m.def("decrypt", &decrypt_pybind);
}
