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
#include "kernels.cuh"
#endif

using namespace at;
using namespace torch::csprng;

// ==================================================== Random ========================================================

Tensor& random_(Tensor& self, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::random_(self, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::random_(self, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to,
                       c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::random_from_to(self, from, to, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::random_from_to(self, from, to, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor& random_to(Tensor& self, int64_t to,
                  c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::random_to(self, to, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::random_to(self, to, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

// ==================================================== Uniform =======================================================

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::uniform_(self, from, to, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::uniform_(self, from, to, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

// ==================================================== Normal ========================================================

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::normal_(self, mean, std, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::normal_(self, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor& normal_Tensor_float_out(Tensor& output, const Tensor& mean, double std, c10::optional<Generator> gen) {
  if (output.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_float_out(output, mean, std, gen);
#ifdef WITH_CUDA
  } else if (output.device().type() == DeviceType::CUDA) {
    return cuda::normal_Tensor_float_out(output, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor& normal_float_Tensor_out(Tensor& output, double mean, const Tensor& std, c10::optional<Generator> gen) {
  if (output.device().type() == DeviceType::CPU) {
    return cpu::normal_float_Tensor_out(output, mean, std, gen);
#ifdef WITH_CUDA
  } else if (output.device().type() == DeviceType::CUDA) {
    return cuda::normal_float_Tensor_out(output, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor& normal_Tensor_Tensor_out(Tensor& output, const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  if (output.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_Tensor_out(output, mean, std, gen);
#ifdef WITH_CUDA
  } else if (output.device().type() == DeviceType::CUDA) {
    return cuda::normal_Tensor_Tensor_out(output, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor normal_Tensor_float(const Tensor& mean, double std, c10::optional<Generator> gen) {
  if (mean.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_float(mean, std, gen);
#ifdef WITH_CUDA
  } else if (mean.device().type() == DeviceType::CUDA) {
    return cuda::normal_Tensor_float(mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor normal_float_Tensor(double mean, const Tensor& std, c10::optional<Generator> gen) {
  if (std.device().type() == DeviceType::CPU) {
    return cpu::normal_float_Tensor(mean, std, gen);
#ifdef WITH_CUDA
  } else if (std.device().type() == DeviceType::CUDA) {
    return cuda::normal_float_Tensor(mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  if (mean.device().type() == DeviceType::CPU) {
    return cpu::normal_Tensor_Tensor(mean, std, gen);
#ifdef WITH_CUDA
  } else if (mean.device().type() == DeviceType::CUDA) {
    return cuda::normal_Tensor_Tensor(mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

// ==================================================== Cauchy ========================================================

Tensor& cauchy_(Tensor& self, double median, double sigma, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::cauchy_(self, median, sigma, gen);
#ifdef WITH_CUDA
    } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::cauchy_(self, median, sigma, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

// ================================================== LogNormal =======================================================

Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::log_normal_(self, mean, std, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::log_normal_(self, mean, std, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

// ================================================== Geometric =======================================================

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::geometric_(self, p, gen);
#ifdef WITH_CUDA
  } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::geometric_(self, p, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
  }
}

// ================================================== Exponential =====================================================

Tensor& exponential_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  if (self.device().type() == DeviceType::CPU) {
    return cpu::exponential_(self, lambda, gen);
#ifdef WITH_CUDA
    } else if (self.device().type() == DeviceType::CUDA) {
    return cuda::exponential_(self, lambda, gen);
#endif
  } else {
    TORCH_CHECK(false, "generator doesn't support tensor device type");
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
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_random_device_generator", &create_random_device_generator, py::arg("token") = nullptr);
  m.def("create_mt19937_generator", &create_mt19937_generator, py::arg("seed") = nullptr);
}
