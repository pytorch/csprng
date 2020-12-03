/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ==================================================== Random ========================================================

at::Tensor& random_(at::Tensor& self, c10::optional<at::Generator> generator) {
  return at::native::templates::random_impl<RandomKernel, CSPRNGGeneratorImpl>(self, generator);
}

at::Tensor& random_from_to(at::Tensor& self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator) {
  return at::native::templates::random_from_to_impl<RandomFromToKernel, CSPRNGGeneratorImpl>(self, from, to, generator);
}

at::Tensor& random_to(at::Tensor& self, int64_t to, c10::optional<at::Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

// ==================================================== Uniform =======================================================

at::Tensor& uniform_(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator) {
  return at::native::templates::uniform_impl_<UniformKernel, CSPRNGGeneratorImpl>(self, from, to, generator);
}

// ==================================================== Normal ========================================================

at::Tensor& normal_(at::Tensor& self, double mean, double std, c10::optional<at::Generator> generator) {
  return at::native::templates::normal_impl_<NormalKernel, CSPRNGGeneratorImpl>(self, mean, std, generator);
}

at::Tensor& normal_Tensor_float_out(at::Tensor& output, const at::Tensor& mean, double std, c10::optional<at::Generator> gen) {
  return at::native::templates::normal_out_impl<NormalKernel, CSPRNGGeneratorImpl>(output, mean, std, gen);
}

at::Tensor& normal_float_Tensor_out(at::Tensor& output, double mean, const at::Tensor& std, c10::optional<at::Generator> gen) {
  return at::native::templates::normal_out_impl<NormalKernel, CSPRNGGeneratorImpl>(output, mean, std, gen);
}

at::Tensor& normal_Tensor_Tensor_out(at::Tensor& output, const at::Tensor& mean, const at::Tensor& std, c10::optional<at::Generator> gen) {
  return at::native::templates::normal_out_impl<NormalKernel, CSPRNGGeneratorImpl>(output, mean, std, gen);
}

at::Tensor normal_Tensor_float(const at::Tensor& mean, double std, c10::optional<at::Generator> gen) {
  return at::native::templates::normal_impl<NormalKernel, CSPRNGGeneratorImpl>(mean, std, gen);
}

at::Tensor normal_float_Tensor(double mean, const at::Tensor& std, c10::optional<at::Generator> gen) {
  return at::native::templates::normal_impl<NormalKernel, CSPRNGGeneratorImpl>(mean, std, gen);
}

at::Tensor normal_Tensor_Tensor(const at::Tensor& mean, const at::Tensor& std, c10::optional<at::Generator> gen) {
  return at::native::templates::normal_impl<NormalKernel, CSPRNGGeneratorImpl>(mean, std, gen);
}

// ==================================================== Cauchy ========================================================

at::Tensor& cauchy_(at::Tensor& self, double median, double sigma, c10::optional<at::Generator> generator) {
  return at::native::templates::cauchy_impl_<CauchyKernel, CSPRNGGeneratorImpl>(self, median, sigma, generator);
}

// ================================================== LogNormal =======================================================

at::Tensor& log_normal_(at::Tensor& self, double mean, double std, c10::optional<at::Generator> gen) {
  return at::native::templates::log_normal_impl_<LogNormalKernel, CSPRNGGeneratorImpl>(self, mean, std, gen);
}

// ================================================== Geometric =======================================================

at::Tensor& geometric_(at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  return at::native::templates::geometric_impl_<GeometricKernel, CSPRNGGeneratorImpl>(self, p, gen);
}

// ================================================== Exponential =====================================================

at::Tensor& exponential_(at::Tensor& self, double lambda, c10::optional<at::Generator> gen) {
  return at::native::templates::exponential_impl_<ExponentialKernel, CSPRNGGeneratorImpl>(self, lambda, gen);
}
