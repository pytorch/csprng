/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ==================================================== Random ========================================================

at::Tensor& random_(at::Tensor& self, c10::optional<at::Generator> generator);

at::Tensor& random_from_to(at::Tensor& self, int64_t from, optional<int64_t> to, c10::optional<at::Generator> generator);

at::Tensor& random_to(at::Tensor& self, int64_t to, c10::optional<at::Generator> generator);

// ==================================================== Uniform =======================================================

at::Tensor& uniform_(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator);

// ==================================================== Normal ========================================================

at::Tensor& normal_(at::Tensor& self, double mean, double std, c10::optional<at::Generator> generator);

at::Tensor& normal_Tensor_float_out(at::Tensor& output, const at::Tensor& mean, double std, c10::optional<at::Generator> gen);

at::Tensor& normal_float_Tensor_out(at::Tensor& output, double mean, const at::Tensor& std, c10::optional<at::Generator> gen);

at::Tensor& normal_Tensor_Tensor_out(at::Tensor& output, const at::Tensor& mean, const at::Tensor& std, c10::optional<at::Generator> gen);

at::Tensor normal_Tensor_float(const at::Tensor& mean, double std, c10::optional<at::Generator> gen);

at::Tensor normal_float_Tensor(double mean, const at::Tensor& std, c10::optional<at::Generator> gen);

at::Tensor normal_Tensor_Tensor(const at::Tensor& mean, const at::Tensor& std, c10::optional<at::Generator> gen);

// ==================================================== Cauchy ========================================================

at::Tensor& cauchy_(at::Tensor& self, double median, double sigma, c10::optional<at::Generator> generator);

// ================================================== LogNormal =======================================================

at::Tensor& log_normal_(at::Tensor& self, double mean, double std, c10::optional<at::Generator> gen);

// ================================================== Geometric =======================================================

at::Tensor& geometric_(at::Tensor& self, double p, c10::optional<at::Generator> gen);

// ================================================== Exponential =====================================================

at::Tensor& exponential_(at::Tensor& self, double lambda, c10::optional<at::Generator> gen);

// ================================================Encrypt/Decrypt=====================================================

Tensor encrypt(Tensor input, Tensor output, Tensor key, const std::string& cipher, const std::string& mode);

Tensor decrypt(Tensor input, Tensor output, Tensor key, const std::string& cipher, const std::string& mode);
