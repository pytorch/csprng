/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ==================================================== Random ========================================================

Tensor& random_(Tensor& self, c10::optional<Generator> generator);

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, c10::optional<Generator> generator);

Tensor& random_to(Tensor& self, int64_t to, c10::optional<Generator> generator);

// ==================================================== Uniform =======================================================

Tensor& uniform_(Tensor& self, double from, double to, c10::optional<Generator> generator);

// ==================================================== Normal ========================================================

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> generator);

Tensor& normal_Tensor_float_out(Tensor& output, const Tensor& mean, double std, c10::optional<Generator> gen);

Tensor& normal_float_Tensor_out(Tensor& output, double mean, const Tensor& std, c10::optional<Generator> gen);

Tensor& normal_Tensor_Tensor_out(Tensor& output, const Tensor& mean, const Tensor& std, c10::optional<Generator> gen);

Tensor normal_Tensor_float(const Tensor& mean, double std, c10::optional<Generator> gen);

Tensor normal_float_Tensor(double mean, const Tensor& std, c10::optional<Generator> gen);

Tensor normal_Tensor_Tensor(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen);

// ==================================================== Cauchy ========================================================

Tensor& cauchy_(Tensor& self, double median, double sigma, c10::optional<Generator> generator);

// ================================================== LogNormal =======================================================

Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen);

// ================================================== Geometric =======================================================

Tensor& geometric_(Tensor& self, double p, c10::optional<Generator> gen);

// ================================================== Exponential =====================================================

Tensor& exponential_(Tensor& self, double lambda, c10::optional<Generator> gen);
