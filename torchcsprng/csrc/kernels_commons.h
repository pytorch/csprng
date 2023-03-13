/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sodium.h>

#include <random>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/DistributionTemplates.h>
#include <torch/utils.h>
#include "macros.h"
#include "block_cipher.h"

inline uint64_t make64BitsFrom32Bits(uint32_t hi, uint32_t lo) {
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

// CUDA CSPRNG is actually CPU generator which is used only to generate a random key on CPU for AES running in a block mode on CUDA
struct CSPRNGGeneratorImpl : public c10::GeneratorImpl {
  CSPRNGGeneratorImpl(at::Tensor key) : c10::GeneratorImpl{at::Device(at::DeviceType::CPU), at::DispatchKeySet(at::DispatchKey::CustomRNGKeyId)}, key_{key} {}
  ~CSPRNGGeneratorImpl() = default;
  uint32_t random() { return 0; }
  uint64_t random64() { return 0; }

  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  CSPRNGGeneratorImpl* clone_impl() const override { throw std::runtime_error("not implemented"); }

  static at::DeviceType device_type() { return at::DeviceType::CPU; }

  void set_state(const c10::TensorImpl& new_state) override { throw std::runtime_error("not implemented"); }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { throw std::runtime_error("not implemented"); }

  at::Tensor& key() { return key_; }

  at::Tensor key_;
};
