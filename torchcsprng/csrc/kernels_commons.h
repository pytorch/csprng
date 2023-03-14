/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define _CSPRNG_SUBKEY_CTX    "randsubk"

#include <sodium.h>

#include <random>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/DistributionTemplates.h>
#include <torch/utils.h>
#include "macros.h"
#include "block_cipher.h"

// CUDA CSPRNG is actually CPU generator which is used only to generate a random key on CPU for AES running in a block mode on CUDA
struct CSPRNGGeneratorImpl : public c10::GeneratorImpl {
  CSPRNGGeneratorImpl(at::Tensor key) : c10::GeneratorImpl{at::Device(at::DeviceType::CPU), at::DispatchKeySet(at::DispatchKey::CustomRNGKeyId)}, key_{key} {
    if (key_.size(0) != crypto_kdf_KEYBYTES) {
      throw std::runtime_error("received key of invalid length");
    }
  }
  ~CSPRNGGeneratorImpl() = default;

  void random_subkey(uint8_t* subkey, size_t len) {
    crypto_kdf_derive_from_key(subkey, len, counter++, _CSPRNG_SUBKEY_CTX, key());
  }

  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  CSPRNGGeneratorImpl* clone_impl() const override { throw std::runtime_error("not implemented"); }

  static at::DeviceType device_type() { return at::DeviceType::CPU; }

  void set_state(const c10::TensorImpl& new_state) override { throw std::runtime_error("not implemented"); }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { throw std::runtime_error("not implemented"); }

  uint8_t* key() { return key_.data_ptr<uint8_t>(); }

  at::Tensor key_;
  uint64_t counter = 0;
};
