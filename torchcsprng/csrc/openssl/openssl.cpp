#include <iostream>
#include <ATen/DynamicLibrary.h>
#include <openssl/err.h>
#include <openssl/evp.h>

namespace {
  at::DynamicLibrary &getCryptoLibrary() {
#if defined(_WIN32)
    static at::DynamicLibrary lib("libcrypto.dll");
#elif defined(__APPLE__) || defined(__MACOSX)
    static at::DynamicLibrary lib("libcrypto.dylib");
#else
    static at::DynamicLibrary lib("libcrypto.so");
#endif
    return lib;
  }

  struct Initializer {
    bool libcrypto_loaded = false;
    Initializer() {
      try {
        getCryptoLibrary();
        libcrypto_loaded = true;
      } catch (...) {
        libcrypto_loaded = false;
      }
    }
  };

  Initializer initializer;
}

void ERR_print_errors_fp(FILE *fp) {
  auto fn = reinterpret_cast<decltype(&ERR_print_errors_fp)>(getCryptoLibrary().sym(
      __func__));
  return fn(fp);
}

EVP_CIPHER_CTX *EVP_CIPHER_CTX_new(void) {
  auto fn = reinterpret_cast<decltype(&EVP_CIPHER_CTX_new)>(getCryptoLibrary().sym(
      __func__));
  return fn();
}

void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *ctx) {
  auto fn = reinterpret_cast<decltype(&EVP_CIPHER_CTX_free)>(getCryptoLibrary().sym(
      __func__));
  return fn(ctx);
}

int EVP_CIPHER_CTX_set_padding(EVP_CIPHER_CTX *x, int padding) {
  auto fn = reinterpret_cast<decltype(&EVP_CIPHER_CTX_set_padding)>(getCryptoLibrary().sym(
      __func__));
  return fn(x, padding);
}

int EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *type,
                       ENGINE *impl, const unsigned char *key,
                       const unsigned char *iv) {
  auto fn = reinterpret_cast<decltype(&EVP_EncryptInit_ex)>(getCryptoLibrary().sym(
      __func__));
  return fn(ctx, type, impl, key, iv);
}

int EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out,
                      int *outl, const unsigned char *in, int inl) {
  auto fn = reinterpret_cast<decltype(&EVP_EncryptUpdate)>(getCryptoLibrary().sym(
      __func__));
  return fn(ctx, out, outl, in, inl);
}

int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl) {
  auto fn = reinterpret_cast<decltype(&EVP_EncryptFinal_ex)>(getCryptoLibrary().sym(
      __func__));
  return fn(ctx, out, outl);
}

int EVP_DecryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *type,
                       ENGINE *impl, const unsigned char *key, const unsigned char *iv) {
  auto fn = reinterpret_cast<decltype(&EVP_DecryptInit_ex)>(getCryptoLibrary().sym(
      __func__));
  return fn(ctx, type, impl, key, iv);
}

int EVP_DecryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out,
                      int *outl, const unsigned char *in, int inl) {
  auto fn = reinterpret_cast<decltype(&EVP_DecryptUpdate)>(getCryptoLibrary().sym(
      __func__));
  return fn(ctx, out, outl, in, inl);
}

int EVP_DecryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *outm, int *outl) {
  auto fn = reinterpret_cast<decltype(&EVP_DecryptFinal_ex)>(getCryptoLibrary().sym(
      __func__));
  return fn(ctx, outm, outl);
}

const EVP_CIPHER *EVP_aes_128_ctr(void) {
  auto fn = reinterpret_cast<decltype(&EVP_aes_128_ctr)>(getCryptoLibrary().sym(
      __func__));
  return fn();
}

const EVP_CIPHER *EVP_aes_128_ecb(void) {
  auto fn = reinterpret_cast<decltype(&EVP_aes_128_ecb)>(getCryptoLibrary().sym(
      __func__));
  return fn();
}
