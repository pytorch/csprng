#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>

namespace {
  struct Initializer {
    Initializer() {
      ERR_load_crypto_strings();
      OpenSSL_add_all_algorithms();
    }
  };
  Initializer initializer;
} // namespace

constexpr size_t block_t_size = 16;

TORCH_CSPRNG_HOST_DEVICE void encrypt(uint8_t* state, const uint8_t* key) {
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (ctx == nullptr) {
    std::cerr << "EVP_CIPHER_CTX_new failed" << std::endl;
    exit(-1);
  }
  if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), nullptr, key, nullptr) != 1) {
    std::cerr << "EVP_EncryptInit_ex failed" << std::endl;
    exit(-1);
  }
  uint8_t encrypted[32];  // WHY??? why 32 but not 16
  int encrypted_len = 0;
  if (EVP_EncryptUpdate(ctx, encrypted, &encrypted_len, state, 16) != 1) {
    std::cerr << "EVP_EncryptUpdate failed" << std::endl;
    exit(-1);
  }
  if (EVP_EncryptFinal_ex(ctx, encrypted + encrypted_len, &encrypted_len) != 1) {
    std::cerr << "EVP_EncryptFinal_ex failed" << std::endl;
    exit(-1);
  }
  std::memcpy(state, encrypted, 16);
  EVP_CIPHER_CTX_free(ctx);
}

TORCH_CSPRNG_HOST_DEVICE void decrypt(uint8_t* state, const uint8_t* key) {
//  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
//  if (ctx == nullptr) {
//    std::cerr << "EVP_CIPHER_CTX_new failed" << std::endl;
//    exit(-1);
//  }
//  if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), nullptr, key, nullptr) != 1) {
//    std::cerr << "EVP_EncryptInit_ex failed" << std::endl;
//    exit(-1);
//  }
//  uint8_t decrypted[32];
//  int decrypted_len = 0;
//  if (EVP_DecryptUpdate(ctx, decrypted, &decrypted_len, state, 16) != 1) {
//    std::cerr << "EVP_DecryptUpdate failed" << std::endl;
//    exit(-1);
//  }
//  if (EVP_DecryptFinal_ex(ctx, decrypted + decrypted_len, &decrypted_len) != 1) {
//    std::cerr << "EVP_DecryptFinal_ex failed" << std::endl;
//    exit(-1);
//  }
//  std::memcpy(state, decrypted, 16);
//  EVP_CIPHER_CTX_free(ctx);
}
