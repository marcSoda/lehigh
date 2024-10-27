#include <cassert>
#include <iostream>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <vector>

#include "err.h"

using namespace std;

/// Run the AES symmetric encryption/decryption algorithm on a buffer of bytes.
/// Note that this will do either encryption or decryption, depending on how the
/// provided CTX has been configured.  After calling, the CTX cannot be used
/// again until it is reset.
///
/// @param ctx The pre-configured AES context to use for this operation
/// @param msg A buffer of bytes to encrypt/decrypt
///
/// @return A vector with the encrypted or decrypted result, or an empty
///         vector if there was an error
vector<uint8_t> aes_crypt_msg(EVP_CIPHER_CTX *ctx, const unsigned char *start, int count) {

  int cipher_block_size = EVP_CIPHER_block_size(EVP_CIPHER_CTX_cipher(ctx));
  std::vector<uint8_t> out_buf(BUFSIZ * cipher_block_size);
  int out_len_update, out_len_final_ex;

  if (!EVP_CipherUpdate(ctx, out_buf.data(), &out_len_update, start, count)) {
    fprintf(stderr, "Error in EVP_CipherUpdate: %s\n", ERR_error_string(ERR_get_error(), nullptr));
  }

  if (!EVP_CipherFinal_ex(ctx, out_buf.data() + out_len_update, &out_len_final_ex)) {
    fprintf(stderr, "Error in EVP_CipherFinal_ex: %s\n", ERR_error_string(ERR_get_error(), nullptr));
  }

  out_buf.resize(out_len_update + out_len_final_ex);

  return out_buf;
}
