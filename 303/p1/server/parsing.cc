#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../common/contextmanager.h"
#include "../common/crypto.h"
#include "../common/err.h"
#include "../common/net.h"
#include "../common/protocol.h"

#include "parsing.h"
#include "responses.h"

using namespace std;

//
//todo: inconsistency in manual running and p1.py Prof. Spear says move on...
//todo: error checking
//todo: cleanup
//

/// When a new client connection is accepted, this code will run to figure out
/// what the client is requesting, and to dispatch to the right function for
/// satisfying the request.
///
/// @param sd      The socket on which communication with the client takes place
/// @param pri     The private key used by the server
/// @param pub     The public key file contents, to possibly send to the client
/// @param storage The Storage object with which clients interact
///
/// @return true if the server should halt immediately, false otherwise
bool parse_request(int sd, RSA *pri, const vector<uint8_t> &pub, Storage *storage) {
  vector<uint8_t> rblock(LEN_RKBLOCK) ;
  if (Reliable_get_to_eof_or_n(sd, rblock.begin(), LEN_RKBLOCK) == -1) return false;

  //handle_key if rblock is a kblock
  if (is_kblock(rblock)) {
    return handle_key(sd, pub);
  }

  //decrypt rblock
  vector<uint8_t> rblock_dec(LEN_RKBLOCK);
  if (RSA_private_decrypt(LEN_RKBLOCK, rblock.data(), rblock_dec.data(), pri, RSA_PKCS1_OAEP_PADDING) < 0) {
    perror("ERROR DECRYPTING");
    return false;
  }

  //parse command
  string cmd(rblock_dec.begin(), rblock_dec.begin() + 8);
  //parse aes key
  vector<uint8_t> aeskey(rblock_dec.begin() + 8, rblock_dec.begin() + AES_KEYSIZE + AES_IVSIZE + 8);
  //parse length of ablock

  vector<uint8_t> ablock_len_vec;
  ablock_len_vec.insert(ablock_len_vec.begin(),
    rblock_dec.begin() + 8 + AES_KEYSIZE + AES_IVSIZE, rblock_dec.begin() + 16 + AES_KEYSIZE + AES_IVSIZE);

  //convert uint8_t vec to uint64_t
  union Conversion{
    uint8_t bytes[8];
    uint64_t value;
  } conversion;

  for (size_t i = 0; i < ablock_len_vec.size(); ++i){
    conversion.bytes[i] = ablock_len_vec.at(i);
  }

  //get ablock
  vector<uint8_t> ablock(conversion.value) ;
  if (Reliable_get_to_eof_or_n(sd, ablock.begin(), conversion.value) == -1) return false;

  // decrypt ablock:
  EVP_CIPHER_CTX *ctx = create_aes_context(aeskey, false);
  vector<uint8_t> ablock_dec = aes_crypt_msg(ctx, ablock);
  reset_aes_context(ctx, aeskey, true);

  // Iterate through possible commands, pick the right one, run it
  vector<string> s = {REQ_REG, REQ_BYE, REQ_SAV, REQ_SET, REQ_GET, REQ_ALL};
  decltype(handle_reg) *cmds[] = {handle_reg, handle_bye, handle_sav, handle_set, handle_get, handle_all};

  for (size_t i = 0; i < s.size(); ++i)
    if (cmd == s[i]) return cmds[i](sd, storage, ctx, ablock_dec);

  return false;
}

bool is_kblock(vector<uint8_t> &block) {
  vector<uint8_t> strvec(REQ_KEY.begin(), REQ_KEY.end());
  strvec.resize(LEN_RKBLOCK);
  if (block == strvec) return true;
  return false;
}

int Reliable_get_to_eof_or_n(int sd, std::vector<uint8_t>::iterator pos, int amnt) {
  int ret = reliable_get_to_eof_or_n(sd, pos, amnt) == -1;
  if (ret == -1) {
    perror("ERROR READING FROM CLIENT");
  }
  return ret;
}
