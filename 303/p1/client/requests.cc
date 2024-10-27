#include <cassert>
#include <cstring>
#include <iostream>
#include <openssl/rand.h>
#include <vector>
#include <fstream>
#include <bitset>

#include "../common/contextmanager.h"
#include "../common/crypto.h"
#include "../common/file.h"
#include "../common/net.h"
#include "../common/protocol.h"
#include "../common/err.h"
#include "requests.h"


using namespace std;

/// req_key() writes a request for the server's key on a socket descriptor.
/// When it gets a key back, it writes it to a file.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param keyfile The name of the file to which the key should be written
void req_key(int sd, const string &keyfile) {
  std::vector<uint8_t> kblock(REQ_KEY.size());
  std::copy(REQ_KEY.begin(), REQ_KEY.end(), kblock.begin());
  
  //automatically pads 
  kblock.resize(LEN_RKBLOCK);

  bool rc;
  if(!(rc = send_reliably(sd, kblock))){
    perror("ERROR with send reliably");
  }
  auto res_o = reliable_get_to_eof(sd);

  write_file(keyfile, res_o, 0);
}

/// req_reg() sends the REG command to register a new user
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
void req_reg(int sd, RSA *pubkey, const string &user, const string &pass,
             const string &, const string &) {
  std::vector<uint8_t> msg;
  
  //create unencrypted ablock
  size_t user_length = user.length();
  size_t pass_length = pass.length();

  //insert user length and pass length
  msg.insert(msg.begin(), (uint8_t*)(&user_length), (uint8_t*)(&user_length) + sizeof(user_length));
  msg.insert(msg.end(), (uint8_t*)(&pass_length), (uint8_t*)(&pass_length) + sizeof(pass_length));

  //insert null block
  auto null_block = null(16);
  msg.insert(msg.end(), null_block.begin(), null_block.end());

  // std::vector<uint8_t> user_pass;
  std::vector<uint8_t> user_pass = ablock_ss(user, pass);
  msg.insert(msg.end(), user_pass.begin(), user_pass.end());

  auto res_o = send_cmd(sd, pubkey, REQ_REG, msg);

  auto response_code = get_response_code(res_o);

  cout << response_code << endl;
}

/// req_bye() writes a request for the server to exit.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
void req_bye(int sd, RSA *pubkey, const string &user, const string &pass,
             const string &, const string &) {
  std::vector<uint8_t> msg;
  
  //create unencrypted ablock
  size_t user_length = user.length();
  size_t pass_length = pass.length();

  //insert user length and pass length
  msg.insert(msg.begin(), (uint8_t*)(&user_length), (uint8_t*)(&user_length) + sizeof(user_length));
  msg.insert(msg.end(), (uint8_t*)(&pass_length), (uint8_t*)(&pass_length) + sizeof(pass_length));

  //insert null block
  auto null_block = null(16);
  msg.insert(msg.end(), null_block.begin(), null_block.end());

  // std::vector<uint8_t> user_pass;
  std::vector<uint8_t> user_pass = ablock_ss(user, pass);
  msg.insert(msg.end(), user_pass.begin(), user_pass.end());

  auto res_o = send_cmd(sd, pubkey, REQ_BYE, msg);

  auto response_code = get_response_code(res_o);

  cout << response_code << endl; 
}

/// req_sav() writes a request for the server to save its contents
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
void req_sav(int sd, RSA *pubkey, const string &user, const string &pass,
             const string &, const string &) {
  std::vector<uint8_t> msg;
  
  //create unencrypted ablock
  size_t user_length = user.length();
  size_t pass_length = pass.length();

  //insert user length and pass length
  msg.insert(msg.begin(), (uint8_t*)(&user_length), (uint8_t*)(&user_length) + sizeof(user_length));
  msg.insert(msg.end(), (uint8_t*)(&pass_length), (uint8_t*)(&pass_length) + sizeof(pass_length));

  //insert null block
  auto null_block = null(16);
  msg.insert(msg.end(), null_block.begin(), null_block.end());

  // std::vector<uint8_t> user_pass;
  std::vector<uint8_t> user_pass = ablock_ss(user, pass);
  msg.insert(msg.end(), user_pass.begin(), user_pass.end());

  auto res_o = send_cmd(sd, pubkey, REQ_SAV, msg);

  auto response_code = get_response_code(res_o);

  cout << response_code << endl;  
}

/// req_set() sends the SET command to set the content for a user
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
/// @param setfile The file whose contents should be sent
void req_set(int sd, RSA *pubkey, const string &user, const string &pass,
             const string &setfile, const string &) {
 std::vector<uint8_t> msg;
  
  //create unencrypted ablock
  size_t user_length = user.length();
  size_t pass_length = pass.length();
  size_t file_length = setfile.length();

  //insert user length and pass length
  msg.insert(msg.begin(), (uint8_t*)(&user_length), (uint8_t*)(&user_length) + sizeof(user_length));
  msg.insert(msg.end(), (uint8_t*)(&pass_length), (uint8_t*)(&pass_length) + sizeof(pass_length));
  msg.insert(msg.end(), (uint8_t*)(&file_length), (uint8_t*)(&file_length) + sizeof(file_length));

  //insert null block
  auto null_block = null(8);
  msg.insert(msg.end(), null_block.begin(), null_block.end());

  // std::vector<uint8_t> user_pass;
  std::vector<uint8_t> user_pass_file = ablock_sss(user, pass, setfile);
  msg.insert(msg.end(), user_pass_file.begin(), user_pass_file.end());

  auto res_o = send_cmd(sd, pubkey, REQ_SET, msg);

  auto response_code = get_response_code(res_o);

  cout << response_code << endl; 
}
 
/// req_get() requests the content associated with a user, and saves it to a
/// file called <user>.file.dat.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
/// @param getname The name of the user whose content should be fetched
void req_get(int sd, RSA *pubkey, const string &user, const string &pass,
             const string &getname, const string &) {
  std::vector<uint8_t> msg;
  
  //create unencrypted ablock
  size_t user_length = user.length();
  size_t pass_length = pass.length();
  size_t getname_length = getname.length();

  //insert user length and pass length
  msg.insert(msg.begin(), (uint8_t*)(&user_length), (uint8_t*)(&user_length) + sizeof(user_length));
  msg.insert(msg.end(), (uint8_t*)(&pass_length), (uint8_t*)(&pass_length) + sizeof(pass_length));
  msg.insert(msg.end(), (uint8_t*)(&getname_length), (uint8_t*)(&getname_length) + sizeof(getname_length));

  //insert null block
  auto null_block = null(8);
  msg.insert(msg.end(), null_block.begin(), null_block.end());

  // std::vector<uint8_t> user_pass;
  std::vector<uint8_t> user_pass_file = ablock_sss(user, pass, getname);
  msg.insert(msg.end(), user_pass_file.begin(), user_pass_file.end());

  auto res_o = send_cmd(sd, pubkey, REQ_GET, msg);

  std::string str(res_o.begin(), res_o.end()); 

  auto response_code = get_response_code(res_o);
  
  if(response_code.substr(0,3) == "ERR") {
    cout << response_code << endl;
  } else {
    auto content_location = str.substr(16, str.length());
    auto content_file = load_entire_file(content_location);
      
    write_file(getname + ".file.dat", content_file, 0);
    cout << response_code << endl;
  }
}

/// req_all() sends the ALL command to get a listing of all users, formatted
/// as text with one entry per line.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
/// @param allfile The file where the result should go
void req_all(int sd, RSA *pubkey, const string &user, const string &pass,
             const string &allfile, const string &) {
 std::vector<uint8_t> msg;
  
  //create unencrypted ablock
  size_t user_length = user.length();
  size_t pass_length = pass.length();

  //insert user length and pass length
  msg.insert(msg.begin(), (uint8_t*)(&user_length), (uint8_t*)(&user_length) + sizeof(user_length));
  msg.insert(msg.end(), (uint8_t*)(&pass_length), (uint8_t*)(&pass_length) + sizeof(pass_length));

  //insert null block
  auto null_block = null(16);
  msg.insert(msg.end(), null_block.begin(), null_block.end());

  // std::vector<uint8_t> user_pass;
  std::vector<uint8_t> user_pass = ablock_ss(user, pass);
  msg.insert(msg.end(), user_pass.begin(), user_pass.end());

  auto res_o = send_cmd(sd, pubkey, REQ_ALL, msg);

  std::string str(res_o.begin(), res_o.end()); 
  // cout << str << endl;
  auto response_code = str.substr(0,8);
  auto content_len = str.substr(8, 16);
  auto content = str.substr(16, str.length());
  // cout << content << endl;
  std::vector<uint8_t> content_vec(content.begin(), content.end());

  write_file(allfile, content_vec, 0);

  cout << response_code << endl; 
}

/// Send a message to the server, using the common format for secure messages,
/// then take the response from the server, decrypt it, and return it.
///
/// Many of the messages in our server have a common form (@rblock.@ablock):
///   - @rblock padR(enc(pubkey, "CMD".aeskey.length(@msg)))
///   - @ablock enc(aeskey, @msg)
///
/// @param sd  An open socket
/// @param pub The server's public key, for encrypting the aes key
/// @param cmd The command that is being sent
/// @param msg The contents of the @ablock
///
/// @returns a vector with the (decrypted) result, or an empty vector on error
vector<uint8_t> send_cmd(int sd, RSA *pub, const string &cmd, const vector<uint8_t> &msg) {
  std::vector<uint8_t> ablock, ablock_enc, rblock(cmd.size()), rblock_enc(RSA_size(pub)), null_block, aeskey, req;
  
  //create aeskey
  aeskey = create_aes_key();
  EVP_CIPHER_CTX *ctx = create_aes_context(aeskey, true);

  //add msg to a block
  ablock.insert(ablock.begin(), msg.begin(), msg.end());

  ablock_enc = aes_crypt_msg(ctx, ablock);
  
  //create unencrypted rblock
  std::copy(cmd.begin(), cmd.end(), rblock.begin());
  rblock.insert(rblock.end(), aeskey.begin(), aeskey.end());

  //add ablock length to rblock
  size_t ablock_len = ablock_enc.size();
  rblock.insert(rblock.end(), (uint8_t*) &ablock_len, ((uint8_t*) &ablock_len) + sizeof(ablock_len));

  if(!(padR(rblock, LEN_RBLOCK_CONTENT))){
    cout << "Error in padding rblock" << endl;
  }

  int len ;
  if ((len= RSA_public_encrypt(rblock.size(), rblock.data(), rblock_enc.data(), pub, RSA_PKCS1_OAEP_PADDING)) < 0){
    cout << "Error in RSA encrypt" << endl;
  }
  
  req.insert(req.begin(), rblock_enc.begin(), rblock_enc.end());
  req.insert(req.end(), ablock_enc.begin(), ablock_enc.end());

  send_reliably(sd, req);
  auto res_o = reliable_get_to_eof(sd);

  reset_aes_context(ctx, aeskey, false);
  auto res_o_decrypted = aes_crypt_msg(ctx, res_o);

  reclaim_aes_context(ctx);

  return res_o_decrypted;
}

/// Create unencrypted ablock contents from two strings
///
/// @param s1 The first string
/// @param s2 The second string
///
/// @return A vec representing the two strings
vector<uint8_t> ablock_ss(const string &s1, const string &s2) {
  std::vector<uint8_t> vec1(s1.length());
  std::vector<uint8_t> vec2(s2.length());

  std::copy(s1.begin(), s1.end(), vec1.begin());
  std::copy(s2.begin(), s2.end(), vec2.begin());
  
  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  
  return vec1;
}

vector<uint8_t> ablock_sss(const string &s1, const string &s2, const string &s3) {
  std::vector<uint8_t> vec1(s1.length());
  std::vector<uint8_t> vec2(s2.length());
  std::vector<uint8_t> vec3(s3.length());
 
  std::copy(s1.begin(), s1.end(), vec1.begin());
  std::copy(s2.begin(), s2.end(), vec2.begin());
  std::copy(s3.begin(), s3.end(), vec3.begin());

  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  vec1.insert(vec1.end(), vec3.begin(), vec3.end());
  
  return vec1;
}

/// Pad a vec with random characters to get it to size sz
///
/// @param v  The vector to pad
/// @param sz The number of bytes to add
///
/// @returns true if the padding was done, false on any error
bool padR(std::vector<uint8_t> &v, size_t sz){
  int required_size = sz - v.size();
  unsigned char buf[required_size];
  
  int rc;
  if(!(rc = RAND_bytes(buf, sizeof(buf)))){
    cout << "RAND_BYTES ERROR : buf size of " << sizeof(buf) << endl; 
  }

  std::vector<uint8_t> tempVec(&buf[0], &buf[(sizeof(buf) / sizeof(unsigned char))]);
  v.insert(v.end(), tempVec.begin(), tempVec.end());

  return true;
}

vector<uint8_t> null(int count){
  std::vector<uint8_t> res;
  for (int i = 0; i < count; i++){
    res.push_back('\0');
  }
  return res;
}

std::string get_response_code(std::vector<uint8_t> &v){
    //convert to string
    std::string str(v.begin(), v.end()); 
    //get first 8
    std::string str_sub = str.substr(0,8);

    return str_sub == RES_OK ? str_sub : str;
}