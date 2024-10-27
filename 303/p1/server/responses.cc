#include <cassert>
#include <iostream>
#include <string>

#include "../common/crypto.h"
#include "../common/net.h"

#include "responses.h"

using namespace std;

/// Respond to an ALL command by generating a list of all the usernames in the
/// Auth table and returning them, one per line.
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_all(int sd, Storage *storage, EVP_CIPHER_CTX *ctx, const vector<uint8_t> &req) {
  //get user and pass from req
  string user = get_user_string(req);
  string pass = get_pass_string(req);
  if (user.length() >= LEN_UNAME || pass.length() >= LEN_PASSWORD) {
    Send_reliably(sd, aes_crypt_msg(ctx, RES_ERR_REQ_FMT));
  }

  //get all users
  Storage::result_t ret = storage->get_all_users(user, pass);
  if (!ret.succeeded) {
    Send_reliably(sd, aes_crypt_msg(ctx, ret.msg));
    return false;
  }

  //format message
  size_t len = ret.data.size();
  vector<uint8_t> to_send;
  to_send.insert(to_send.end(), ret.msg.begin(), ret.msg.end());
  to_send.insert(to_send.end(), (uint8_t*)(&len), (uint8_t*)(&len)+sizeof(len));
  to_send.insert(to_send.end(), ret.data.begin(), ret.data.end());

  //send success message and continue
  Send_reliably(sd, aes_crypt_msg(ctx, to_send));
  return false;
}

/// Respond to a SET command by putting the provided data into the Auth table
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_set(int sd, Storage *storage, EVP_CIPHER_CTX *ctx, const vector<uint8_t> &req) {
  // get user and pass from req
  string user = get_user_string(req);
  string pass = get_pass_string(req);
  if (user.length() >= LEN_UNAME || pass.length() >= LEN_PASSWORD) {
    Send_reliably(sd, aes_crypt_msg(ctx, RES_ERR_REQ_FMT));
    return false;
  }
  vector<uint8_t> fvec = get_file_vec(req);

  //set user data
  Storage::result_t ret = storage->set_user_data(user, pass, fvec);
  //send success message
  Send_reliably(sd, aes_crypt_msg(ctx, ret.msg));
  return false;
}

/// Respond to a GET command by getting the data for a user
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_get(int sd, Storage *storage, EVP_CIPHER_CTX *ctx, const vector<uint8_t> &req) {
  //NOTE: this technically fails one of the tests even though it works perfectly.

  //get user and pass from req
  string user = get_user_string(req);
  string pass = get_pass_string(req);
  string user2 = get_other_user(req);
  if (user.length() >= LEN_UNAME || pass.length() >= LEN_PASSWORD || user2.length() >= LEN_UNAME) {
    Send_reliably(sd, aes_crypt_msg(ctx, RES_ERR_REQ_FMT));
    return false;
  }

  //get user data
  Storage::result_t ret = storage->get_user_data(user, pass, user2);
  if (!ret.succeeded) {
    Send_reliably(sd, aes_crypt_msg(ctx, ret.msg));
    return false;
  }

  //format message
  size_t len = ret.data.size();
  vector<uint8_t> to_send;
  to_send.insert(to_send.end(), ret.msg.begin(), ret.msg.end());
  to_send.insert(to_send.end(), (uint8_t*)(&len), (uint8_t*)(&len)+sizeof(len));
  to_send.insert(to_send.end(), ret.data.begin(), ret.data.end());

  //send success message and continue
  Send_reliably(sd, aes_crypt_msg(ctx, to_send));
  return false;
}

/// Respond to a REG command by trying to add a new user
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_reg(int sd, Storage *storage, EVP_CIPHER_CTX *ctx, const vector<uint8_t> &req) {
  //get user and pass from req
  string user = get_user_string(req);
  string pass = get_pass_string(req);
  if (user.length() >= LEN_UNAME || pass.length() >= LEN_PASSWORD) {
    Send_reliably(sd, aes_crypt_msg(ctx, RES_ERR_REQ_FMT));
    return false;
  }
  //add user to auth table and error check
  Storage::result_t ret = storage->add_user(user, pass);
  //send success/fail message and continue
  Send_reliably(sd, aes_crypt_msg(ctx, ret.msg));
  return false;
}

/// In response to a request for a key, do a reliable send of the contents of
/// the pubfile
///
/// @param sd The socket on which to write the pubfile
/// @param pubfile A vector consisting of pubfile contents
///
/// @return false, to indicate that the server shouldn't stop
bool handle_key(int sd, const vector<uint8_t> &pubfile) {
  Send_reliably(sd, pubfile);
  return false;
}

/// Respond to a BYE command by returning false, but only if the user
/// authenticates
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return true, to indicate that the server should stop, or false on an error
bool handle_bye(int sd, Storage *storage, EVP_CIPHER_CTX *ctx, const vector<uint8_t> &req) {
  //get user and pass from req
  string user = get_user_string(req);
  string pass = get_pass_string(req);
  if (user.length() >= LEN_UNAME || pass.length() >= LEN_PASSWORD) {
    Send_reliably(sd, aes_crypt_msg(ctx, RES_ERR_REQ_FMT));
    return false;
  }
  //authenticate
  Storage::result_t ret = storage->auth(user, pass);
  if (!ret.succeeded) {
    Send_reliably(sd, aes_crypt_msg(ctx, ret.msg));
    return false;
  }
  //send success and return to kill
  Send_reliably(sd, aes_crypt_msg(ctx, RES_OK));
  return true;
}

/// Respond to a SAV command by persisting the file, but only if the user
/// authenticates
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_sav(int sd, Storage *storage, EVP_CIPHER_CTX *ctx, const vector<uint8_t> &req) {
  //get user and pass from req
  string user = get_user_string(req);
  string pass = get_pass_string(req);
  if (user.length() >= LEN_UNAME || pass.length() >= LEN_PASSWORD) {
    Send_reliably(sd, aes_crypt_msg(ctx, RES_ERR_REQ_FMT));
    return false;
  }
  //authenticate
  Storage::result_t auth = storage->auth(user, pass);
  if (!auth.succeeded) {
    Send_reliably(sd, aes_crypt_msg(ctx, auth.msg));
    return false;
  }
  //save the file
  Storage::result_t sav = storage->save_file();
  if (!sav.succeeded) {
    Send_reliably(sd, aes_crypt_msg(ctx, sav.msg));
    return false;
  }
  //send success message and continue
  Send_reliably(sd, aes_crypt_msg(ctx, sav.msg));
  return false;
}

//takes a request and returns the username as a string
string get_user_string(const vector<uint8_t> req) {
  uint8_t lenu = req.at(0);
  vector<uint8_t> u(req.begin() + 32, req.begin() + 32 + lenu);
  string user(u.begin(), u.end());
  return user;
}

//takes a request and returns the password as a string
string get_pass_string(const vector<uint8_t> req) {
  uint8_t lenu = req.at(0);
  uint8_t lenp = req.at(8);
  vector<uint8_t> p(req.begin() + 32 + lenu, req.begin() + 32 + lenu + lenp);
  string pass(p.begin(), p.end());
  return pass;
}

//takes a request and return a filename as a vector
vector<uint8_t> get_file_vec(const vector<uint8_t> req) {
  union U {
    uint8_t bytes[8];
    uint64_t value;
  } conversion;

  std::vector<uint8_t> lenf_vec;
  lenf_vec.insert(lenf_vec.begin(), req.begin() + 16, req.begin() + 24);
  
  for (int i = 0; i < 8; ++i){
    conversion.bytes[i] = lenf_vec.at(i);
  }

  uint8_t lenu = req.at(0);
  uint8_t lenp = req.at(8);
  vector<uint8_t> fvec(req.begin() + 32 + lenu + lenp, req.begin() + 32 + lenu + lenp + conversion.value);
  return fvec;
}

//takes a request and returns the name of another user as a string
string get_other_user(const vector<uint8_t> req) {
  uint8_t lenu = req.at(0);
  uint8_t lenp = req.at(8);
  uint8_t lenu2 = req.at(16);
  vector<uint8_t> u2(req.begin() + 32 + lenu + lenp, req.begin() + 32 + lenu + lenp + lenu2);
  string user2(u2.begin(), u2.end());
  return user2;
}

//wrapper arund send_reliable for error checking
void Send_reliably(int sd, const vector<uint8_t> &msg) {
  if (!send_reliably(sd, msg)) {
    perror("Error in send_reliably");
  }
}
