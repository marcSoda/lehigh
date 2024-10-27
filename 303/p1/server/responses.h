#pragma once

#include <openssl/pem.h>
#include <vector>

#include "../common/protocol.h"

#include "storage.h"

/// In response to a request for a key, do a reliable send of the contents of
/// the pubfile
///
/// @param sd      The socket on which to write the pubfile
/// @param pubfile A vector consisting of pubfile contents
///
/// @return false, to indicate that the server shouldn't stop
bool handle_key(int sd, const std::vector<uint8_t> &pubfile);

/// Respond to an ALL command by generating a list of all the usernames in the
/// Auth table and returning them, one per line.
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_all(int sd, Storage *storage, EVP_CIPHER_CTX *ctx,
                const std::vector<uint8_t> &req);

/// Respond to a SET command by putting the provided data into the Auth table
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_set(int sd, Storage *storage, EVP_CIPHER_CTX *ctx,
                const std::vector<uint8_t> &req);

/// Respond to a GET command by getting the data for a user
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_get(int sd, Storage *storage, EVP_CIPHER_CTX *ctx,
                const std::vector<uint8_t> &req);

/// Respond to a REG command by trying to add a new user
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_reg(int sd, Storage *storage, EVP_CIPHER_CTX *ctx,
                const std::vector<uint8_t> &req);

/// Respond to a BYE command by returning false, but only if the user
/// authenticates
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return true, to indicate that the server should stop, or false on an error
bool handle_bye(int sd, Storage *storage, EVP_CIPHER_CTX *ctx,
                const std::vector<uint8_t> &req);

/// Respond to a SAV command by persisting the file, but only if the user
/// authenticates
///
/// @param sd      The socket onto which the result should be written
/// @param storage The Storage object, which contains the auth table
/// @param ctx     The AES encryption context
/// @param req     The unencrypted contents of the request
///
/// @return false, to indicate that the server shouldn't stop
bool handle_sav(int sd, Storage *storage, EVP_CIPHER_CTX *ctx,
                const std::vector<uint8_t> &req);

//takes a request and returns the username as a string
std::string get_user_string(const std::vector<uint8_t> req);

//takes a request and returns the password as a string
std::string get_pass_string(const std::vector<uint8_t> req);

//takes a request and return a filename as a vector
std::vector<uint8_t> get_file_vec(const std::vector<uint8_t> req);

//takes a request and returns the name of another user as a string
std::string get_other_user(const std::vector<uint8_t> req);

//wrapper arund send_reliable for error checking
void Send_reliably(int sd, const std::vector<uint8_t> &msg);
