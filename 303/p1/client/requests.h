#pragma once

#include <openssl/rsa.h>
#include <string>
#include <vector>

/// req_key() writes a request for the server's key on a socket descriptor.
/// When it gets a key back, it writes it to a file.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param keyfile The name of the file to which the key should be written
void req_key(int sd, const std::string &keyfile);

// NB: The remaining function declarations have the same signature, so that we
//     can store pointers to them in an array

/// req_reg() sends the REG command to register a new user
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
void req_reg(int sd, RSA *pubkey, const std::string &user,
             const std::string &pass, const std::string &, const std::string &);

/// req_bye() writes a request for the server to exit.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
void req_bye(int sd, RSA *pubkey, const std::string &user,
             const std::string &pass, const std::string &, const std::string &);

/// req_sav() writes a request for the server to save its contents
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
void req_sav(int sd, RSA *pubkey, const std::string &user,
             const std::string &pass, const std::string &, const std::string &);

/// req_set() sends the SET command to set the content for a user
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
/// @param setfile The file whose contents should be sent
void req_set(int sd, RSA *pubkey, const std::string &user,
             const std::string &pass, const std::string &setfile,
             const std::string &);

/// req_get() requests the content associated with a user, and saves it to a
/// file called <user>.file.dat.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
/// @param getname The name of the user whose content should be fetched
void req_get(int sd, RSA *pubkey, const std::string &user,
             const std::string &pass, const std::string &getname,
             const std::string &);

/// req_all() sends the ALL command to get a listing of all users, formatted
/// as text with one entry per line.
///
/// @param sd      The open socket descriptor for communicating with the server
/// @param pubkey  The public key of the server
/// @param user    The name of the user doing the request
/// @param pass    The password of the user doing the request
/// @param allfile The file where the result should go
void req_all(int sd, RSA *pubkey, const std::string &user,
             const std::string &pass, const std::string &allfile,
             const std::string &);

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
std::vector<uint8_t> send_cmd(int sd, RSA *pub, const std::string &cmd, const std::vector<uint8_t> &msg);

/// Create unencrypted ablock contents from two strings
///
/// @param s1 The first string
/// @param s2 The second string
///
/// @return A vec representing the two strings
std::vector<uint8_t> ablock_ss(const std::string &s1, const std::string &s2);


/// Create unencrypted ablock contents from two strings
///
/// @param s1 The first string
/// @param s2 The second string
/// @param s3 third string
/// @return A vec representing the three strings
std::vector<uint8_t> ablock_sss(const std::string &s1, const std::string &s2, const std::string &s3);


/// Pad a vec with random characters to get it to size sz
///
/// @param v  The vector to pad
/// @param sz The number of bytes to add
///
/// @returns true if the padding was done, false on any error
bool padR(std::vector<uint8_t> &v, size_t sz);

/// If a buffer consists of OKbbbbd+, where bbbb is a 4-byte binary integer
/// and d+ is a string of characters, write the bytes (d+) to a file
///
/// @param buf      The buffer holding a response
/// @param filename The name of the file to write
void send_result_to_file(const std::vector<uint8_t> &buf, const std::string &filename);

std::vector<uint8_t> null(int count);

std::string get_response_code(std::vector<uint8_t> &v);