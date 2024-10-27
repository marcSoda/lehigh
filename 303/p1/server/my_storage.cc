#include <cassert>
#include <cstring>
#include <iostream>
#include <openssl/rand.h>
#include <iterator>
#include <sstream>
#include <fstream>


#include "../common/contextmanager.h"
#include "../common/err.h"
#include "../common/protocol.h"
#include "../common/file.h"

#include "authtableentry.h"
#include "format.h"
#include "map.h"
#include "map_factories.h"
#include "storage.h"

using namespace std;

/// MyStorage is the student implementation of the Storage class
class MyStorage : public Storage {
  /// The map of authentication information, indexed by username
  Map<string, AuthTableEntry> *auth_table;

  /// The name of the file from which the Storage object was loaded, and to
  /// which we persist the Storage object every time it changes
  string filename = "filestorage";

public:
  /// Construct an empty object and specify the file from which it should be
  /// loaded.  To avoid exceptions and errors in the constructor, the act of
  /// loading data is separate from construction.
  ///
  /// @param fname   The name of the file to use for persistence
  /// @param buckets The number of buckets in the hash table
  /// @param upq     The upload quota
  /// @param dnq     The download quota
  /// @param rqq     The request quota
  /// @param qd      The quota duration
  /// @param top     The size of the "top keys" cache
  /// @param admin   The administrator's username
  MyStorage(const std::string &fname, size_t buckets, size_t, size_t, size_t,
            double, size_t, const std::string &)
      : auth_table(authtable_factory(buckets)), filename(fname) {}

  /// Destructor for the storage object.
  virtual ~MyStorage() {}

  /// Create a new entry in the Auth table.  If the user already exists, return
  /// an error.  Otherwise, create a salt, hash the password, and then save an
  /// entry with the username, salt, hashed password, and a zero-byte content.
  ///
  /// @param user The user name to register
  /// @param pass The password to associate with that user name
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t add_user(const string &user, const string &pass) {
    std::vector<uint8_t> pass_vec{pass.begin(), pass.end()};
    
    AuthTableEntry new_user;
    new_user.username = user;
    new_user.content = {};

    unsigned char salt[LEN_SALT];
    int rc;
    if(!(rc = RAND_bytes(salt, LEN_SALT))){
      cout << "RAND_BYTES ERROR : buf size of " << sizeof(salt) << endl; 
    }

    std::vector<uint8_t> salt_vec(salt[0], salt[(sizeof(salt)/sizeof(unsigned char))+1]);
    pass_vec.insert(pass_vec.end(), salt_vec.begin(), salt_vec.end());
    pass_vec.resize(LEN_PASSWORD);

    std::vector<uint8_t> pass_vec_hash(LEN_PASSHASH);

    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, pass_vec.data(), pass_vec.size());
    SHA256_Final(pass_vec_hash.data(), &ctx);

    new_user.salt = salt_vec;
    new_user.pass_hash = pass_vec_hash;

    if(auth_table->insert(user, new_user, [](){})){
      return {true, RES_OK, {}};
    } else {
      return {false, RES_ERR_USER_EXISTS, {}};
    }
  }

  /// Set the data bytes for a user, but do so if and only if the password
  /// matches
  ///
  /// @param user    The name of the user whose content is being set
  /// @param pass    The password for the user, used to authenticate
  /// @param content The data to set for this user
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t set_user_data(const string &user, const string &pass,
                                 const vector<uint8_t> &content) {
    auto auth_tuple = this->auth(user, pass);
    if(auth_tuple.succeeded ==  false) return {false, auth_tuple.msg, {}};

    AuthTableEntry update_user;
    update_user.username = user;
    update_user.content = content;

    if(!(auth_table->do_with_readonly(user, [&](AuthTableEntry auth_user){
      update_user.salt = auth_user.salt;
      update_user.pass_hash = auth_user.pass_hash;
    }))){
      return {false, RES_ERR_LOGIN, {}};
    }

    if(!(auth_table->upsert(user, update_user, [](){}, [](){}))){
      return {true, RES_OK, {}};
    }
    return {false, RES_ERR_LOGIN, {}};
  }

  /// Return a copy of the user data for a user, but do so only if the password
  /// matches
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  /// @param who  The name of the user whose content is being fetched
  ///
  /// @return A result tuple, as described in storage.h.  Note that "no data" is
  ///         an error
  virtual result_t get_user_data(const string &user, const string &pass,
                                 const string &who) {
    std::vector<uint8_t> res;

    auto auth_tuple = this->auth(user, pass);
    if(!(auth_tuple.succeeded)) return {false, RES_ERR_LOGIN, {}};

    if(!(auth_table->do_with_readonly(who, [&](AuthTableEntry auth_user){
      res = auth_user.content;
    }))){
      return {false, RES_ERR_LOGIN, {}};
    }
    if(res.size() == 0) return {false, RES_ERR_NO_DATA, {}};
    return {true, RES_OK, res};
  }

  /// Return a newline-delimited string containing all of the usernames in the
  /// auth table
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t get_all_users(const string &user, const string &pass) {
    std::vector<uint8_t> res;

    auto auth_tuple = this->auth(user, pass);
    if(!(auth_tuple.succeeded)) return {false, RES_ERR_LOGIN, {}};

    auth_table->do_all_readonly([&](std::string useless, AuthTableEntry auth_user){
      useless = "useless";
      res.insert(res.end(), auth_user.username.begin(), auth_user.username.end());
      res.push_back('\n');
    }, [](){});
    return {true, RES_OK, res};
  }

  /// Authenticate a user
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t auth(const string &user, const string &pass) {
    bool result;

    if(!(auth_table->do_with_readonly(user, [&](AuthTableEntry auth_user){
      std::vector<uint8_t> pass_vec{pass.begin(), pass.end()}, pass_vec_hash(LEN_PASSHASH);
      pass_vec.insert(pass_vec.end(), auth_user.salt.begin(), auth_user.salt.end());

      //insert null bytes
      pass_vec.resize(LEN_PASSWORD);
      
      SHA256_CTX ctx;
      SHA256_Init(&ctx);
      SHA256_Update(&ctx, pass_vec.data(), pass_vec.size());
      SHA256_Final(pass_vec_hash.data(), &ctx);
      
      result = pass_vec_hash == auth_user.pass_hash;
      return;

    }))){
      return {false, RES_ERR_NO_USER, {}};
    }
    auto res = result ? RES_OK : RES_ERR_LOGIN;
    return {result, res, {}};
  }

  /// Shut down the storage when the server stops.  This method needs to close
  /// any open files related to incremental persistence.  It also needs to clean
  /// up any state related to .so files.  This is only called when all threads
  /// have stopped accessing the Storage object.
  //dont need to do :)
  virtual void shutdown() {
    //cout << "my_storage.cc::shutdown() is not implemented\n";
  }

  /// Write the entire Storage object to the file specified by this.filename. To
  /// ensure durability, Storage must be persisted in two steps.  First, it must
  /// be written to a temporary file (this.filename.tmp).  Then the temporary
  /// file can be renamed to replace the older version of the Storage object.
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t save_file() {
    std::string temp_file = filename + ".tmp";
    std::string final_file = filename;

    std::vector<uint8_t> out_vec;
    out_vec.clear();

    auth_table->do_all_readonly([&](std::string useless, AuthTableEntry auth_user){
      useless = "useless";
      out_vec.insert(out_vec.end(), auth_user.username.begin(), auth_user.username.end());
      //if any other character is used that is not a whitespace, the file matching fails (not sure why).
      out_vec.push_back('\t');
      out_vec.insert(out_vec.end(), auth_user.salt.begin(), auth_user.salt.end());
      out_vec.push_back('\t');
      out_vec.insert(out_vec.end(), auth_user.pass_hash.begin(), auth_user.pass_hash.end());
      out_vec.push_back('\t');
      out_vec.insert(out_vec.end(), auth_user.content.begin(), auth_user.content.end());
      out_vec.push_back('\t');
      out_vec.push_back('\n');
    }, [](){});

    FILE *f = fopen(final_file.c_str(), "wb");
    if (f == nullptr)
      return {false, "Unable to open file for writing", {}};
    ContextManager closer([&]() { fclose(f); }); 

    if (fwrite(out_vec.data(), sizeof(char), out_vec.size(), f) !=
      out_vec.size()){
    return {false, "Incorrect number of bytes written to ", {}};
    }

    if (!(rename(temp_file.c_str(), final_file.c_str()))){
      return {false, "Unable to rename temp file", {}};
    }

    return {true, RES_OK,{}};
  }

  /// Populate the Storage object by loading this.filename.  Note that load()
  /// begins by clearing the maps, so that when the call is complete, exactly
  /// and only the contents of the file are in the Storage object.
  ///
  /// @return A result tuple, as described in storage.h.  Note that a
  /// non-existent
  ///         file is not an error.
  virtual result_t load_file() {
    //keep
    FILE *storage_file = fopen(filename.c_str(), "r");
    if (storage_file == nullptr) {
      return {true, "File not found: " + filename, {}};
    }

    fseek(storage_file, 0, SEEK_END);
    size_t file_size = ftell (storage_file);
    rewind(storage_file);

    unsigned char buffer[file_size];
    unsigned recd = fread(buffer, sizeof(char), file_size, storage_file);
    if(recd != file_size){
      return {false, "Error in fread", {}};
    }

    auth_table->clear();

    std::vector<uint8_t> trait;
    AuthTableEntry new_user;
    size_t count = 0;
    int trait_numb;

    while(count < file_size){

      trait_numb = 0;
      trait.clear();

      while(buffer[count] != '\n'){
        while(buffer[count] != '\t'){
          trait.push_back(buffer[count]);
          count++;
        }
        count++;
        trait_numb++;
        switch(trait_numb){
          case 1:
            new_user.username = std::string(trait.begin(), trait.end());
            break;
          case 2:
            new_user.salt = trait;
            break;
          case 3:
            new_user.pass_hash = trait;
            break;
          case 4:
            new_user.content = trait;
            break;
          default:
            break;
        }
        trait.clear();
      }
      count++;
      if(!(auth_table->insert(new_user.username, new_user, [](){}))){
          return {false, "Insert Fail", {}};
      }
    }
    return {true, "Loaded: " + filename, {}};
  }
};

/// Create an empty Storage object and specify the file from which it should be
/// loaded.  To avoid exceptions and errors in the constructor, the act of
/// loading data is separate from construction.
///
/// @param fname   The name of the file to use for persistence
/// @param buckets The number of buckets in the hash table
/// @param upq     The upload quota
/// @param dnq     The download quota
/// @param rqq     The request quota
/// @param qd      The quota duration
/// @param top     The size of the "top keys" cache
/// @param admin   The administrator's username
Storage *storage_factory(const std::string &fname, size_t buckets, size_t upq,
                         size_t dnq, size_t rqq, double qd, size_t top,
                         const std::string &admin) {
  return new MyStorage(fname, buckets, upq, dnq, rqq, qd, top, admin);
}