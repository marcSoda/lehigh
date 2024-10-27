#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <mutex>

#include "../common/contextmanager.h"
#include "../common/err.h"
#include "../common/protocol.h"

#include "authtableentry.h"
#include "format.h"
#include "map.h"
#include "map_factories.h"
#include "persist.h"
#include "storage.h"

using namespace std;

/// MyStorage is the student implementation of the Storage class
class MyStorage : public Storage {
  /// The map of authentication information, indexed by username
  Map<string, AuthTableEntry> *auth_table;

  /// The map of key/value pairs
  Map<string, vector<uint8_t>> *kv_store;

  //mutex for opening file in save_file
  std::mutex mutex;

  /// The name of the file from which the Storage object was loaded, and to
  /// which we persist the Storage object every time it changes
  string filename = "";

  /// The open file
  FILE *storage_file = nullptr;

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
      : auth_table(authtable_factory(buckets)),
        kv_store(kvstore_factory(buckets)), filename(fname) {}

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
    AuthTableEntry u;
    u.username = user;
    u.content = {};

    std::vector<uint8_t> salt_vec((LEN_SALT));
    if (!RAND_bytes(salt_vec.data(), LEN_SALT)) cout << "RAND_BYTES ERROR : buf size of " << sizeof(salt_vec.size()) << endl;
    u.salt = salt_vec;

    std::vector<uint8_t> pass_vec { pass.begin(), pass.end() };
    pass_vec.insert(pass_vec.end(), salt_vec.begin(), salt_vec.end());
    pass_vec.resize(LEN_PASSWORD);

    std::vector<uint8_t> pass_hash((LEN_PASSHASH));
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, pass_vec.data(), pass_vec.size());
    SHA256_Final(pass_hash.data(), &ctx);
    u.pass_hash = pass_hash;

    bool written = false;
    std::vector<uint8_t> dat;
    if (auth_table->insert(user, u, [&]() {
      populate_dat(dat, u);
      written = writef(dat, storage_file);
    })) {
      if (!written) return { false, RES_ERR_SERVER, {} };
      return { true, RES_OK, {} };
    }
    else { return { false, RES_ERR_USER_EXISTS, {}}; }
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
    if(!auth_tuple.succeeded) return { false, RES_ERR_LOGIN, {} };

    AuthTableEntry u;
    u.username = user;
    u.content = content;

    std::vector<uint8_t> pass_hash((LEN_PASSHASH));
    std::vector<uint8_t> salt((LEN_SALT));
    if (!(auth_table->do_with_readonly(user, [&](const AuthTableEntry &auth_user){
      u.salt = auth_user.salt;
      u.pass_hash = auth_user.pass_hash;
    }))) {
      RAND_bytes(salt.data(), LEN_SALT);
      std::vector<uint8_t> pass_vec {pass.begin(), pass.end()};
      pass_vec.insert(pass_vec.end(), salt.begin(), salt.end());
      pass_vec.resize(LEN_PASSWORD);

      SHA256_CTX ctx;
      SHA256_Init(&ctx);
      SHA256_Update(&ctx, pass_vec.data(), pass_vec.size());
      SHA256_Final(pass_hash.data(), &ctx);
    }

    std::vector<uint8_t> dat;
    bool written = false;
    if (!(auth_table->upsert(user, u, [&](){
      populate_dat(dat, u);
      written = writef(dat, storage_file);
    }, [&]() {
      dat.clear();
      dat.insert(dat.begin(), AUTHDIFF.begin(), AUTHDIFF.end());
      size_t lenu = u.username.size();
      dat.insert(dat.end(), (uint8_t*)(&lenu), (uint8_t*)(&lenu) + sizeof(lenu));
      size_t lenc = u.content.size();
      dat.insert(dat.end(), (uint8_t*)(&lenc), (uint8_t*)(&lenc) + sizeof(lenc));
      dat.insert(dat.end(), user.begin(), user.end());
      dat.insert(dat.end(), u.content.begin(), u.content.end());
      int pad = 8 - ((lenu + lenc) % 8);
      dat.resize(dat.size() + pad, 0);
      written = writef(dat, storage_file);
    }))) {
      if (!written) return { false, RES_ERR_SERVER, {} };
      return { true, RES_OK, {} };
    }
    return { false, RES_ERR_LOGIN, {} };
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
    if(!auth_tuple.succeeded) return { false, RES_ERR_LOGIN, {} };

    if(!(auth_table->do_with_readonly(who, [&](AuthTableEntry auth_user){
      res = auth_user.content;
    }))) {
      return { false, RES_ERR_LOGIN, {} };
    }
    if(res.size() == 0) return { false, RES_ERR_NO_DATA, {} };
    return { true, RES_OK, res };
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
    if(!auth_tuple.succeeded) return { false, RES_ERR_LOGIN, {} };

    auth_table->do_all_readonly([&](std::string useless, AuthTableEntry auth_user){
      useless = "useless";
      res.insert(res.end(), auth_user.username.begin(), auth_user.username.end());
      res.push_back('\n');
    }, [](){});
    return { true, RES_OK, res };
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

      pass_vec.resize(LEN_PASSWORD);

      SHA256_CTX ctx;
      SHA256_Init(&ctx);
      SHA256_Update(&ctx, pass_vec.data(), pass_vec.size());
      SHA256_Final(pass_vec_hash.data(), &ctx);

      result = pass_vec_hash == auth_user.pass_hash;
      return;

    }))) {
      return {false, RES_ERR_NO_USER, {}};
    }
    auto res = result ? RES_OK : RES_ERR_LOGIN;
    return {result, res, {}};
  }

  /// Create a new key/value mapping in the table
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  /// @param key  The key whose mapping is being created
  /// @param val  The value to copy into the map
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t kv_insert(const string &user, const string &pass,
                             const string &key, const vector<uint8_t> &val) {
    auto auth_tuple = this->auth(user, pass);
    if(!auth_tuple.succeeded) return { false, RES_ERR_LOGIN, {} };

    std::vector<uint8_t> dat;
    bool written = false;
    if (kv_store->insert(key, val, [&](){
      dat.clear();
      dat.insert(dat.begin(), KVENTRY.begin(), KVENTRY.end());
      size_t lenk = key.length();
      dat.insert(dat.end(), (uint8_t*)(&lenk), (uint8_t*)(&lenk) + sizeof(lenk));
      size_t lenv = val.size();
      dat.insert(dat.end(), (uint8_t*)(&lenv), (uint8_t*)(&lenv) + sizeof(lenv));
      dat.insert(dat.end(), key.begin(), key.end());
      dat.insert(dat.end(), val.begin(), val.end());
      int pad = 8 - ((dat.size()) % 8);
      dat.resize(dat.size() + pad);
      written = writef(dat, storage_file);
    })) {
      if (!written) return { false, RES_ERR_SERVER, {} };
      return { true, RES_OK, {} };
    }
    return { false, RES_ERR_KEY, {} };
  };

  /// Get a copy of the value to which a key is mapped
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  /// @param key  The key whose value is being fetched
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t kv_get(const string &user, const string &pass,
                          const string &key) {
    std::vector<uint8_t> res;

    auto auth_tuple = this->auth(user, pass);
    if(!(auth_tuple.succeeded)) return {false, RES_ERR_LOGIN, {}};

    if(!(kv_store->do_with_readonly(key, [&](std::vector<uint8_t> auth_vec){
      res = auth_vec;
    }))) {
      return { false, RES_ERR_KEY, {} };
    }

    if(res.size() == 0) return { false, RES_ERR_NO_DATA, {} };

    return { true, RES_OK, res };
  };

  /// Delete a key/value mapping
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  /// @param key  The key whose value is being deleted
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t kv_delete(const string &user, const string &pass,
                             const string &key) {
    bool written = true;

    auto auth_tuple = this->auth(user, pass);
    if(!(auth_tuple.succeeded)) return {false, RES_ERR_LOGIN, {}};

    std::vector<uint8_t> out_vec;
    if(!(kv_store->remove(key, [&](){
      out_vec.clear();
      out_vec.insert(out_vec.begin(), KVDELETE.begin(), KVDELETE.end());

      size_t key_length = key.length();
      out_vec.insert(out_vec.end(), (uint8_t*)(&key_length), (uint8_t*)(&key_length) + sizeof(key_length));
      out_vec.insert(out_vec.end(), key.begin(), key.end());

      size_t padding = out_vec.size() % 8;
      unsigned char padding_arr[8 - padding];
      RAND_bytes(padding_arr, (8 - padding));
      for(size_t i = 0; i < sizeof(padding_arr); i++){ out_vec.push_back(padding_arr[i]); }

      if (fwrite(out_vec.data(), sizeof(char), out_vec.size(), storage_file) != out_vec.size()){ written = false; }

      out_vec.clear();
      fflush(storage_file);
      fsync(fileno(storage_file));
    }))){
      if (!written){ return {false, "Write to file failed.", {}}; }
      return {false, RES_ERR_SERVER, {}};
    }
    return {true, RES_OK, {}};
  };

  /// Insert or update, so that the given key is mapped to the give value
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  /// @param key  The key whose mapping is being upserted
  /// @param val  The value to copy into the map
  ///
  /// @return A result tuple, as described in storage.h.  Note that there are
  ///         two "OK" messages, depending on whether we get an insert or an
  ///         update.
  virtual result_t kv_upsert(const string &user, const string &pass,
                             const string &key, const vector<uint8_t> &val) {
    bool written = true;

    auto auth_tuple = this->auth(user, pass);
    if(!(auth_tuple.succeeded)) return {false, RES_ERR_LOGIN, {}};

    std::vector<uint8_t> out_vec;
    if(!(kv_store->upsert(key, val, [&](){
      out_vec.clear();
      out_vec.insert(out_vec.begin(), KVENTRY.begin(), KVENTRY.end());

      size_t key_length = key.length();
      size_t val_length = val.size();
      out_vec.insert(out_vec.end(), (uint8_t*)(&key_length), (uint8_t*)(&key_length) + sizeof(key_length));
      out_vec.insert(out_vec.end(), (uint8_t*)(&val_length), (uint8_t*)(&val_length) + sizeof(val_length));
      out_vec.insert(out_vec.end(), key.begin(), key.end());
      out_vec.insert(out_vec.end(), val.begin(), val.end());

      size_t padding = out_vec.size() % 8;
      unsigned char padding_arr[8 - padding];
      RAND_bytes(padding_arr, (8 - padding));
      for(size_t i = 0; i < sizeof(padding_arr); i++){ out_vec.push_back(padding_arr[i]); }

      if (fwrite(out_vec.data(), sizeof(char), out_vec.size(), storage_file) != out_vec.size()){ written = false; }

      out_vec.clear();
      fflush(storage_file);
      fsync(fileno(storage_file));

    }, [&](){
      out_vec.clear();
      out_vec.insert(out_vec.begin(), KVUPDATE.begin(), KVUPDATE.end());

      size_t key_length = key.length();
      size_t val_length = val.size();

      out_vec.insert(out_vec.end(), (uint8_t*)(&key_length), (uint8_t*)(&key_length) + sizeof(key_length));
      out_vec.insert(out_vec.end(), (uint8_t*)(&val_length), (uint8_t*)(&val_length) + sizeof(val_length));
      out_vec.insert(out_vec.end(), key.begin(), key.end());
      out_vec.insert(out_vec.end(), val.begin(), val.end());

      size_t padding = out_vec.size() % 8;
      unsigned char padding_arr[8 - padding];
      RAND_bytes(padding_arr, (8 - padding));
      for(size_t i = 0; i < sizeof(padding_arr); i++){ out_vec.push_back(padding_arr[i]); }

      if (fwrite(out_vec.data(), sizeof(char), out_vec.size(), storage_file) != out_vec.size()){ written = false; }

      out_vec.clear();
      fflush(storage_file);
      fsync(fileno(storage_file));
    }))){
      if (!written){ return {false, "Write to file failed.", {}}; }
      return {true, RES_OKUPD, {}};
    }
    return {true, RES_OKINS, {}};
  };

  /// Return all of the keys in the kv_store, as a "\n"-delimited string
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t kv_all(const string &user, const string &pass) {
    std::string res;

    auto auth_tuple = this->auth(user, pass);
    if(!(auth_tuple.succeeded)) return {false, RES_ERR_LOGIN, {}};

    kv_store->do_all_readonly([&](std::string auth_key, std::vector<uint8_t> auth_vec){
      assert(auth_vec.size() > 0);
      res += auth_key + "\n";
    }, [](){});

    std::vector<uint8_t> res_vec;
    res_vec.insert(res_vec.begin(), res.begin(), res.end());

    return { true, RES_OK, res_vec };
  };

  /// Shut down the storage when the server stops.  This method needs to close
  /// any open files related to incremental persistence.  It also needs to clean
  /// up any state related to .so files.  This is only called when all threads
  /// have stopped accessing the Storage object.
  virtual void shutdown() {
  }

  /// Write the entire Storage object to the file specified by this.filename. To
  /// ensure durability, Storage must be persisted in two steps.  First, it must
  /// be written to a temporary file (this.filename.tmp).  Then the temporary
  /// file can be renamed to replace the older version of the Storage object.
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t save_file() {
    bool written = true;
    std::string temp_file = filename + ".tmp";
    std::string final_file = filename;

    mutex.lock();
    FILE *f = fopen(temp_file.c_str(), "wb");
    if (f == nullptr)
      return {false, "Unable to open file for writing", {}};
    ContextManager closer([&]() { fclose(f); });

    std::vector<uint8_t> out_vec;

    auth_table->do_all_readonly([&](std::string, AuthTableEntry auth_user){
      out_vec.clear();
      out_vec.insert(out_vec.end(), AUTHENTRY.begin(), AUTHENTRY.end());
      size_t user_length = auth_user.username.length();
      size_t salt_length = auth_user.salt.size();
      size_t hash_length = auth_user.pass_hash.size();
      size_t file_length = auth_user.content.size();

      out_vec.insert(out_vec.end(), (uint8_t*)(&user_length), (uint8_t*)(&user_length) + sizeof(user_length));
      out_vec.insert(out_vec.end(), (uint8_t*)(&salt_length), (uint8_t*)(&salt_length) + sizeof(salt_length));
      out_vec.insert(out_vec.end(), (uint8_t*)(&hash_length), (uint8_t*)(&hash_length) + sizeof(hash_length));
      out_vec.insert(out_vec.end(), (uint8_t*)(&file_length), (uint8_t*)(&file_length) + sizeof(file_length));

      out_vec.insert(out_vec.end(), auth_user.username.begin(), auth_user.username.end());
      out_vec.insert(out_vec.end(), auth_user.salt.begin(), auth_user.salt.end());
      out_vec.insert(out_vec.end(), auth_user.pass_hash.begin(), auth_user.pass_hash.end());
      out_vec.insert(out_vec.end(), auth_user.content.begin(), auth_user.content.end());

      size_t padding = out_vec.size() % 8;
      unsigned char padding_arr[8 - padding];
      RAND_bytes(padding_arr, (8 - padding));
      for(size_t i = 0; i < sizeof(padding_arr); i++){
        out_vec.push_back(padding_arr[i]);
      }
      if (fwrite(out_vec.data(), sizeof(char), out_vec.size(), f) != out_vec.size()){
        written = false;
      }
      out_vec.clear();
    }, [&](){kv_store->do_all_readonly([&](std::string auth_key, std::vector<uint8_t> auth_vec){
      out_vec.clear();
      out_vec.insert(out_vec.end(), KVENTRY.begin(), KVENTRY.end());
      size_t key_length = auth_key.length();
      size_t vec_length = auth_vec.size();

      out_vec.insert(out_vec.end(), (uint8_t*)(&key_length), (uint8_t*)(&key_length) + sizeof(key_length));
      out_vec.insert(out_vec.end(), (uint8_t*)(&vec_length), (uint8_t*)(&vec_length) + sizeof(vec_length));

      out_vec.insert(out_vec.end(), auth_key.begin(), auth_key.end());
      out_vec.insert(out_vec.end(), auth_vec.begin(), auth_vec.end());

      size_t padding = out_vec.size() % 8;
      unsigned char padding_arr[8-padding];
      RAND_bytes(padding_arr, (8-padding));
      for(size_t i = 0; i< sizeof(padding_arr); i++){
        out_vec.push_back(padding_arr[i]);
      }
      if (fwrite(out_vec.data(), sizeof(char), out_vec.size(), f) != out_vec.size()){
        written = false;
      }
      fflush(f);
      fsync(fileno(f));
      out_vec.clear();
    }, [&](){
      rename(temp_file.c_str(), filename.c_str());
    });});
    mutex.unlock();
    if (!written){ return {false, "Write to file failed.", {}}; }
    return {true, RES_OK,{}};
  }

  /// Populate the Storage object by loading this.filename.  Note that load()
  /// begins by clearing the maps, so that when the call is complete, exactly
  /// and only the contents of the file are in the Storage object.
  ///
  /// @return A result tuple, as described in storage.h.  Note that a
  ///         non-existent file is not an error.
  virtual result_t load_file() {
    storage_file = fopen(filename.c_str(), "a+");
    if (storage_file == nullptr) {
      return {false, "File not found: " + filename, {}};
    }

    //same as p1
    fseek(storage_file, 0, SEEK_END);
    size_t file_size = ftell (storage_file);
    rewind(storage_file);

    uint8_t file_buffer[file_size];
    size_t result = fread(file_buffer, sizeof(char), file_size, storage_file);
    if (result != file_size){
      return {false, "Error reading file", {}};
    }

    auth_table->clear();
    kv_store->clear();

    AuthTableEntry new_user;

    size_t count = 0, username_length, salt_length, hash_length, file_length, key_length, val_length;

    std::string new_key;
    std::vector<uint8_t> bytevec;

    union Conversion{
      uint8_t bytes[8];
      uint64_t value;
    } conversion;

    while (count < file_size){
      if (file_buffer[count] == 'A' && file_buffer[count+4] == 'A'){
        count += 8;

        //get lengths to parse
        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        username_length = conversion.value;

        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        salt_length = conversion.value;

        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        hash_length = conversion.value;

        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        file_length = conversion.value;

        //after getting lengths find username, salt, hash, and file length
        std::string username;
        for(size_t i = 0; i < username_length; i++){ username += file_buffer[count++]; }
        new_user.username = username;

        for (size_t i = 0; i < salt_length; i++){ bytevec.push_back(file_buffer[count++]); }
        new_user.salt = bytevec;

        bytevec.clear();
        for (size_t i = 0; i < hash_length; i++){ bytevec.push_back(file_buffer[count++]); }
        new_user.pass_hash = bytevec;

        bytevec.clear();
        for (size_t i = 0; i < file_length; i++){ bytevec.push_back(file_buffer[count++]); }
        new_user.content = bytevec;

        if(!(auth_table->insert(username, new_user, [](){}))){
          return {false, "Insert Failure for auth_table", {}};
        }

        count += 8 - (count % 8);
        bytevec.clear();
        username.clear();

      } else if (file_buffer[count] == 'K' && file_buffer[count+2] == 'K'){
        count += 8;

        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        key_length = conversion.value;

        for(size_t i = 0; i < 8; i++){ conversion.bytes[i] = file_buffer[count++]; }
        val_length = conversion.value;

        for(size_t i = 0; i < key_length; i++){ new_key += file_buffer[count++]; }

        for(size_t i = 0; i < val_length; i++) { bytevec.push_back(file_buffer[count++]); }

        if (!(kv_store->insert(new_key, bytevec, [](){}))){
          return {false, "Insert Failure for kv_store", {}};
        }

        count += 8 - (count % 8);
        bytevec.clear();
        new_key.clear();
      } else if (file_buffer[count] == 'A' && file_buffer[count+4] == 'D'){ //authdiff
        count += 8;

        for (size_t i = 0; i < 8; i++) {conversion.bytes[i] = file_buffer[count++];}
        username_length = conversion.value;

        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        file_length = conversion.value;

        std::string username;
        for(size_t i = 0; i < username_length; i++){ username += file_buffer[count++]; }

        bytevec.clear();
        for (size_t i = 0; i < file_length; i++){ bytevec.push_back(file_buffer[count++]); }

        auth_table->do_with(username, [&](AuthTableEntry &auth_user){
          auth_user.content = bytevec;
        });
        count += 8 - (count % 8);

        bytevec.clear();
        username.clear();
      } else if (file_buffer[count] == 'K' && file_buffer[count+2] == 'U'){ //kv update
        count += 8;

        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        key_length = conversion.value;

        for(size_t i = 0; i < 8; i++){conversion.bytes[i] = file_buffer[count++]; }
        val_length = conversion.value;

        for(size_t i = 0; i < key_length; i++){ new_key += file_buffer[count++]; }

        bytevec.clear();
        for(size_t i = 0; i < val_length; i++) { bytevec.push_back(file_buffer[count++]); }

        kv_store->upsert(new_key, bytevec, [](){}, [](){});
        count += 8 - (count % 8);

        bytevec.clear();
        new_key.clear();
      } else if (file_buffer[count] == 'K' && file_buffer[count+2] == 'D'){ //kv delete
        count += 8;
        for (size_t i = 0; i < 8; i++) { conversion.bytes[i] = file_buffer[count++]; }
        key_length = conversion.value;

        for(size_t i = 0; i < key_length; i++){ new_key += file_buffer[count++]; }

        kv_store->remove(new_key, [](){});

        count += 8 - (count % 8);;
        new_key.clear();
      }
    }
    return {true, "Loaded: " + filename, {}};
  };
};
/// Create an empty Storage object and specify the file from which it should
/// be loaded.  To avoid exceptions and errors in the constructor, the act of
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
