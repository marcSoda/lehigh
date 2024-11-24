#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/wait.h>

#include "../common/contextmanager.h"
#include "../common/protocol.h"

#include "functable.h"
#include "helpers.h"
#include "map.h"
#include "map_factories.h"
#include "mru.h"
#include "quotas.h"
#include "storage.h"

using namespace std;

/// MyStorage is the student implementation of the Storage class
class MyStorage : public Storage {
  /// The map of authentication information, indexed by username
  Map<string, AuthTableEntry> *auth_table;

  /// The map of key/value pairs
  Map<string, vector<uint8_t>> *kv_store;

  /// The name of the file from which the Storage object was loaded, and to
  /// which we persist the Storage object every time it changes
  string filename = "";

  /// The open file
  FILE *storage_file = nullptr;

  /// The upload quota
  const size_t up_quota;

  /// The download quota
  const size_t down_quota;

  /// The requests quota
  const size_t req_quota;

  /// The number of seconds over which quotas are enforced
  const double quota_dur;

  /// The table for tracking the most recently used keys
  mru_manager *mru;

  /// A table for tracking quotas
  Map<string, Quotas *> *quota_table;

  /// The name of the admin user
  string admin_name;

  /// The function table, to support executing map/reduce on the kv_store
  FuncTable *funcs;

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
  MyStorage(const std::string &fname, size_t buckets, size_t upq, size_t dnq,
            size_t rqq, double qd, size_t top, const std::string &admin)
      : auth_table(authtable_factory(buckets)),
        kv_store(kvstore_factory(buckets)), filename(fname), up_quota(upq),
        down_quota(dnq), req_quota(rqq), quota_dur(qd), mru(mru_factory(top)),
        quota_table(quotatable_factory(buckets)), admin_name(admin),
        funcs(functable_factory()) {}

  /// Destructor for the storage object.
  virtual ~MyStorage() {
    cout << "my_storage.cc::~MyStorage() is not implemented\n";
  }

  /// Create a new entry in the Auth table.  If the user already exists, return
  /// an error.  Otherwise, create a salt, hash the password, and then save an
  /// entry with the username, salt, hashed password, and a zero-byte content.
  ///
  /// @param user The user name to register
  /// @param pass The password to associate with that user name
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t add_user(const string &user, const string &pass) {
    return add_user_helper(user, pass, auth_table, storage_file);
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
    return set_user_data_helper(user, pass, content, auth_table, storage_file);
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
    return get_user_data_helper(user, pass, who, auth_table);
  }

  /// Return a newline-delimited string containing all of the usernames in the
  /// auth table
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t get_all_users(const string &user, const string &pass) {
    return get_all_users_helper(user, pass, auth_table);
  }

  /// Authenticate a user
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t auth(const string &user, const string &pass) {
    return auth_helper(user, pass, auth_table);
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
    return kv_insert_helper(user, pass, key, val, auth_table, kv_store,
                            storage_file, mru, up_quota, down_quota, req_quota,
                            quota_dur, quota_table);
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
    return kv_get_helper(user, pass, key, auth_table, kv_store, mru, up_quota,
                         down_quota, req_quota, quota_dur, quota_table);
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
    return kv_delete_helper(user, pass, key, auth_table, kv_store, storage_file,
                            mru, up_quota, down_quota, req_quota, quota_dur,
                            quota_table);
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
    return kv_upsert_helper(user, pass, key, val, auth_table, kv_store,
                            storage_file, mru, up_quota, down_quota, req_quota,
                            quota_dur, quota_table);
  };

  /// Return all of the keys in the kv_store, as a "\n"-delimited string
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t kv_all(const string &user, const string &pass) {
    return kv_all_helper(user, pass, auth_table, kv_store, up_quota, down_quota,
                         req_quota, quota_dur, quota_table);
  };

  /// Return all of the keys in the kv_store's MRU cache, as a "\n"-delimited
  /// string
  ///
  /// @param user The name of the user who made the request
  /// @param pass The password for the user, used to authenticate
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t kv_top(const string &user, const string &pass) {
    return kv_top_helper(user, pass, auth_table, mru, up_quota, down_quota,
                         req_quota, quota_dur, quota_table);
  };

  /// Register a .so with the function table
  ///
  /// @param user   The name of the user who made the request
  /// @param pass   The password for the user, used to authenticate
  /// @param mrname The name to use for the registration
  /// @param so     The .so file contents to register
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t register_mr(const string &user, const string &pass,
                               const string &mrname,
                               const vector<uint8_t> &so) {
    auto auth_tuple = this->auth(user, pass);
    if(!auth_tuple.succeeded || user != admin_name) return {false, RES_ERR_LOGIN, {}};
    auto res = funcs->register_mr(mrname, so);
    return {true, res, {}};
  };

  /// Run a map/reduce on all the key/value tuples of the kv_store
  ///
  /// @param user   The name of the user who made the request
  /// @param pass   The password for the user, to authenticate
  /// @param mrname The name of the map/reduce functions to use
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t invoke_mr(const string &user, const string &pass,
                             const string &mrname) {
    auto auth_tuple = this->auth(user, pass);
    if (!auth_tuple.succeeded) return {false, RES_ERR_LOGIN, {}};
    auto vals = funcs->get_mr(mrname);
    if(vals.first == nullptr || vals.second == nullptr) return {false, RES_ERR_FUNC, {}};
    int p_1[2], p_2[2];
    if(pipe(p_1) < 0 || pipe(p_2) < 0) return {false, RES_ERR_SERVER, {}};

    pid_t pid, wait_pid;
    pid = fork();
    if (pid < 0) return {false, RES_ERR_SERVER, {}};
    
    bool error = false;
    size_t res, key_length, val_length;
    std::vector<uint8_t> out_vec;

    if (pid > 0){
      close(p_1[0]);
      close(p_2[1]);
      kv_store->do_all_readonly([&](const string auth_key, const vector<uint8_t> auth_vec){
        key_length = auth_key.length();
        val_length = auth_vec.size();

        out_vec.insert(out_vec.end(), (uint8_t *)(&key_length), (uint8_t *)(&key_length) + sizeof(key_length));
        out_vec.insert(out_vec.end(), auth_key.begin(), auth_key.end());
        out_vec.insert(out_vec.end(), (uint8_t *)(&val_length), (uint8_t *)(&val_length) + sizeof(val_length));
        out_vec.insert(out_vec.end(), auth_vec.begin(), auth_vec.end());
        if ((res = write(p_1[1], out_vec.data(), out_vec.size())) != out_vec.size()) error = true;
        out_vec.clear();
      }, [&](){
        size_t neg_one = -1;
        out_vec.insert(out_vec.end(), (uint8_t*)(&neg_one), (uint8_t*)(&neg_one) + sizeof(neg_one));
        if ((res = write(p_1[1], out_vec.data(), out_vec.size())) != out_vec.size()) error = true;
        out_vec.clear();
      });
      if (error) return {false, RES_ERR_SERVER, {}};

      int stat_loc;
      key_length = 0;
      if ((res = read(p_2[0], &key_length, 8)) != 8) return {false, RES_ERR_SERVER, {}};
      out_vec.resize(key_length);
      if ((res = read(p_2[0], out_vec.data(), key_length)) != key_length) return {false, RES_ERR_SERVER, {}};
      if ((wait_pid = waitpid(pid, &stat_loc, 0)) != pid) return {false, RES_ERR_SERVER, {}};
    
      close(p_1[1]);
      close(p_2[0]);
      return {true, RES_OK, out_vec};
    }

    if (pid == 0) {
      // Start by closing the pipes
      close(p_2[0]);
      close(p_1[1]);
      
      // Make sure no new files can be made
      if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT)) {
        _exit(-1);
      }

      // Create a vector to be used in the loop that holds the keys and values
      vector<vector<uint8_t>> tempvec;
      while(true){
        // Read in the length of the key
        size_t size_val;
        int num_bytes = read(p_1[0], &size_val, 8);
        if (num_bytes != 8){
          _exit(-1);
        }
        
        // If the length is -1 we have reached the end and should stop
        if (size_val == (size_t)-1){
          break;
        }
        
        // Read in the key
        vector<uint8_t> data(size_val + 1);
        num_bytes = read(p_1[0], data.data(), size_val);
        if (num_bytes != (int)size_val){
          _exit(-1);
        }

        // Read in the length of the value
        std::string temp_key(data.begin(), data.begin() + size_val);
        num_bytes = read(p_1[0], &size_val, 8);
        if (num_bytes != 8){
          _exit(-1);
        }

        // Read in the value
        vector<uint8_t> temp_value(size_val);
        num_bytes = read(p_1[0], temp_value.data(), size_val);
        if (num_bytes != (int)size_val){
          _exit(-1);
        }

        // Push the values read in to the back of the vector
        tempvec.push_back(vals.first(temp_key, temp_value));
      }
      // Read in from the vector and then write the values to the parent, starting with the length then the value
      std::vector<uint8_t> result_vec = vals.second(tempvec);
      std::vector<uint8_t> write_vec;
      auto vec_length = result_vec.size();
      write_vec.insert(write_vec.end(), (uint8_t*)&vec_length, ((uint8_t*)&vec_length) + sizeof(vec_length));
      write_vec.insert(write_vec.end(), result_vec.begin(), result_vec.end());
      
      // Write the vector to the parent pipe, then make sure that it worked correctly
      size_t write_bytes = write(p_2[1], write_vec.data(), write_vec.size());
      if (write_bytes != write_vec.size()){
        _exit(-1);
      }
      // Close both pipes and exit
      close(p_2[1]);
      close(p_1[0]);
      _exit(0);
    }

    return {false, RES_ERR_SERVER, {}};
  }

  /// Shut down the storage when the server stops.  This method needs to close
  /// any open files related to incremental persistence.  It also needs to clean
  /// up any state related to .so files.  This is only called when all threads
  /// have stopped accessing the Storage object.
  virtual void shutdown() {
    funcs->shutdown();
  }

  /// Write the entire Storage object to the file specified by this.filename. To
  /// ensure durability, Storage must be persisted in two steps.  First, it must
  /// be written to a temporary file (this.filename.tmp).  Then the temporary
  /// file can be renamed to replace the older version of the Storage object.
  ///
  /// @return A result tuple, as described in storage.h
  virtual result_t save_file() {
    return save_file_helper(auth_table, kv_store, filename, storage_file);
  }

  /// Populate the Storage object by loading this.filename.  Note that load()
  /// begins by clearing the maps, so that when the call is complete, exactly
  /// and only the contents of the file are in the Storage object.
  ///
  /// @return A result tuple, as described in storage.h.  Note that a
  /// non-existent
  ///         file is not an error.
  virtual result_t load_file() {
    return load_file_helper(auth_table, kv_store, filename, storage_file, mru);
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
