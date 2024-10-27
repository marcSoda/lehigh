#include <atomic>
#include <cassert>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/contextmanager.h"
#include "../common/err.h"
#include "../common/file.h"
#include "../common/protocol.h"

#include "functable.h"
#include "functypes.h"

using namespace std;

/// func_table is a table that stores functions that have been registered with
/// our server, so that they can be invoked by clients on the key/value pairs in
/// kv_store.
class my_functable : public FuncTable {

public:
  /// Construct a function table for storing registered functions
  my_functable() {}

  std::vector<void*> v_void;

  std::unordered_map<std::string, std::pair<map_func, reduce_func>> map;

  std::mutex m;

  /// Destruct a function table
  virtual ~my_functable() {}

  /// Register the map() and reduce() functions from the provided .so, and
  /// associate them with the provided name.
  ///
  /// @param mrname The name to associate with the functions
  /// @param so     The so contents from which to find the functions
  ///
  /// @return a status message
  virtual std::string register_mr(const std::string &mrname,
                                  const std::vector<uint8_t> &so) {
    unique_lock lock(m);

    std::string so_name = SO_PREFIX + mrname;

    auto func = map.find(mrname);
    if (func != map.end()) return RES_ERR_FUNC;

    bool res;
    if(!(res = write_file(so_name, so, 0))) return RES_ERR_FUNC;

    auto handle = dlopen(so_name.c_str(), RTLD_LAZY);
    if (dlerror() != NULL) {dlclose(handle); return RES_ERR_FUNC;}

    map_func mf = (map_func)dlsym(handle, MAP_FUNC_NAME.c_str());
    if (dlerror() != NULL) {dlclose(handle); return RES_ERR_SO;}
    reduce_func rf = (reduce_func)dlsym(handle, REDUCE_FUNC_NAME.c_str());
    if (dlerror() != NULL) {dlclose(handle); return RES_ERR_SO;};
    std::pair<map_func, reduce_func> mr = make_pair(mf, rf);
    map.insert({mrname, mr});

    v_void.push_back(handle);

    return RES_OK;
  }

  /// Get the (already-registered) map() and reduce() functions associated with
  /// a name.
  ///
  /// @param name The name with which the functions were mapped
  ///
  /// @return A pair of function pointers, or {nullptr, nullptr} on error
  virtual std::pair<map_func, reduce_func> get_mr(const std::string &mrname) {
    auto it = map.find(mrname);
    auto val = it->second;
    if (it == map.end()) return {nullptr, nullptr}; 
    return {val.first, val.second};
  }

  /// When the function table shuts down, we need to de-register all the .so
  /// files that were loaded.
  virtual void shutdown() {
    unique_lock lock(m);
    for (void* i: v_void){ dlclose(i); }
  }
};

/// Create a FuncTable
FuncTable *functable_factory() { return new my_functable(); };