#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <vector>

#include "map.h"

/// ConcurrentHashMap is a concurrent implementation of the Map interface (a
/// Key/Value store).  It is implemented as a vector of buckets, with one lock
/// per bucket.  Since the number of buckets is fixed, performance can suffer if
/// the thread count is high relative to the number of buckets.  Furthermore,
/// the asymptotic guarantees of this data structure are dependent on the
/// quality of the bucket implementation.  If a vector is used within the bucket
/// to store key/value pairs, then the guarantees will be poor if the key range
/// is large relative to the number of buckets.  If an unordered_map is used,
/// then the asymptotic guarantees should be strong.
///
/// The ConcurrentHashMap is templated on the Key and Value types.
///
/// This map uses std::hash to map keys to positions in the vector.  A
/// production map should use something better.
///
/// This map provides strong consistency guarantees: every operation uses
/// two-phase locking (2PL), and the lambda parameters to methods enable nesting
/// of 2PL operations across maps.
///
/// @param K The type of the keys in this map
/// @param V The type of the values in this map

template <typename K, typename V> class ConcurrentHashMap : public Map<K, V> {
  struct bucket {
    std::mutex lock;
    std::list<std::pair<K, V>> entries;
  };
  std::vector<bucket*> buckets;
  size_t num_buckets;

public:
  /// Construct by specifying the number of buckets it should have
  /// @param _buckets The number of buckets
  // ConcurrentHashMap(size_t _buckets) : buckets(std::vector<bucket>(_buckets)) {}
  ConcurrentHashMap(size_t _buckets) : num_buckets(_buckets) {
    for (size_t i = 0; i < _buckets; i++) buckets.emplace_back(new bucket);
  }

  /// Destruct the ConcurrentHashMap
  virtual ~ConcurrentHashMap() {
    for (auto *b : buckets) b->lock.lock();
    for (auto *b : buckets) delete b;
    for (auto *b : buckets) b->lock.unlock();
  }

  /// Clear the map.  This operation needs to use 2pl
  virtual void clear() {
    for (auto *b : buckets) b->lock.lock();
    for (auto *b : buckets) b->entries.clear();
    for (auto *b : buckets) b->lock.unlock();
  }

  /// Insert the provided key/value pair only if there is no mapping for the key yet
  /// @param key        The key to insert
  /// @param val        The value to insert
  /// @param on_success Code to run if the insertion succeeds
  /// @return true if the key/value was inserted, false if the key already existed on the table
  virtual bool insert(K key, V val, std::function<void()> on_success) {
    std::hash<K> hash_func;
    bucket *b = buckets.at(hash_func(key) % num_buckets);
    std::lock_guard<std::mutex> guard(b->lock);
    for (auto i : b->entries) if (key == i.first) return false;
    b->entries.push_back(std::pair(key, val));
    on_success();
    return true;
  }

  /// Insert the provided key/value pair if there is no mapping for the key yet.
  /// If there is a key, then update the mapping by replacing the old value with
  /// the provided value
  /// @param key    The key to upsert
  /// @param val    The value to upsert
  /// @param on_ins Code to run if the upsert succeeds as an insert
  /// @param on_upd Code to run if the upsert succeeds as an update
  /// @return true if the key/value was inserted, false if the key already
  ///         existed in the table and was thus updated instead
  virtual bool upsert(K key, V val, std::function<void()> on_ins, std::function<void()> on_upd) {
    std::hash<K> hash_func;
    bucket *b = buckets.at(hash_func(key) % num_buckets);
    std::lock_guard<std::mutex> guard(b->lock);
    for (auto i = b->entries.begin(); i != b->entries.end(); i++) {
      if (key == i->first) {
        i->second = val;
        on_upd();
        return false;
      }
    }
    b->entries.push_back(std::pair(key, val));
    on_ins();
    return true;
  }

  /// Apply a function to the value associated with a given key.  The function
  /// is allowed to modify the value.
  /// @param key The key whose value will be modified
  /// @param f   The function to apply to the key's value
  /// @return true if the key existed and the function was applied, false otherwise
  virtual bool do_with(K key, std::function<void(V &)> f) {
    std::hash<K> hash_func;
    bucket *b = buckets.at(hash_func(key) % num_buckets);
    std::lock_guard<std::mutex> guard(b->lock);
    for (auto i = b->entries.begin(); i != b->entries.end(); i++) {
      if (key == i->first) {
        f(i->second);
        return true;
      }
    }
    return false;
  }

  /// Apply a function to the value associated with a given key.  The function
  /// is not allowed to modify the value.
  /// @param key The key whose value will be modified
  /// @param f   The function to apply to the key's value
  /// @return true if the key existed and the function was applied, false otherwise
  virtual bool do_with_readonly(K key, std::function<void(const V &)> f) {
    std::hash<K> hash_func;
    bucket *b = buckets.at(hash_func(key) % num_buckets);
    std::lock_guard<std::mutex> guard(b->lock);
    for (auto i : b->entries) {
      if (key == i.first) {
        f(i.second);
        return true;
      }
    }
    return false;
  }

  /// Remove the mapping from a key to its value
  /// @param key        The key whose mapping should be removed
  /// @param on_success Code to run if the remove succeeds
  /// @return true if the key was found and the value unmapped, false otherwise
  virtual bool remove(K key, std::function<void()> on_success) {
    std::hash<K> hash_func;
    bucket *b = buckets.at(hash_func(key) % num_buckets);
    std::lock_guard<std::mutex> guard(b->lock);
    for (auto i = b->entries.begin(); i != b->entries.end(); i++) {
      if (key == i->first) {
        b->entries.erase(i);
        on_success();
        return true;
      }
    }
    return false;
  }

  /// Apply a function to every key/value pair in the map.  Note that the
  /// function is not allowed to modify keys or values.
  /// @param f    The function to apply to each key/value pair
  /// @param then A function to run when this is done, but before unlocking...
  virtual void do_all_readonly(std::function<void(const K, const V &)> f, std::function<void()> then) {
    for (auto *b : buckets) b->lock.lock();
    for (auto *b : buckets) for (auto i : b->entries) f(i.first, i.second);
    then();
    for (auto *b : buckets) b->lock.unlock();
  }
};
