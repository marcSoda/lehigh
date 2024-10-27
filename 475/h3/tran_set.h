#ifndef TRANSET_H_
#define TRANSET_H_

#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <random>
#include "common.h"

template <typename K, typename V>
class tran_set {
public:
    int _num_resize = 0;

    tran_set(size_t initial_capacity = 256, float load_factor = 0.5)
    : _capacity(initial_capacity), _size(0), _load_factor(load_factor),
        _tables{std::vector<hash_entry<K, V>>(), std::vector<hash_entry<K, V>>()} {
        _tables[0].resize(_capacity);
        _tables[1].resize(_capacity);
    }

    bool add(const K& key, const V& value) transaction_safe {
        if (contains(key)) return false;
        hash_entry<K, V> x(key, value);
        __transaction_atomic {
            for (int i = 0; i < _limit; i++) {
                if ((x = swp(0, hash0(x.key), x)).empty) {
                    _size++;
                    check_load();
                    return true;
                } else if ((x = swp(1, hash1(x.key), x)).empty) {
                    _size++;
                    check_load();
                    return true;
                }
            }
        }
        resize();
        return add(x.key, x.value);
    }

    bool remove(const K& key) {
        if (!contains(key)) return false;
        __transaction_atomic {
            for (int i = 0; i < 2; i++) {
                size_t hash = i == 0 ? hash0(key) : hash1(key);
                if (_tables[i][hash].key == key && !_tables[i][hash].empty) {
                    _tables[i][hash] = hash_entry<K, V>{};
                    _size--;
                    return true;
                }
            }
        }
        return false;
    }

    bool contains(const K& key) const {
        size_t h0 = hash0(key); size_t h1 = hash1(key);
        __transaction_atomic {
            return (_tables[0][h0].key == key && !_tables[0][h0].empty) ||
                   (_tables[1][h1].key == key && !_tables[1][h1].empty);
        }
    }

    size_t size() {
        return _size;
    }

    void populate(size_t n, size_t max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, max_val);
        for (size_t i = 0; i < n; i++)
            add(dis(gen), dis(gen));
    }

private:
    std::vector<hash_entry<K, V>> _tables[2];
    size_t _capacity;
    size_t _size;
    float _load_factor;
    static constexpr int _limit = 4;

    size_t hash0(const K& key) const {
        return std::hash<K>{}(key) % _capacity;
    }

    size_t hash1(const K& key) const {
        return (std::hash<K>{}(key) ^ 0x9E3779B9) % _capacity;
    }

    hash_entry<K, V> swp(int table_index, size_t hash, hash_entry<K, V>& entry) {
        std::swap(_tables[table_index][hash], entry);
        return entry;
    }

    // resize if need be
    void check_load() {
        __transaction_atomic {
            if (static_cast<float>(_size) / static_cast<float>(_capacity) > _load_factor)
                resize();
        }
    }

    void resize() transaction_safe {
        size_t new_capacity = _capacity * 2;
        std::vector<hash_entry<K, V>> new_table0(new_capacity);
        std::vector<hash_entry<K, V>> new_table1(new_capacity);
        __transaction_atomic {
            for (int i = 0; i < 2; i++) {
                for (size_t j = 0; j < _tables[i].size(); j++) {
                    auto& entry = _tables[i][j];
                    if (!entry.empty) {
                        size_t hash = i == 0 ? hash0(entry.key) : hash1(entry.key);
                        if (new_table0[hash].empty) {
                            new_table0[hash] = entry;
                        } else {
                            new_table1[hash] = entry;
                        }
                    }
                }
            }
            _num_resize++;
            _capacity = new_capacity;
            _tables[0] = std::move(new_table0);
            _tables[1] = std::move(new_table1);
        }
    }
};

#endif
