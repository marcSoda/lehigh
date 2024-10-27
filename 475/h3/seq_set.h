#ifndef SEQSET_H_
#define SEQSET_H_

#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <random>
#include "common.h"

template <typename K, typename V>
class seq_set {
public:
    int _num_resize = 0; // todo: remove? or useful for bench?

    seq_set(size_t initial_capacity = 256, float load_factor = 0.5)
    : _capacity(initial_capacity), _size(0), _load_factor(load_factor),
        _tables{std::vector<hash_entry<K, V>>(), std::vector<hash_entry<K, V>>()} {
        _tables[0].resize(_capacity);
        _tables[1].resize(_capacity);
    }

    bool add(const K& key, const V& value) {
        if (contains(key))
            return false;
        hash_entry<K, V> x(key, value);
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
        resize();
        return add(x.key, x.value);
    }

    bool remove(const K& key) {
        for (int i = 0; i < 2; i++) {
            size_t hash = i == 0 ? hash0(key) : hash1(key);
            if (_tables[i][hash].key == key && !_tables[i][hash].empty) {
                _tables[i][hash] = hash_entry<K, V>{};
                _size--;
                return true;
            }
        }
        return false;
    }

    bool contains(const K& key) const {
        return (_tables[0][hash0(key)].key == key && !_tables[0][hash0(key)].empty) ||
               (_tables[1][hash1(key)].key == key && !_tables[1][hash1(key)].empty);
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
        if (static_cast<float>(_size) / static_cast<float>(_capacity) > _load_factor)
            resize();
    }

    void resize() {
        _num_resize++;
        _capacity *= 2;
        std::vector<hash_entry<K, V>> old_table0 = move(_tables[0]);
        std::vector<hash_entry<K, V>> old_table1 = move(_tables[1]);
        _tables[0].clear();
        _tables[1].clear();
        _tables[0].resize(_capacity);
        _tables[1].resize(_capacity);
        _size = 0;

        for (const auto& entry : old_table0)
            if (!entry.empty) add(entry.key, entry.value);

        for (const auto& entry : old_table1)
            if (!entry.empty) add(entry.key, entry.value);
    }
};

#endif // SEQSET_H_
