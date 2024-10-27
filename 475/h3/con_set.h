#ifndef CONSET_H_
#define CONSET_H_

#include <atomic>
#include <iostream>
#include <ostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <functional>
#include <random>
#include "common.h"

const int THRESHOLD = 2;
const int PROBE_SIZE = 4;
const int LIMIT = 8;

template <typename K, typename V>
class con_set {
private:
    int _capacity;
    std::vector<hash_entry<K, V>>** _tables;
    std::recursive_mutex** _locks;
    std::mutex _rsz_mtx;

    size_t hash0(const K& key) const {
        return std::hash<K>{}(key);
    }

    size_t hash1(const K& key) const {
        return (std::hash<K>{}(key) ^ 0x9E3779B9);
    }

    void acquire(K x) {
        while (true) {
            if (_locks[0][hash0(x) % _capacity].try_lock()) {
                if (_locks[1][hash1(x) % _capacity].try_lock()) break;
                else _locks[0][hash0(x) % _capacity].unlock();
            }
            std::this_thread::yield();
        }
    }

    void release(K x) {
        _locks[0][hash0(x) % _capacity].unlock();
        _locks[1][hash1(x) % _capacity].unlock();
    }

    bool relocate(int i, int hi) {
        int hj = 0;
        int j = 1 - i;
        for (int round = 0; round < LIMIT; round++) {
            std::vector<hash_entry<K, V>>& iSet = _tables[i][hi];
            if (iSet.empty()) return false;
            hash_entry<K, V> y = iSet[0];
            switch (i) {
                case 0: hj = hash1(y.key) % _capacity; break;
                case 1: hj = hash0(y.key) % _capacity; break;
            }
            acquire(y.key);
            std::vector<hash_entry<K, V>>& jSet = _tables[j][hj];
            bool iSetRemoved = false;

            if (!iSet.empty()) {
                auto it = std::find_if(iSet.begin(), iSet.end(), [&](const hash_entry<K, V>& entry) {
                    return !entry.empty && entry.key == y.key;
                });
                if (it != iSet.end()) {
                    iSet.erase(it);
                    iSetRemoved = true;
                }

            }
            if (iSetRemoved) {
                if (jSet.size() < THRESHOLD) {
                    jSet.push_back(y);
                    release(y.key);
                    return true;
                } else if (jSet.size() < PROBE_SIZE) {
                    jSet.push_back(y);
                    i = 1 - i;
                    hi = hj;
                    j = 1 - j;
                } else {
                    iSet.push_back(y);
                    release(y.key);
                    return false;
                }
            } else if (iSet.size() >= THRESHOLD) {
                release(y.key);
                continue;
            } else {
                release(y.key);
                return true;
            }
            release(y.key);
        }
        return false;
    }

    void resize() {
        if (!_rsz_mtx.try_lock()) return;
        // std::cout << "resize" << std::endl;
        int old_capacity = _capacity;
        int new_capacity = old_capacity * 2;
        std::vector<hash_entry<K, V>>** new_tables = new std::vector<hash_entry<K, V>>*[2];

        for (int i = 0; i < 2; i++) {
            new_tables[i] = new std::vector<hash_entry<K, V>>[new_capacity];
            for (int j = 0; j < new_capacity; j++) {
                new_tables[i][j] = std::vector<hash_entry<K,V>>();
            }
        }

        std::recursive_mutex** new_locks = new std::recursive_mutex*[2];
        for (int i = 0; i < 2; i++) {
            new_locks[i] = new std::recursive_mutex[new_capacity];
        }

        for (int i = 0; i < old_capacity; i++) {
            for (int j = 0; j < 2; j++) {
                _locks[j][i].lock();
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < old_capacity; j++) {
                std::vector<hash_entry<K, V>>& old_set = _tables[i][j];
                for (const auto& entry : old_set) {
                    int h = i == 0 ? hash0(entry.key) % new_capacity : hash1(entry.key) % new_capacity;
                    new_tables[i][h].push_back(entry);
                }
                old_set.clear();
            }
        }

        std::swap(_tables, new_tables);
        std::swap(_locks, new_locks);
        _capacity = new_capacity;

        for (int i = 0; i < old_capacity; i++) {
            for (int j = 0; j < 2; j++) {
                _locks[j][i].unlock();
            }
        }

        for (int i = 0; i < 2; i++) {
            delete[] new_tables[i];
            delete[] new_locks[i];
        }
        delete[] new_tables;
        delete[] new_locks;
        _rsz_mtx.unlock();
        // std::cout << "done" << std::endl;
    }

public:
    con_set(int size) {
        _capacity = size;
        _tables = new std::vector<hash_entry<K, V>>*[2];
        for (int i = 0; i < 2; i++) {
            _tables[i] = new std::vector<hash_entry<K, V>>[_capacity];
            for (int j = 0; j < _capacity; j++) {
                _tables[i][j] = std::vector<hash_entry<K,V>>();
            }
        }
        _locks = new std::recursive_mutex*[2];
        for (int i = 0; i < 2; i++) {
            _locks[i] = new std::recursive_mutex[_capacity];
        }
    }

    ~con_set() {
        for (int i = 0; i < 2; i++) {
            delete[] _tables[i];
            delete[] _locks[i];
        }
        delete[] _tables;
        delete[] _locks;
    }

    bool contains(K x) {
        acquire(x);
        bool found = false;
        std::vector<hash_entry<K, V>>& set0 = _tables[0][hash0(x) % _capacity];
        auto it0 = std::find_if(set0.begin(), set0.end(), [&](const hash_entry<K, V>& entry){
            return !entry.empty && entry.key == x;
        });
        if (it0 != set0.end()) {
            found = true;
        } else {
            std::vector<hash_entry<K, V>>& set1 = _tables[1][hash1(x) % _capacity];
            auto it1 = std::find_if(set1.begin(), set1.end(), [&](const hash_entry<K, V>& entry){
                return !entry.empty && entry.key == x;
            });
            if (it1 != set1.end()) {
                found = true;
            }
        }
        release(x);
        return found;
    }

    bool add(K k, V v) {
        hash_entry<K, V> x(k, v);
        acquire(k);
        if (contains(k)) {
            release(k);
            return false;
        }
        int h0 = hash0(k) % _capacity, h1 = hash1(k) % _capacity;
        int i = -1, h = -1;
        bool mustResize = false;
        std::vector<hash_entry<K, V>>& set0 = _tables[0][h0];
        std::vector<hash_entry<K, V>>& set1 = _tables[1][h1];
        if (set0.size() < THRESHOLD) {
            set0.push_back(x);
            release(k);
            return true;
        } else if (set1.size() < THRESHOLD) {
            set1.push_back(x);
            release(k);
            return true;
        } else if (set0.size() < PROBE_SIZE) {
            set0.push_back(x);
            i = 0;
            h = h0;
        } else if (set1.size() < PROBE_SIZE) {
            set1.push_back(x);
            i = 1;
            h = h1;
        } else
            mustResize = true;

        release(k);

        if (mustResize) {
            while (true) {
                resize();
                if (add(k, v)) break;
            }
        } else if (!relocate(i, h))
            resize();
        return true;
    }

    bool remove(K x) {
        if (!contains(x)) return false;
        acquire(x);
        std::vector<hash_entry<K, V>>& set0 = _tables[0][hash0(x) % _capacity];
        if (set0.size() > 0) {
            for (auto it = set0.begin(); it != set0.end(); it++) {
                if (it->key == x) {
                    set0.erase(it);
                    release(x);
                    return true;
                }
            }
        }
        std::vector<hash_entry<K, V>>& set1 = _tables[1][hash1(x) % _capacity];
        if (set1.size() > 0) {
            for (auto it = set1.begin(); it != set1.end(); it++) {
                if (it->key == x) {
                    set1.erase(it);
                    release(x);
                    return true;
                }
            }
        }
        release(x);
        return false;;
    }

    void populate(size_t n, size_t max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, max_val);
        for (size_t i = 0; i < n; i++)
            add(dis(gen), dis(gen));
    }

    size_t size() const {
        size_t count = 0;
        for (int i = 0; i < _capacity; i++) {
            count += _tables[0][i].size();
            count += _tables[1][i].size();
        }
        return count;
    }
};

#endif
