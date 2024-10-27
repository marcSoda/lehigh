#ifndef COMMON_H_
#define COMMON_H_

template <typename K, typename V>
struct hash_entry {
    K key;
    V value;
    bool empty;

    hash_entry() : empty(true) {}
    hash_entry(const K& k, const V& v) : key(k), value(v), empty(false) {}
};

#endif // COMMON_H_
