#include <optional.h>
#include <SiKV.h>
#include <vector>
#include <glue/Set.h>

#pragma once

namespace sikv {
namespace glue {

template<typename K, typename V>
struct Log {

    SIKV_HOST Log() {}

    Log(const Log& other) : s{other.s.start, other.s.end}, keys(other.keys), values(other.values) {

    }

    SIKV_HOST Log(Log&& other) noexcept : s{other.s.start, other.s.end}, keys(std::move(other.keys)), values(std::move(other.values)) {

    }

    SIKV_HOST Log& operator=(Log&& other) noexcept {
        s = {other.s.start, other.s.end};
        keys = std::move(other.keys);
        values = std::move(other.values);
        return *this;
    }

    Set s;

    std::vector<K> keys;
    // nullptr means delete
    std::vector<V*> values;

    // concurrent if 
    SIKV_HOST bool concurrent(const Log<K,V>& l) {
        return (s.start <= l.s.end && s.start >= l.s.start) || (s.end >= l.s.start && s.end <= l.s.end); 
    }

    SIKV_HOST Set intersect(const Log<K,V>& l) {
        return s.intersect(l.s);
    }

};

} // namespace glue
} // namespace sikv
