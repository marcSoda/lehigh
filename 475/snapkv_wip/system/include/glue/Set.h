#include <SiKV.h>
#include <cstdint>
#include <algorithm>

#pragma once

namespace sikv {
namespace glue {

using time_t = uint64_t;

struct Set {

    SIKV_HOST_DEVICE Set() {}

    SIKV_HOST_DEVICE Set(time_t s, time_t e) : start(s), end(e) {}

    time_t start;
    time_t end;

    SIKV_HOST Set intersect(const Set& l) const {
        SPDLOG_DEBUG("Found intersection {} to {}", std::max(start, l.start), std::min(end, l.end));
        return Set{std::max(start, l.start), std::min(end, l.end)};
    }

    SIKV_HOST bool isNull() const {
        return end <= start;
    }
};

} // namespace glue
} // namespace sikv
