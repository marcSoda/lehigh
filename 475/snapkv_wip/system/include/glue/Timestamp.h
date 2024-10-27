#include <atomic>

#pragma once

namespace sikv {
namespace glue {

/// Global timestamp
inline std::atomic<time_t> gts; 

} // namespace glue
} // namespace sikv
