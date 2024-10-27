/**
 * @file
 */
#include <SiKV.h>

#pragma once

namespace cuda {
namespace std {

struct nullopt_t {
    explicit constexpr nullopt_t(int) {}
};

inline constexpr nullopt_t nullopt{0};

template<typename T>
struct optional {

    SIKV_HOST_DEVICE constexpr optional() : value{}, isNull{true} {}

    SIKV_HOST_DEVICE constexpr optional(nullopt_t) noexcept : value{}, isNull{true} {}

    SIKV_HOST_DEVICE constexpr optional(T&& x) noexcept : value{x}, isNull{false} {}
    
    SIKV_HOST_DEVICE constexpr optional(const T& x) : value{x}, isNull{false} {}

    //SIKV_HOST_DEVICE constexpr optional(const optional<T>& other) : value{other.value}, isNull{other.isNull} {}

    //SIKV_HOST_DEVICE constexpr optional(optional<T>&& other) : value{other.value}, isNull{other.isNull} {}

    //template<typename U>
    //SIKV_HOST_DEVICE constexpr optional(const optional<U>& other) : value(other.value), isNull(other.isNull) {}

    //template<typename U>
    //SIKV_HOST_DEVICE constexpr optional(optional<U>&& other) : value(other.value), isNull(other.isNull) {}
    
    SIKV_HOST_DEVICE constexpr optional& operator=(nullopt_t) noexcept {
        value = T{};
        isNull = true;
        return *this;
    }

    //SIKV_HOST_DEVICE constexpr optional& operator=(const optional<T>& other) {
    //    value = other.value;
    //    isNull = other.isNull;
    //    return *this;
    //}

    //SIKV_HOST_DEVICE constexpr optional& operator=(optional<T>&& other) noexcept {
    //    value = other.value;
    //    isNull = other.isNull;
    //    return *this;
    //}

    //template<typename U>
    //SIKV_HOST_DEVICE constexpr optional& operator=(const optional<U>& other) {
    //    value = other.value;
    //    isNull = other.isNull;
    //    return *this;
    //}

    //template<typename U>
    //SIKV_HOST_DEVICE constexpr optional& operator=(optional<U>&& other) noexcept {
    //    value = other.value;
    //    isNull = other.isNull;
    //    return *this;
    //}

    SIKV_HOST_DEVICE const T* operator->() const noexcept {
        return &value;
    }

    SIKV_HOST_DEVICE T* operator->() noexcept {
        return &value;
    }

    SIKV_HOST_DEVICE T& operator*() noexcept {
        return value;
    }

    SIKV_HOST_DEVICE const T& operator*() const noexcept {
        return value;
    }

    SIKV_HOST_DEVICE constexpr bool operator==(nullopt_t) const noexcept {
        return isNull;
    }

    SIKV_HOST_DEVICE constexpr bool operator!=(nullopt_t) const noexcept {
        return !isNull;
    }

    //SIKV_HOST_DEVICE constexpr bool operator==(const optional<T>& rhs) const noexcept {
    //    bool bothNull = isNull && rhs.isNull;
    //    bool bothNotNull = !isNull && !rhs.isNull;
    //    return bothNull || (bothNotNull && value == rhs.value);
    //}

    //SIKV_HOST_DEVICE constexpr bool operator!=(const optional<T>& rhs) const noexcept {
    //    bool bothNull = isNull && rhs.isNull;
    //    bool bothNotNull = !isNull && !rhs.isNull;
    //    return !(bothNull || (bothNotNull && value == rhs.value));
    //}

private:
    T value;
    bool isNull;
};

}
}
