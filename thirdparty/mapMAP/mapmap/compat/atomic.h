#ifndef MAPMAP_COMPAT_ATOMIC_H
#define MAPMAP_COMPAT_ATOMIC_H

#include <atomic>

namespace mapmap {

/* Compatibility shim for the atomic API used by the original mapMAP source. */
template <typename T>
class atomic : public std::atomic<T> {
public:
    atomic() noexcept : std::atomic<T>(T{}) {}
    atomic(T value) noexcept : std::atomic<T>(value) {}
    atomic(const atomic& other) noexcept
        : std::atomic<T>(other.load(std::memory_order_relaxed)) {}

    atomic& operator=(const atomic& other) noexcept {
        this->store(other.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
        return *this;
    }

    T compare_and_swap(T value, T comparand) {
        T expected = comparand;
        this->compare_exchange_strong(expected, value);
        return expected;
    }

    T fetch_and_store(T value) {
        return this->exchange(value);
    }

    T fetch_and_increment() {
        return this->fetch_add(1);
    }

    T fetch_and_decrement() {
        return this->fetch_sub(1);
    }
};

} // namespace mapmap

#endif
