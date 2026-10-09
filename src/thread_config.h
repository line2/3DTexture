#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <thread>

inline bool thread_count_overridden() {
    return std::getenv("3DTEXTURE_THREADS") != nullptr;
}

inline int configured_thread_count() {
    int thread_count = static_cast<int>(std::thread::hardware_concurrency());
    if (const char* value = std::getenv("3DTEXTURE_THREADS")) {
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (errno == 0 && end != value && *end == '\0' &&
            parsed >= 1 && parsed <= 1024) {
            thread_count = static_cast<int>(parsed);
        }
        else {
            thread_count = 1;
        }
    }
    return std::max(1, thread_count);
}
