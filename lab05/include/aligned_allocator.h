#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>

template <typename T, std::size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T1, std::size_t A1, typename T2, std::size_t A2>
constexpr bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept {
    return A1 == A2;
}

template <typename T1, std::size_t A1, typename T2, std::size_t A2>
constexpr bool operator!=(const AlignedAllocator<T1, A1>& a, const AlignedAllocator<T2, A2>& b) noexcept {
    return !(a == b);
}