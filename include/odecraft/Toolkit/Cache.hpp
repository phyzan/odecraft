#ifndef ODECRAFT_CACHE_HPP
#define ODECRAFT_CACHE_HPP

#include "Tools.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>


namespace ode{

namespace detail{


/// @brief Storage half of a cached scalar. Read and written through operator T&.
template<typename T>
class MutCache {
public:
    inline explicit MutCache(const T& value) : copy_(value) {}

    MutCache& operator=(const T& other){
        copy_ = other;
        return *this;
    }

    inline operator T& () {
        return copy_;
    }

    inline operator const T& () const {
        return copy_;
    }
private:
    T copy_;
};


/// @brief Storage half of a cached read-only array.
template<typename T, size_t N>
class ConstCachedArray {
public:
    ConstCachedArray(const T* array) {
        std::copy(array, array+N, copy.data());
    }

    inline const T& operator[](size_t i) const {
        assert(i < N && "Array out of bounds");
        return copy[i];
    }

    inline const T* data() const {
        return copy.data();
    }

protected:
    std::array<T, N> copy;
};


/// @brief Storage half of a cached output array.
template<typename T, size_t N>
class MutCachedArray : public ConstCachedArray<T, N> {
    using Base = ConstCachedArray<T, N>;
public:
    // Base already snapshots the array; nothing left to copy here.
    inline explicit MutCachedArray(const T* array) : Base(array) {}

    inline const T& operator[](size_t i) const {
        return this->copy[i];
    }

    inline T& operator[](size_t i) {
        assert(i < N && "Array out of bounds");
        return this->copy[i];
    }

    operator T*() {
        return this->copy.data();
    }

    operator const T*() const {
        return this->copy.data();
    }
    
    inline T* data() {
        return this->copy.data();
    }

    inline const T* data() const {
        return this->copy.data();
    }

};


/// @brief Write-back half of a cached array: copies the cache back over the array it shadows when
/// it goes out of scope. Non-copyable so the flush happens exactly once.
template<typename T, size_t N>
class WriteBackArray {
public:
    inline WriteBackArray(const T* src, T* dst) : src_(src), dst_(dst) {}

    WriteBackArray(const WriteBackArray&) = delete;
    WriteBackArray(WriteBackArray&&) = delete;
    WriteBackArray& operator=(const WriteBackArray&) = delete;
    WriteBackArray& operator=(WriteBackArray&&) = delete;

    inline ~WriteBackArray(){
        std::copy(src_, src_+N, dst_);
    }

private:
    const T* src_;
    T* dst_;
};


/// @brief Write-back half of a cached scalar.
template<typename T>
class WriteBackScalar {
public:
    inline WriteBackScalar(const T& src, T& dst) : src_(&src), dst_(&dst) {}

    WriteBackScalar(const WriteBackScalar&) = delete;
    WriteBackScalar(WriteBackScalar&&) = delete;
    WriteBackScalar& operator=(const WriteBackScalar&) = delete;
    WriteBackScalar& operator=(WriteBackScalar&&) = delete;

    inline ~WriteBackScalar(){
        *dst_ = *src_;
    }

private:
    const T* src_;
    T* dst_;
};


/// @brief Stand-in for the uncached paths, which work through the caller's storage directly and so
/// have nothing to flush.
struct NoWriteBack {};

} // namespace ode::detail


template<typename T>
inline decltype(auto) const_cache(const T& item){
    if constexpr (std::is_trivially_copyable_v<T>){
        return detail::copy_elem(item);
    } else {
        return detail::ref_elem(item);
    }
}

// For a non-trivially-copyable T, const_cache hands back a reference to its argument, so a
// temporary would dangle at the end of the full expression. Reject rvalues outright rather than
// letting that through for the trivially copyable types and failing silently for the rest.
template<typename T>
void const_cache(T&&) = delete;


/// @brief Caches a mutable scalar. Pair it with cache_write_back on the very next line, or the
/// value is never handed back to `elem`.
template<typename T>
inline decltype(auto) mut_cache(T& elem){
    if constexpr (std::is_trivially_copyable_v<T>){
        return detail::MutCache<T>{elem};
    } else {
        return detail::mut_ref_elem(elem);
    }
}


template<typename T, size_t N>
inline decltype(auto) const_cached_array(const T* array, size_t n){
    assert((N == n || N == 0) && "Invalid array size");
    (void)n;   // only read by the assert in the cached branch
    if constexpr (detail::scratch_is_static<T, N>){
        // The cached form copies exactly N elements, so a mismatch here reads out of bounds.
        return detail::ConstCachedArray<T, N>{array};
    } else {
        return View1D<T, N>{array, n};
    }
}


/// @brief Caches a mutable array. Pair it with cache_write_back on the very next line, or the
/// contents are never handed back to `array`.
template<typename T, size_t N>
inline decltype(auto) mut_cached_array(T* array, size_t n){
    assert((N == n || N == 0) && "Invalid array size");
    (void)n;   // only read by the assert in the cached branch
    if constexpr (detail::scratch_is_static<T, N>){
        return detail::MutCachedArray<T, N>{array};
    } else {
        return MutView<T, ndspan::Layout::C, N>{array, n};
    }
}


/// @brief Flushes a cache built by mut_cached_array back into `array` at the end of the scope.
template<typename T, size_t N, typename Cache>
inline decltype(auto) cache_write_back(Cache& cache, T* array){
    if constexpr (detail::scratch_is_static<T, N>){
        return detail::WriteBackArray<T, N>{cache.data(), array};
    } else {
        (void)cache; (void)array;   // the uncached branch already writes through `array`
        return detail::NoWriteBack{};
    }
}


/// @brief Flushes a cache built by mut_cache back into `elem` at the end of the scope.
template<typename T, typename Cache>
inline decltype(auto) cache_write_back(Cache& cache, T& elem){
    if constexpr (std::is_trivially_copyable_v<T>){
        return detail::WriteBackScalar<T>{cache, elem};
    } else {
        (void)cache; (void)elem;    // the uncached branch already writes through `elem`
        return detail::NoWriteBack{};
    }
}


} // namespace ode


#endif // ODECRAFT_CACHE_HPP
