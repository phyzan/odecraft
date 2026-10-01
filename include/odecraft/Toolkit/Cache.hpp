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


template<typename T>
class MutCache {
public:
    inline explicit MutCache(T& ref) : copy_(ref), original_(ref) {}

    // Non-copyable and non-movable: every live instance writes back to original_ on destruction,
    // so a second one aiming at the same referent would clobber it with a stale copy.
    MutCache(const MutCache&) = delete;
    MutCache(MutCache&&) = delete;
    MutCache& operator=(const MutCache&) = delete;
    MutCache& operator=(MutCache&&) = delete;

    MutCache& operator=(const T& other){
        copy_ = other;
        return *this;
    }

    inline operator T& () {
        return copy_;
    }

    inline ~MutCache(){
        // pass the value back to the original
        original_ = copy_;
    }

private:
    T copy_;
    T& original_;
};


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


template<typename T, size_t N>
class MutCachedArray : public ConstCachedArray<T, N> {
    using Base = ConstCachedArray<T, N>;
public:
    // Base already snapshots the array; nothing left to copy here.
    inline explicit MutCachedArray(T* array) : Base(array), original(array) {}

    // Non-copyable and non-movable, for the same reason as MutCache.
    MutCachedArray(const MutCachedArray&) = delete;
    MutCachedArray(MutCachedArray&&) = delete;
    MutCachedArray& operator=(const MutCachedArray&) = delete;
    MutCachedArray& operator=(MutCachedArray&&) = delete;

    inline const T& operator[](size_t i) const {
        return this->copy[i];
    }

    inline T& operator[](size_t i) {
        assert(i < N && "Array out of bounds");
        return this->copy[i];
    }

    inline T* data() {
        return this->copy.data();
    }

    inline const T* data() const {
        return this->copy.data();
    }

    inline ~MutCachedArray(){
        std::copy(this->copy.begin(), this->copy.end(), original);
    }

private:
    T* original;
};

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


} // namespace ode


#endif // ODECRAFT_CACHE_HPP