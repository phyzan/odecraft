#ifndef ODECRAFT_CACHE_HPP
#define ODECRAFT_CACHE_HPP

#include "Tools.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>


namespace ode{


namespace detail{


template<typename T>
class write_back_scalar_t {
public:

    inline write_back_scalar_t(const T& src, T& dst) : src_(&src), dst_(&dst) {}

    write_back_scalar_t(const write_back_scalar_t&) = delete;
    write_back_scalar_t(write_back_scalar_t&&) = delete;
    write_back_scalar_t& operator=(const write_back_scalar_t&) = delete;
    write_back_scalar_t& operator=(write_back_scalar_t&&) = delete;

    inline ~write_back_scalar_t(){
        *dst_ = *src_;
    }

private:
    const T* src_;
    T* dst_;
};

template<typename T, size_t N>
class write_back_array_t {
public:
    inline write_back_array_t(const T* src, T* dst) : src_(src), dst_(dst) {}

    write_back_array_t(const write_back_array_t&) = delete;
    write_back_array_t(write_back_array_t&&) = delete;
    write_back_array_t& operator=(const write_back_array_t&) = delete;
    write_back_array_t& operator=(write_back_array_t&&) = delete;

    inline ~write_back_array_t(){
        std::copy(src_, src_+N, dst_);
    }

private:
    const T* src_;
    T* dst_;
};

template<typename T, size_t N>
struct no_write_back_t{};

// for const array view, use array_view_t<const T, N>
template<typename T, size_t N>
class array_view_t {
    using storage_t
        = std::conditional_t<
            scratch_is_static<T, N>,
            std::array<std::remove_const_t<T>, N>,
            T*
        >;
    static constexpr bool is_copy = scratch_is_static<T, N>;
public:
    
    // --------------- CONSTRUCTORS ---------------
    inline array_view_t(T* data) requires is_copy {
        std::copy(data, data+N, data_.data());
    }

    inline array_view_t(T* data) requires (!is_copy) : data_(data){}
    // --------------------------------------------


    inline operator const T*() const{
        if constexpr (is_copy) {
            return data_.data();
        } else {
            return data_;
        }
    }

    inline operator T*() {
        if constexpr (is_copy) {
            return data_.data();
        } else {
            return data_;
        }
    }

    template<std::integral Int>
    inline const T& operator[](Int i) const{
        return data_[i];
    }

    template<std::integral Int>
    inline T& operator[](Int i){
        return data_[i];
    }

    inline const T* data() const {
        if constexpr (is_copy) {
            return data_.data();
        } else {
            return data_;
        }
    }

    inline T* data() {
        if constexpr (is_copy) {
            return data_.data();
        } else {
            return data_;
        }
    }

    array_view_t(const array_view_t&) = delete;
    array_view_t(array_view_t&&) = delete;
    array_view_t& operator=(const array_view_t&) = delete;
    array_view_t& operator=(array_view_t&&) = delete;
    ~array_view_t() = default;
private:
    storage_t data_;
};


template<typename T>
using const_cache_storage_t = std::conditional_t<std::is_trivially_copyable_v<T>, T, const T&>;

template<typename T>
using scratch_storage_t = std::conditional_t<std::is_trivially_copyable_v<T>, T, T&>;

// For constant references
template<typename... U>
struct cache_group{

    cache_group(const U&... args) : data{args...} {}

    std::tuple<const_cache_storage_t<U>...> data;
};


} // namespace ode::detail

/// @brief Opens a scratch scope over `args...`: every trivially copyable argument is shadowed by a local copy, and the rest are bound by reference and worked on in place.
/// Pair it with make_write_back on the next line, or nothing is ever handed back to `args...`.
/// @code
/// void f(T& x_, T& y_){
///     auto  scratch = make_scratch(x_, y_);
///     auto& [x, y]  = scratch;
///     [[maybe_unused]] auto write_back_x = make_write_back(x, x_);
///     [[maybe_unused]] auto write_back_y = make_write_back(y, y_);
/// }
/// @endcode
template<typename... U>
[[nodiscard]] inline auto make_scratch(U&... args){
    static_assert((!std::is_const_v<U> && ...),
                  "make_scratch writes back into its arguments, so none of them may be const");
    return std::tuple<detail::scratch_storage_t<U>...>(args...);
}


/// @brief Flushes one cached scalar back into `elem` at the end of the scope. Takes the scratch
template<typename T>
[[nodiscard]] inline auto make_write_back(const T& src, T& elem){
    if constexpr (std::is_trivially_copyable_v<T>){
        return detail::write_back_scalar_t<T>(src, elem);
    } else {
        // The scratch holds a reference for these, so `elem` was written through in place.
        (void)src; (void)elem;
        return detail::no_write_back_t<T, 0>{};
    }
}


/// @brief Read-only counterpart of make_scratch: trivially copyable arguments are copied, the rest are bound by const reference. There is no write-back half, so the values destructure directly.
///
/// @code
/// void f(const T& x_, const T& y_){
///     const auto& [x, y] = make_scratch_view(x_, y_);
/// }
/// @endcode
template<typename... U>
[[nodiscard]] inline auto make_scratch_view(const U&... args){
    return std::tuple<const detail::const_cache_storage_t<U>...>(args...);
}


template<typename... U>
    requires (!(std::is_lvalue_reference_v<U> && ...))
void make_scratch_view(U&&...) = delete;


/// @brief Snapshots a read-only array of N elements, or just points at it when a snapshot is not worthwhile
template<size_t N, typename T>
[[nodiscard]] inline auto cache_array(const T* array, size_t size){
    // The cached form copies exactly N elements, so a mismatch here reads out of bounds.
    assert((N == size || N == 0) && "Invalid array size");
    (void)size;   // only read by the assert
    return detail::array_view_t<const T, N>(array);
}


/// @brief Opens a scratch scope over a mutable array of N elements. One array per call
template<size_t N, typename T>
[[nodiscard]] inline auto cache_mut_array(T* array, size_t size){
    assert((N == size || N == 0) && "Invalid array size");
    (void)size;   // only read by the assert
    return detail::array_view_t<T, N>(array);
}


/// @brief Flushes a view built by cache_mut_array back into `array` at the end of the scope.
template<size_t N, typename T, typename View>
[[nodiscard]] inline auto make_array_write_back([[maybe_unused]] View& view, T* array){
    if constexpr (detail::scratch_is_static<T, N>){
        return detail::write_back_array_t<T, N>(view.data(), array);
    } else {
        // the uncached branch already writes through `array`
        (void)array;
        return detail::no_write_back_t<T, N>{};
    }
}


} // namespace ode


#endif // ODECRAFT_CACHE_HPP
