#ifndef ODECRAFT_SCRATCH_HPP
#define ODECRAFT_SCRATCH_HPP

#include <tuple>
#include <type_traits>
#include <utility>


namespace ode{

/// @brief What a function works on in place of one of its outputs: a local copy when T is trivially
/// copyable, which the compiler knows aliases nothing, otherwise a reference to the output itself.
/// The mutable counterpart of cache_t.
template<typename T>
using scratch_t = std::conditional_t<std::is_trivially_copyable_v<T>, T, T&>;


namespace detail{

// Calls f when it goes out of scope. It cannot be copied, so f runs exactly once.
template<typename F>
class scope_exit{
public:
    explicit scope_exit(F f) : f_(std::move(f)) {}
    scope_exit(const scope_exit&) = delete;
    scope_exit& operator=(const scope_exit&) = delete;
    ~scope_exit(){ f_(); }
private:
    F f_;
};

// A copied slot is handed back to its item. A referenced one already is the item.
template<typename T>
void copy_back(T& item, const T& slot){
    if constexpr (!std::is_reference_v<scratch_t<T>>){
        item = slot;
    }
}

} // namespace ode::detail


/// @brief Opens a scratch scope over `items...`, holding a scratch_t of each to work on in their place.
/// Pair it with make_write_back, which hands the copies back to the items when the scope ends.
/// @code
/// void f(T& t_, State& q_){
///     auto  scratch    = make_scratch(t_, q_);
///     auto& [t, q]     = scratch;
///     auto  write_back = make_write_back(scratch, t_, q_);
///     // ... work on t and q only
/// }
/// @endcode
template<typename... U>
[[nodiscard]] std::tuple<scratch_t<U>...> make_scratch(U&... items){
    static_assert((!std::is_const_v<U> && ...), "make_scratch writes back into its items, so none of them may be const");
    return {items...};
}


/// @brief Writes `scratch` back into `items...`, the items it was made from, when the scope ends.
template<typename... S, typename... U>
[[nodiscard]] auto make_write_back(std::tuple<S...>& scratch, U&... items){
    static_assert(std::is_same_v<std::tuple<S...>, std::tuple<scratch_t<U>...>>,
                  "make_write_back takes the items of make_scratch, in the same order");
    return detail::scope_exit([&scratch, &items...]{
        std::apply([&items...](const auto&... slots){
            (detail::copy_back(items, slots), ...);
        }, scratch);
    });
}

} // namespace ode

#endif // ODECRAFT_SCRATCH_HPP
