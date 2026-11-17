#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <odecraft/Events/Events.hpp>
#include <odecraft/Contracts/DenseDescriptor.hpp>


/**
 * @brief Declare ode::tags::NAME, the tag of the stepper class template STEPPER: detail::StepperTag<STEPPER, "NAME">.
 *
 * A tag is how a stepper is chosen: passed by value to the factories (make_solver, ODE, ...)
 * for a choice at compile time, or listed in a Catalog for a choice at runtime. Tags are types,
 * so the set of steppers is open: one of your own is selectable like the built-in ones once its
 * tag is declared, and it can be put in a Catalog next to them.
 *
 * STEPPER must be a class template STEPPER<T, SD, OdeType, SolverPolicy SP, typename Derived> built
 * on the solver bases, and NAME is the name its solvers report through method(). The stepper need
 * not be in `ode`, but its tag always is, in ode::tags: use this at global scope, after the stepper,
 * which it names in full, e.g. ODECRAFT_DECLARE_STEPPER(Midpoint, mylib::Midpoint). Since all tags
 * share ode::tags, two steppers cannot be declared under the same NAME.
 *
 * It also declares the overload of ode::detail::stepper_tag through which a solver finds the tag
 * of its stepper, by argument-dependent lookup.
 */
#define ODECRAFT_DECLARE_STEPPER(NAME, STEPPER)                                                             \
namespace ode::tags {                                                                                       \
using NAME = ::ode::detail::StepperTag<STEPPER, #NAME>;                                                     \
}                                                                                                           \
namespace ode::detail {                                                                                     \
template<typename T, typename SD, typename OdeType, ::ode::SolverPolicy SP, typename Derived>              \
::ode::tags::NAME stepper_tag(const STEPPER<T, SD, OdeType, SP, Derived>*, ::ode::detail::adl_tag);        \
}

namespace ode {

enum class UtilPolicy : std::uint8_t{
    Virtual,
    RichVirtual,
};

enum class SolverPolicy : std::uint8_t{
    Static,
    RichStatic,
    Virtual,
    RichVirtual,
};

template<typename T, typename SD>
class OdeSolver;

template<typename T, typename SD>
class OdeRichSolver;

namespace detail{

template<typename T, typename SD, UtilPolicy UP>
struct SolverBoxSelector{
    using type = void;
};

template<typename T, typename SD>
struct SolverBoxSelector<T, SD, UtilPolicy::RichVirtual>{
    using type = pbox::Box<OdeRichSolver<T, SD>>;
};

template<typename T, typename SD>
struct SolverBoxSelector<T, SD, UtilPolicy::Virtual>{
    using type = pbox::Box<OdeSolver<T, SD>>;
};

} // namespace ode::detail


template<typename T, typename SD, UtilPolicy UP>
using BoxedSolver = detail::SolverBoxSelector<T, SD, UP>::type;

template<SolverPolicy SP>
constexpr bool is_rich = (SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual);


namespace detail{

/// @brief A string literal as a template argument, e.g. StepperTag<RK45, "RK45">.
template<size_t N>
struct fixed_string{
    std::array<char, N> data = {};

    // From the literal itself, so that N is deduced from it
    constexpr fixed_string(const char (&str)[N]) {
        std::copy_n(str, N, data.begin());
    }

    constexpr operator std::string_view() const {
        return {data.data(), N - 1};
    }
};


/**
 * @brief The tag of the stepper class template Stepper, called Name.
 *
 * ODECRAFT_DECLARE_STEPPER(NAME, STEPPER) declares ode::tags::NAME as StepperTag<STEPPER, "NAME">.
 */
template<template<typename, typename, typename, SolverPolicy, typename> typename Stepper, fixed_string Name>
struct StepperTag{
    static constexpr std::string_view name = Name;

    template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived = void>
    using type = Stepper<T, SD, OdeType, SP, Derived>;
};


/// @brief Whether S is a stepper tag, such as the ones ODECRAFT_DECLARE_STEPPER declares.
template<typename S>
concept isStepper = requires {
    { S::name } -> std::convertible_to<std::string_view>;
};

} // namespace ode::detail


/**
 * @brief A list of stepper tags, for choosing a stepper at runtime by its Id or its name.
 *
 * Any tags can be listed, those of user-defined steppers included, so a catalog is the
 * extensible counterpart of a fixed enum. ode::Steppers is the one of the built-in steppers.
 */
template<typename... Stepper>
class Catalog {
    
    template<typename /*Stepper*/>
    struct tag {};

    template<typename S, UtilPolicy UP, typename SD, typename OdeType, typename T, typename... Args>
    static BoxedSolver<T, SD, UP> construct(OdeType rhs, T t0, Args&&... args) {
        static_assert(std::constructible_from<S, OdeType, T, Args...>, "Stepper not constructible from these arguments");
        return std::make_unique<S>(std::move(rhs), std::move(t0), std::forward<Args>(args)...);
    }

    template<typename S, typename ReturnType, typename Callable>
    static ReturnType call_with(Callable&& callable) {
        return std::forward<Callable>(callable).template operator()<S>();
    }

    // The position of S in the list
    template<typename S>
    static constexpr size_t index_of() {
        constexpr std::array hit{std::is_same_v<tag<S>, tag<Stepper>>...};
        static_assert(std::ranges::count(hit, true) == 1, "Stepper must appear exactly once in the catalog");
        return static_cast<size_t>(std::ranges::find(hit, true) - hit.begin());
    }

public:

    /**
     * @brief The id of a stepper of this catalog, which wraps its position in the list.
     *
     * Only the catalog makes ids, through id<S> and getID, so an id always stands for one of its
     * steppers, and make, getName and from_stepper need no bounds check. Neither an integer nor
     * the id of another catalog converts to it: each catalog has an Id type of its own.
     */
    class Id {
        friend Catalog;

        size_t index_;

        constexpr explicit Id(size_t i) : index_(i) {}

    public:
        /// @brief The position of the stepper in the catalog.
        constexpr size_t index() const { return index_; }

        constexpr bool operator==(const Id&) const = default;
    };

    // Compile-time id of a stepper tag. For a case label, switch on index(): case id<S>.index().
    template<typename S>
    static constexpr Id id = Id(index_of<S>());

    // Runtime choice: builds the stepper with id `which`, instantiated for <T, SD, OdeType, SP>.
    template<UtilPolicy UP, typename SD, typename T, typename OdeType, typename... Args>
    static BoxedSolver<T, SD, UP> make(Id which, OdeType ode, T t0, Args&&... args) {
        constexpr SolverPolicy SP = UP == UtilPolicy::RichVirtual ? SolverPolicy::RichVirtual : SolverPolicy::Virtual;
        static constexpr std::array table{&construct<typename Stepper::template type<T, SD, OdeType, SP>, UP, SD, OdeType, T, Args...>...};
        return table[which.index()](std::move(ode), std::move(t0), std::forward<Args>(args)...);
    }

    // Id -> string
    static std::string_view getName(Id stepper){
        static constexpr std::array Names{Stepper::name...};
        return Names[stepper.index()];
    }

    // string -> Id
    static Id getID(std::string_view name){
        // v counts the entries before the first one with this name
        size_t v{};
        if (!((Stepper::name == name || (++v, false)) || ...)){
            throw std::invalid_argument("Unknown integrator: " + std::string(name));
        }
        return Id(v);
    }

    // Calls callable.template operator()<S>(), where S is the stepper tag with id `stepper`.
    template<typename ReturnType, typename Callable>
    static ReturnType from_stepper(Id stepper, Callable&& callable) {
        static constexpr std::array table{&call_with<Stepper, ReturnType, Callable>...};
        return table[stepper.index()](std::forward<Callable>(callable));
    }

};


template<typename SD, SolverPolicy SP, typename Stepper, typename T, typename OdeType>
inline auto getSolver(Stepper /*stepper*/, OdeType ode, T t0, state_t<T, SD> q0, real_state_t<T, SD> rtol, real_state_t<T, SD> atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, state_t<T, SD>> events = {}){
    return typename Stepper::template type<T, SD, OdeType, SP>(std::move(ode), std::move(t0), std::move(q0), std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir, std::move(events));
}


template<UtilPolicy UP, typename SD, typename Stepper, typename T, typename OdeType, typename... Args>
inline BoxedSolver<T, SD, UP> make_solver(Stepper /*stepper*/, OdeType ode, T t0, state_t<T, SD> q0, Args&&... args){
    constexpr SolverPolicy SP = UP == UtilPolicy::RichVirtual ? SolverPolicy::RichVirtual : SolverPolicy::Virtual;
    using Solver = Stepper::template type<T, SD, OdeType, SP>;
    return pbox::make_box<Solver>(std::move(ode), std::move(t0), std::move(q0), std::forward<Args>(args)...);
}

template<typename SD, typename Stepper, typename T, typename OdeType, typename... Args>
inline BoxedSolver<T, SD, UtilPolicy::Virtual> make_vsolver(Stepper /*method*/, OdeType ode, T t0, state_t<T, SD> q0, Args&&... args){
    return make_solver<UtilPolicy::Virtual, SD>(Stepper{}, std::move(ode), std::move(t0), std::move(q0), std::forward<Args>(args)...);
}

template<typename SD, typename Stepper, typename T, typename OdeType, typename... Args>
inline BoxedSolver<T, SD, UtilPolicy::RichVirtual> make_rich_vsolver(Stepper /*stepper*/, OdeType ode, T t0, state_t<T, SD> q0, Args&&... args){
    return make_solver<UtilPolicy::RichVirtual, SD>(Stepper{}, std::move(ode), std::move(t0), std::move(q0), std::forward<Args>(args)...);
}

} // namespace ode
