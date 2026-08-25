#ifndef ODECRAFT_SOLVER_FACTORY_HPP
#define ODECRAFT_SOLVER_FACTORY_HPP

#include <cstdint>
#include <stdexcept>
#include <odecraft/Core/Events.hpp>

#define ODECRAFT_TEMPLATE template<typename T, size_t, SolverPolicy, hasRhsFunc<T>, typename>

namespace ode {

enum class UtilPolicy : std::uint8_t{ Virtual, RichVirtual};

enum class SolverPolicy : std::uint8_t{ Static, RichStatic, Virtual, RichVirtual};

template<typename T, size_t N>
class OdeSolver;

template<typename T, size_t N>
class OdeRichSolver;


enum class Integrator : uint8_t{
    Custom,
    Euler,
    RK4,
    RK23,
    RK45,
    DOP853,
    BDF
};

namespace detail{

template<typename T, size_t N, UtilPolicy UP>
struct SolverBoxSelector{
    using type = void;
};

template<typename T, size_t N>
struct SolverBoxSelector<T, N, UtilPolicy::RichVirtual>{
    using type = pbox::Box<OdeRichSolver<T, N>>;
};

template<typename T, size_t N>
struct SolverBoxSelector<T, N, UtilPolicy::Virtual>{
    using type = pbox::Box<OdeSolver<T, N>>;
};

template<Integrator M, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
struct SolverTypeGetter{
    using type = void;
};

} // namespace ode::detail


template<typename T, size_t N, UtilPolicy UP>
using BoxedSolver = typename detail::SolverBoxSelector<T, N, UP>::type;

template<SolverPolicy SP>
constexpr bool is_rich = (SP == SolverPolicy::RichStatic || SP == SolverPolicy::RichVirtual);




template<typename ReturnType, typename Callable, typename... Args>
inline ReturnType choose_integrator_case(Integrator method, Callable&& callable, Args&&... args){
    switch (method){
        case Integrator::Euler:
            return callable.template operator()<Integrator::Euler>(std::forward<Args>(args)...);
        case Integrator::RK4:
            return callable.template operator()<Integrator::RK4>(std::forward<Args>(args)...);
        case Integrator::RK23:
            return callable.template operator()<Integrator::RK23>(std::forward<Args>(args)...);
        case Integrator::RK45:
            return callable.template operator()<Integrator::RK45>(std::forward<Args>(args)...);
        case Integrator::DOP853:
            return callable.template operator()<Integrator::DOP853>(std::forward<Args>(args)...);
        case Integrator::BDF:
            return callable.template operator()<Integrator::BDF>(std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unknown integrator enum value");
    }
}

template<ODECRAFT_TEMPLATE typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (is_rich<SP>)
inline Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events = {}) {
    return Solver<T, N, SP, OdeType, void>(std::move(ode), t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(events));
}

template<ODECRAFT_TEMPLATE typename Solver, SolverPolicy SP, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!is_rich<SP>)
inline Solver<T, N, SP, OdeType, void> getSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1) {
    return Solver<T, N, SP, OdeType, void>(std::move(ode), t0, q0, rtol, atol, min_step, max_step, stepsize, dir);
}


template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
inline BoxedSolver<T, N, UP> make_solver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args) {
    constexpr SolverPolicy SP = UP == UtilPolicy::RichVirtual ? SolverPolicy::RichVirtual : SolverPolicy::Virtual;
    return choose_integrator_case<BoxedSolver<T, N, UP>>(method,
        [&]<Integrator M>(){
            using Solver = typename detail::SolverTypeGetter<M, T, N, SP, OdeType>::type;
            return pbox::make_box<Solver>(std::move(ode), t0, q0, std::forward<Args>(args)...);
        }
    );
}

template<typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
inline BoxedSolver<T, N, UtilPolicy::Virtual> make_vsolver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args) {
    return make_solver<UtilPolicy::Virtual>(method, std::move(ode), t0, q0, std::forward<Args>(args)...);
}

template<typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
inline BoxedSolver<T, N, UtilPolicy::RichVirtual> make_rich_vsolver(Integrator method, OdeType ode, T t0, View1D<T, N> q0, Args&&... args) {
    return make_solver<UtilPolicy::RichVirtual>(method, std::move(ode), t0, q0, std::forward<Args>(args)...);
}

} // namespace ode

#endif // ODECRAFT_SOLVER_FACTORY_HPP