#ifndef ODECRAFT_EULER_HPP
#define ODECRAFT_EULER_HPP

#include <odecraft/Core/RichBase.hpp>

namespace ode{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class Euler : public detail::BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<Euler<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr int ERR_EST_ORDER = 1;
    static constexpr bool IS_IMPLICIT = false;

    DEFAULT_RULE_OF_FOUR(Euler)

    Euler(OdeType ode, T t0, View1D<T, N> q0, T stepsize, int dir=1) requires (!is_rich<SP>);

    Euler(OdeType ode, T t0, View1D<T, N> q0, T stepsize, int dir=1, EventList<T> events = {}) requires (is_rich<SP>);

    // Constructor signature that follows the main constructor pattern.
    Euler(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1) requires (!is_rich<SP>);

    Euler(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events = {}) requires (is_rich<SP>);

    Stepper method() const;

    auto local_interp() const;

protected:

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

};


namespace detail{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Stepper::Euler, T, N, SP, OdeType, Derived>{
    using type = Euler<T, N, SP, OdeType, Derived>;
};

} // namespace ode::detail

} // namespace ode

#endif // ODECRAFT_EULER_HPP
