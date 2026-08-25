#ifndef ODECRAFT_RUNGEKUTTA_HPP
#define ODECRAFT_RUNGEKUTTA_HPP


#include <odecraft/Core/RichBase.hpp>

namespace ode{

// Compile with ODECRAFT_RK4_DENSE to use rk4 steps for interpolation instead of the default Hermite polynomials
// This increases accuracy at the cost of performance and memory

template<typename T, typename RhsType>
void rk4_step(RhsType&& rhs, T* y_new, const T& t, const T& h, const T* y, T* k, size_t n, T* worker);

template<typename T>
void rk4_interp(T* out, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n);


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class RK4 : public detail::BaseDispatcher<GetDerived<RK4<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<RK4<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    template<typename... Type>
    RK4(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, Type&&... extras);

    Integrator method() const;

    auto  local_interp() const;

    void        Reset();

    static constexpr int            ERR_EST_ORDER = 4;
    static constexpr size_t         INTERP_ORDER = 4;
    static constexpr bool           IS_IMPLICIT = false;

protected:

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

    void        ReAdjust(const T* new_vector);

    void        set_interp_data() const;

    // 4 stages of size N, plus one auxiliary array. if ODECRAFT_RK4_DENSE, K has 4 extra stages for dense output. So visually K = [k1, k2, k3, k4, aux | k1, k2, l3, k4 ]
#ifdef ODECRAFT_RK4_DENSE
    mutable Array2D<T, 9, N>    K;
#else
    mutable Array2D<T, 5, N>    K;
#endif
    mutable bool        interp_data_set = false;

};


namespace detail{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Integrator::RK4, T, N, SP, OdeType, Derived>{
    using type = RK4<T, N, SP, OdeType, Derived>;
};

} // namespace ode::detail


} // namespace ode


#endif // ODECRAFT_RUNGEKUTTA_HPP
