#ifndef ODECRAFT_DOPRI_HPP
#define ODECRAFT_DOPRI_HPP

/**
 * @file DOPRI.hpp
 * @brief Machinery shared by the adaptive Runge-Kutta steppers.
 *
 * Scratch storage, the Butcher-tableau holder, and the method-agnostic step-size control and
 * dense-output assembly used by RK23, RK45 and DOP853. The solvers themselves live in
 * RK23_DOPRI.hpp and RK45_DOPRI.hpp; this header carries nothing specific to either.
 */

#include <odecraft/Core/RichSolver/RichBase.hpp>




namespace ode::detail{

template<typename T, size_t NSYS, size_t NCOUNT>
class StaticRKScratch{

public:

    StaticRKScratch(size_t nsys){
        assert(nsys == NSYS && "RKScratch: nsys must match template parameter NSYS for fixed-size systems.");
    }

    std::array<T, NSYS*NCOUNT> stage_scratch() const {return std::array<T, NSYS*NCOUNT>{};}
    std::array<T, NSYS> rhs_scratch() const {return std::array<T, NSYS>{};}
    std::array<T, NSYS> fsal_scratch() const {return std::array<T, NSYS>{};}
    std::array<T, NSYS+2> state_scratch() const {return std::array<T, NSYS+2>{};}

};


// Heap-backed stage scratch: runtime-sized systems, and any scalar that is not trivially
// constructible/copyable (mpfr::mpreal and friends), where a fresh stack array per step would
// mean constructing and destroying nsys*NCOUNT heap-owning objects on every attempt.
template<typename T, size_t NSYS, size_t NCOUNT>
class DynamicRKScratch{

public:

    DynamicRKScratch(size_t nsys) : stage_scratch_(nsys * NCOUNT), rhs_scratch_(nsys),
                                    fsal_scratch_(nsys), state_scratch_(nsys + 2) {
        assert(nsys > 0 && "RKScratch: nsys must be greater than zero.");
    }

    std::vector<T>& stage_scratch() const {return stage_scratch_;}
    std::vector<T>& rhs_scratch() const {return rhs_scratch_;}
    std::vector<T>& fsal_scratch() const {return fsal_scratch_;}
    std::vector<T>& state_scratch() const {return state_scratch_;}

private:
    mutable std::vector<T> stage_scratch_;
    mutable std::vector<T> rhs_scratch_;
    mutable std::vector<T> fsal_scratch_;
    mutable std::vector<T> state_scratch_;

};

// Same rule as the solver's own scratch (see scratch_is_static in BaseSolver.hpp): automatic
// storage only for a compile-time size and a trivial scalar; everything else is heap-backed
// and allocated once.
template<typename T, size_t NSYS, size_t NCOUNT>
using RKScratchSpace = std::conditional_t<scratch_is_static<T, NSYS>,
                                          StaticRKScratch<T, NSYS, NCOUNT>,
                                          DynamicRKScratch<T, NSYS, NCOUNT>>;

// A Butcher-tableau constant (A, B, C, E, P, ...) as a class member: a static constexpr
// table (shared, zero per-instance storage) when T is arithmetic, since it is then a
// compile-time constant; otherwise (e.g. T = mpfr::mpreal, which cannot be constant-evaluated)
// a plain instance member computed once at construction. Either way it's used the same:
// Coef(1,0), Coef[i], Coef.data().
template<typename T, auto Generator>
struct StaticCoefTable{
    using Matrix = decltype(Generator());
    static constexpr Matrix table = Generator();

    template<typename... Idx>
    inline constexpr const auto& operator()(Idx... idx) const { return table(idx...); }
    inline constexpr const auto& operator[](size_t i) const { return table[i]; }
    inline constexpr const auto* data() const { return table.data(); }
};

template<typename T, auto Generator>
struct DynamicCoefTable{
    using Matrix = decltype(Generator());
    Matrix table = Generator();

    template<typename... Idx>
    inline const auto& operator()(Idx... idx) const { return table(idx...); }
    inline const auto& operator[](size_t i) const { return table[i]; }
    inline const auto* data() const { return table.data(); }
};

template<typename T, auto Generator>
using CoefTable = std::conditional_t<std::is_arithmetic_v<T>, StaticCoefTable<T, Generator>, DynamicCoefTable<T, Generator>>;


// ============================================================================
// Shared adaptive Runge-Kutta building blocks, used by RK23, RK45 and DOP853.
// The per-stage arithmetic itself (e.g. h * (a21*K0 + ...)) stays hardcoded in each
// solver's step_impl for performance; only the surrounding, method-agnostic machinery
// (step-size control, dense-output coefficient assembly) is shared here.
// ============================================================================

/// derivatives K (Nstages+1 rows) and interpolation weights P (Nstages+1 x order).
template<typename T>
void rk_interp_matrix(T* coef_mat, const T* K, const T* K0, const T* KF, const T* P, size_t Nstages, size_t order, size_t n);

/// @brief Shared step-size control loop: repeatedly calls step_fn(res, state, h) -> err_norm,
/// halving/growing habs until the local error is accepted (mirrors scipy/boost step control).
template<typename T, typename StepFn>
StepResult rk_adapt_step(T* res, const T* state, size_t n,
                          const T& min_step, const T& max_step, const T& min_step_abs,
                          const T& safety, const T& max_factor, const T& min_factor,
                          const T& err_exp, const T& inc_exp, const T& min_err,
                          int direction, StepFn&& step_fn);

} // namespace ode::detail



#endif // ODECRAFT_DOPRI_HPP
