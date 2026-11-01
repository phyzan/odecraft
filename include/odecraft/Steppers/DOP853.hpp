#pragma once

#include <odecraft/Core/RichSolver/RichBase.hpp>
#include <odecraft/Steppers/DOPRI.hpp>


namespace ode{

// ============================================================================
// Raw Butcher-tableau constants for DOP853 (Hairer, Norsett & Wanner).
// Purely data - no solver logic lives here.
// ============================================================================

template<typename T>
struct DOP_COEFS{

    static constexpr size_t N_STAGES = 12;
    static constexpr size_t N_STAGES_EXT = 16;
    static constexpr size_t INTERP_ORDER = 7;
    static constexpr int ERR_EST_ORDER = 7;

    using DOP_A = CArray2D<T, N_STAGES_EXT, N_STAGES_EXT>;
    using DOP_B = Array1D<T, N_STAGES>;
    using DOP_C = Array1D<T, N_STAGES_EXT>;
    using DOP_D = CArray2D<T, INTERP_ORDER - 3, N_STAGES_EXT>;
    using DOP_E = Array1D<T, N_STAGES+1>;

    static constexpr DOP_A make_A();

    static constexpr DOP_B make_B();

    static constexpr DOP_C make_C();

    static constexpr DOP_E make_E3();

    static constexpr DOP_E make_E5();

    static constexpr DOP_D make_D();

};

namespace detail{

/// @brief One DOP853 stage sweep. On entry K[0] holds the derivative at the start of the step;
/// the sweep fills K[1..12], K[12] being the final (FSAL) stage. Returns the scaled error norm.
template<typename T, typename SD, typename Atab, typename Ctab, typename Etab, typename RhsFn>
T dop853_step_impl(T& t_new, state_t<T, SD>& q_new, cache_t<T> t, cache_t<T> h, const state_t<T, SD>& q,
                   std::array<state_t<T, SD>, DOP_COEFS<T>::N_STAGES_EXT>& K, state_t<T, SD>& r,
                   const real_state_t<T, SD>& rtol, const real_state_t<T, SD>& atol,
                   const Atab& A, const Ctab& C, const Etab& E3, const Etab& E5, RhsFn&& rhs);

/// @brief DOP853's combined 3rd/5th order embedded error norm (Hairer, Norsett & Wanner), of the
/// step from q to q_new whose stages are K.
template<typename T, typename SD, typename Etab>
T dop853_error_norm(const std::array<state_t<T, SD>, DOP_COEFS<T>::N_STAGES_EXT>& K, const Etab& E3, const Etab& E5,
                    const state_t<T, SD>& q, const state_t<T, SD>& q_new, cache_t<T> h,
                    const real_state_t<T, SD>& rtol, const real_state_t<T, SD>& atol);

/// @brief The dense output at t of the step [t1, t2] that starts at y1, from its coefficients F.
template<typename T, typename SD>
void coef_mat_interp_dop853(state_t<T, SD>& result, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1,
                            const std::array<state_t<T, SD>, DOP_COEFS<T>::INTERP_ORDER>& F);

} // namespace ode::detail


template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived = void>
class DOP853 : public detail::BaseDispatcher<GetDerived<DOP853<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>{

    ODECRAFT_DECLARE_STEPPER;

    using Base = detail::BaseDispatcher<GetDerived<DOP853<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>;
    using Coefs = DOP_COEFS<T>;

public:
    using State = state_t<T, SD>;
    using RealState = real_state_t<T, SD>;

    static constexpr size_t N_STAGES       = Coefs::N_STAGES;       // 12
    static constexpr size_t N_STAGES_EXT   = Coefs::N_STAGES_EXT;   // 16: the FSAL stage, then 3 for dense output only
    static constexpr size_t Norder         = 8;
    static constexpr size_t INTERP_ORDER   = Coefs::INTERP_ORDER;   // 7
    static constexpr int    ERR_EST_ORDER  = Coefs::ERR_EST_ORDER;  // 7
    static constexpr bool   IS_IMPLICIT    = false;

    DOP853(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events = {});

    DEFAULT_RULE_OF_FOUR(DOP853)

    Stepper method() const;

    void Reset();

protected:

    void        ReAdjust(const State& new_vector);
    StepResult  adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q);
    auto        local_interp_impl() const;

private:

    T           step_impl(T& t_new, State& q_new, const T& t, const T& h, const State& q);

    void        cache_coef_matrix() const;

    // ndspan cannot constant-evaluate these tables (they are built with fill() and indexed writes),
    // so, regardless of T, they are plain (computed-once-at-construction) instance members. A is
    // the extended matrix: its row 12 holds the weights b, rows 13..15 the dense-output stages.
    Coefs::DOP_A A  = Coefs::make_A();
    Coefs::DOP_C C  = Coefs::make_C();
    Coefs::DOP_D D  = Coefs::make_D();
    Coefs::DOP_E E3 = Coefs::make_E3();
    Coefs::DOP_E E5 = Coefs::make_E5();

    T                                           h_last_ = 0; // step size actually used by the last sweep
    mutable std::array<State, N_STAGES_EXT>     K_;          // the stages of the last step
    mutable std::array<State, INTERP_ORDER>     coef_mat_;
    mutable bool                                coef_mat_is_cached_ = false;
    worker_t<State>                             r_scratch_;  // the stage input of a sweep

    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1);
    T INC_EXP = T(-1)/T(Norder);
    T MIN_ERR = T(1)/pow(T(5), Norder);
};

namespace detail{

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
struct SolverTypeGetter<Stepper::DOP853, SP, T, SD, OdeType, Derived>{
    using type = DOP853<T, SD, OdeType, SP, Derived>;
};

} // namespace ode::detail


} // namespace ode
