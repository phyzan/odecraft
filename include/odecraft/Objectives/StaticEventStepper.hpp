#pragma once

#include <odecraft/Core/RichSolver/RichBase.hpp>

namespace ode{

template<typename T, typename Callable>
struct ObjFunData{

    Callable func;
    T ftol = 0; // tolerance for root finding (0 means machine precision)
    int dir = 0; // 0 means any direction, 1 means increasing, -1 means decreasing

};

/*
StaticEventStepper passes itself as the Derived type to the base solver class,
and as a result it is the most derived class in the CRTP hierarchy.
That means that any class deriving from StaticEventStepper that overrides functions like `Adv_Impl` will not have those overrides called by the base solver class,.
For virtual inheritance this might not be the case, so it is best not to override any of the base solver functions, but only extend the functionality.
*/
template<typename Stepper, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
class StaticEventStepper : public Stepper::template type<T, SD, OdeType, SP, StaticEventStepper<Stepper, T, SD, OdeType, SP, ObjFun...>>{

    ODECRAFT_DERIVED_STEPPER;

    using Base = Stepper::template type<T, SD, OdeType, SP, StaticEventStepper<Stepper, T, SD, OdeType, SP, ObjFun...>>;
    using State = state_t<T, SD>;

    static_assert((isObjFun<ObjFun, T, State> && ...), "At least one invalid objective function");

public:

    static constexpr size_t NOBJ = sizeof...(ObjFun);

    template<typename... Args>
    StaticEventStepper(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, T t0, State q0, Args&&... args);

    template<size_t I>
    const auto& objective() const {
        return std::get<I>(obj);
    }

    void Reset();

    bool is_at_objective() const;

    int current_objective() const;

protected:

    template<typename... Args>
    bool Adv_Impl(Args&&... args);

    void ReAdjust(const State& new_vector);

    bool RequestTimeFloor(T& out);

private:

    bool get_nearest_floor(T& out, size_t& idx) const;

    void cache_current_signs();

    std::array<T, NOBJ> values = {};
    std::array<int, NOBJ> cached_sign = {};
    std::array<bool, NOBJ> detected = {};
    worker_t<State> worker;
    std::tuple<ObjFunData<T, ObjFun>...> obj;
    int current_idx = -1;
};

template<typename Stepper, typename T, typename SD, typename OdeType, SolverPolicy SP, typename ObjFun>
class SingleStaticEventStepper : public StaticEventStepper<Stepper, T, SD, OdeType, SP, ObjFun>{

    using Base = StaticEventStepper<Stepper, T, SD, OdeType, SP, ObjFun>;
    using State = state_t<T, SD>;
public:

    template<typename... Args>
    SingleStaticEventStepper(ObjFunData<T, ObjFun> data, OdeType ode, Args&&... args);

    template<typename... Args>
    SingleStaticEventStepper(ObjFun obj_fun, OdeType ode, T t0, State q0, Args&&... args);

};

template<typename Stepper, SolverPolicy SP, typename T, typename SD, typename OdeType, typename... ObjFun, typename... Args>
auto getEventStepper(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, Args&&... args);




}; // namespace ode

