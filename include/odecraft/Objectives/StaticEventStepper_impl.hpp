#pragma once

#include <odecraft/Objectives/StaticEventStepper.hpp>

namespace ode{

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
template<typename... Args>
StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::StaticEventStepper(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, T t0, State q0, Args&&... args) : Base(std::move(ode), std::move(t0), q0, std::forward<Args>(args)...), worker(q0), obj(std::move(funcs)){
    this->cache_current_signs();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
void StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::Reset(){
    Base::Reset();
    cache_current_signs();
    detected.fill(false);
    current_idx = -1;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
bool StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::is_at_objective() const {
    return current_idx != -1;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
int StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::current_objective() const {
    return current_idx;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
template<typename... Args>
bool StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::Adv_Impl(Args&&... args){
    T nearest_floor;
    size_t idx;
    current_idx = -1;
    if (this->is_at_new_state()){
        bool success = Base::Adv_Impl(std::forward<Args>(args)...);
        if (success && this->get_nearest_floor(nearest_floor, idx)){
            if (this->t() == nearest_floor){
                detected[idx] = false;
                // RequestTimeFloor already cached the sign at t_new. Only a root at t_new itself
                // must not count as the start of a crossing in the next step.
                if (nearest_floor == this->t_new()){
                    cached_sign[idx] = 0;
                }
                current_idx = int(idx);
            }
            return true;
        } else {
            return success;
        }
    } else if (this->get_nearest_floor(nearest_floor, idx)){
        if (Base::Adv_Impl(nearest_floor, std::forward<Args>(args)...)){
            if (this->t() == nearest_floor){
                detected[idx] = false; //turn off for next step, since the goal was achieved
                if (nearest_floor == this->t_new()){
                    cached_sign[idx] = 0; // see above
                }
                current_idx = int(idx);
            }
            return true;
        } else {
            return false;
        }
    } else {
        return Base::Adv_Impl(std::forward<Args>(args)...);
    }
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
void StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::ReAdjust(const State& new_vector){
    Base::ReAdjust(new_vector);
    cache_current_signs();
    detected.fill(false);
    current_idx = -1;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
bool StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::RequestTimeFloor(T& out){
    bool base_floor = Base::RequestTimeFloor(out);
    const int d = this->direction();
    T my_floor = this->t_new();
    NDSPAN_FOR_LOOP(I, NOBJ,
        const int old_sgn = cached_sign[I];
        const T f_new = std::get<I>(obj).func(this->t_new(), this->vector_new());
        const int new_sign = sgn(f_new);
        cached_sign[I] = new_sign;
        detected[I] = false;
        if (old_sgn != 0){
            // means we were not in the objective function in the previous step
            const int dir = std::get<I>(obj).dir;
            bool crossed;
            if (dir==0){
                crossed = old_sgn*new_sign <= 0;
            } else if (dir*d == 1){
                crossed = old_sgn < 0 && new_sign >= 0;
            } else {
                crossed = old_sgn > 0 && new_sign <= 0;
            }

            if (crossed){
                const T& t_old = this->t_old();
                const State& q_old = this->vector_old();
                T f_old = std::get<I>(obj).func(t_old, q_old);
                decltype(auto) w_state = worker.get();
                RootStatus rs = bisect_root(
                    values[I],
                    static_cast<T*>(nullptr),
                    [&, f_open = this->local_interp_impl()](const T& t){
                        f_open(w_state, t);
                        return std::get<I>(obj).func(t, w_state);
                    },
                    t_old,
                    this->t_new(),
                    std::move(f_old),
                    std::move(f_new),
                    std::get<I>(obj).ftol
                );
                if ((detected[I] = (rs == RootStatus::Success))){
                    my_floor = this->nearest_time(my_floor, values[I]);
                }
            }
        }
    );

    if (base_floor){
        out = this->nearest_time(my_floor, out);
    } else {
        out = my_floor;
    }
    return true;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
bool StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::get_nearest_floor(T& out, size_t& idx) const{
    bool found = false;
    NDSPAN_FOR_LOOP(I, NOBJ,
        if (detected[I]){
            if (!found){
                out = values[I];
                idx = I;
                found = true;
            } else if (values[I]*this->direction() < out*this->direction()){
                out = values[I];
                idx = I;
            }
        }
    );
    return found;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename... ObjFun>
void StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>::cache_current_signs(){
    NDSPAN_FOR_LOOP(I, NOBJ,
        cached_sign[I] = sgn(std::get<I>(obj).func(this->t(), this->vector_new()));
    );
}


template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename ObjFun>
template<typename... Args>
SingleStaticEventStepper<S, T, SD, OdeType, SP, ObjFun>::SingleStaticEventStepper(ObjFunData<T, ObjFun> data, OdeType ode, Args&&... args) : Base(std::tuple{data}, std::move(ode), std::forward<Args>(args)...) {}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename ObjFun>
template<typename... Args>
SingleStaticEventStepper<S, T, SD, OdeType, SP, ObjFun>::SingleStaticEventStepper(ObjFun obj_fun, OdeType ode, T t0, State q0, Args&&... args) : Base(std::tuple{ObjFunData{std::move(obj_fun), T{0.0}, 1}}, std::move(ode), std::move(t0), std::move(q0), std::forward<Args>(args)...) {}

template<Stepper S, SolverPolicy SP, typename T, typename SD, typename OdeType, typename... ObjFun, typename... Args>
auto getEventStepper(std::tuple<ObjFunData<T, ObjFun>...> funcs, OdeType ode, Args&&... args){
    return StaticEventStepper<S, T, SD, OdeType, SP, ObjFun...>(std::move(funcs), std::move(ode), std::forward<Args>(args)...);
}

} // namespace ode

