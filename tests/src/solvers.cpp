#include "odecraft/Contracts/DenseDescriptor.hpp"
#include "odecraft/Core/SolverFactory.hpp"
#include <odecraft/odecraft.hpp>

using namespace ode;

/*
Testing a simple non-virtual solver, using all available methods
*/

template<size_t N>
using SD = contracts::VectorDescriptor<N>;

template<typename T, size_t N>
using State = state_t<T, SD<N>>;

static constexpr auto rhs = [](auto& out, double /*t*/, const auto& q) -> void {
    const double mu = 300.;
    const auto& x = q[0];
    const auto& xdot = q[1];
    out[0] = xdot;
    out[1] = mu * (1 - x*x) * xdot - x;
};

template<Stepper S>
void run(const char* label){
    // declare ics
    const double t0 = 0;
    const double min_step = 0;
    const double max_step = 0;
    const double stepsize = 0.0001;

    const double tmax = 1.0;
    std::cout << "\n---------- Testing " << label << " method------------------\n" << std::endl;

    // Stack allocated solver (the size is a compile-time parameter)
    State<double, 2> q0_stack = {1, 0};
    State<double, 2> tol_stack = {1e-10, 1e-10};
    auto solver_1 = getSolver<S, SolverPolicy::Static, double, SD<2>>(
        OdeData{.Rhs=rhs}, t0, q0_stack, tol_stack, tol_stack, min_step, max_step, stepsize, 1);

    // Heap allocated solver (the size is a runtime parameter)
    State<double, 0> q0_heap = {1, 0};
    State<double, 0> tol_heap = {1e-10, 1e-10};
    auto solver_2 = getSolver<S, SolverPolicy::Static, double, SD<0>>(
        OdeData{.Rhs=rhs}, t0, q0_heap, tol_heap, tol_heap, min_step, max_step, stepsize, 1);

    
    
    solver_1.advance_until(tmax);
    solver_2.advance_until(tmax);


    State<double, 2> q1;
    State<double, 0> q2 = q0_heap; // the current vector is copied into q2, so it must have the size of the state
    solver_1.fill_current_vector(q1);
    solver_2.fill_current_vector(q2);
    if (q1[0] != q2[0] || q1[1] != q2[1]){
        std::cerr << "Error: Stack and Heap solvers produced different results!" << std::endl;
    } else {
        std::cout << "Result ( t = " << tmax << " ): x = " << q1[0] << ", xdot = " << q1[1] << std::endl;

        State<double, 2> arr;
        solver_1.Rhs(arr, tmax, q1);
        
        std::cout << "xdotdot = " << arr[0] << " " << arr[1] << std::endl;
    }
}

void test_solvers(){

    run<Stepper::Euler>("Euler");
    run<Stepper::RK23>("RK23");
    run<Stepper::RK45>("RK45");
    run<Stepper::DOP853>("DOP853");
    run<Stepper::BDF>("BDF");
    run<Stepper::RK4>("RK4");

}