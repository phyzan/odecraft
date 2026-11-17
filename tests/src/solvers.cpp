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

template<typename Stepper>
void run(){
    const std::string_view label = Stepper::name;
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
    auto solver_1 = getSolver<SD<2>, SolverPolicy::Static>(
        Stepper{},
        OdeData{.Rhs=rhs}, t0, q0_stack, tol_stack, tol_stack, min_step, max_step, stepsize, 1);

    // Heap allocated solver (the size is a runtime parameter)
    State<double, 0> q0_heap = {1, 0};
    State<double, 0> tol_heap = {1e-10, 1e-10};
    auto solver_2 = getSolver<SD<0>, SolverPolicy::Static>(
        Stepper{},
        OdeData{.Rhs=rhs}, t0, q0_heap, tol_heap, tol_heap, min_step, max_step, stepsize, 1);

    
    
    // The solver knows its stepper through the tag
    if (solver_1.method() != label){
        std::cerr << "Error: method() is " << solver_1.method() << " instead of " << label << std::endl;
    }

    solver_1.advance_until(tmax);
    solver_2.advance_until(tmax);


    State<double, 2> q1;
    State<double, 0> q2 = q0_heap; // the current vector is copied into q2, so it must have the size of the state
    solver_1.FillCurrentState(q1);
    solver_2.FillCurrentState(q2);
    if (q1[0] != q2[0] || q1[1] != q2[1]){
        std::cerr << "Error: Stack and Heap solvers produced different results!" << std::endl;
    } else {
        std::cout << "Result ( t = " << tmax << " ): x = " << q1[0] << ", xdot = " << q1[1] << std::endl;

        State<double, 2> arr;
        solver_1.rhs(arr, tmax, q1);
        
        std::cout << "xdotdot = " << arr[0] << " " << arr[1] << std::endl;
    }
}

void test_solvers(){

    run<tags::Euler>();
    run<tags::RK23>();
    run<tags::RK45>();
    run<tags::DOP853>();
    run<tags::BDF>();
    run<tags::RK4>();

}