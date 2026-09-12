#include <odecraft/Compiled/odecraft.hpp>

#include <cmath>
#include <iostream>

/*
Exercises the pre-compiled ode::crafted interface.

The point of this file is as much what it does NOT do as what it does: it includes only
Compiled/ headers, so if it links, no solver was instantiated here -- every symbol it needs
came out of src/. Any instantiation that went missing from src/ shows up as a link error,
and any that is duplicated shows up when this object is linked next to the library.
*/

using namespace ode::crafted;


// Harmonic oscillator: x'' = -x, so (x, v)' = (v, -x). Exact solution from x(0)=1, v(0)=0
// is x(t) = cos(t), which gives every check below a closed form to compare against.
static void rhs(double* out, const double& /*t*/, const double* q){
    out[0] = q[1];
    out[1] = -q[0];
}

// d(q')/dq, column-major: out[j + i*n] = df_i/dq_j.
static void jac(double* out, const double& /*t*/, const double* /*q*/){
    out[0] =  0; out[1] = -1;   // column 0: df0/dq0, df1/dq0
    out[2] =  1; out[3] =  0;   // column 1: df0/dq1, df1/dq1
}

static int failures = 0;

static void check(bool ok, const std::string& what){
    if (ok){
        std::cout << "  ok   : " << what << std::endl;
    } else {
        std::cerr << "  FAIL : " << what << std::endl;
        ++failures;
    }
}

static void check_close(double got, double expected, double tol, const std::string& what){
    const double err = std::abs(got - expected);
    if (err <= tol){
        std::cout << "  ok   : " << what << " (" << got << ")" << std::endl;
    } else {
        std::cerr << "  FAIL : " << what << " -- got " << got << ", expected " << expected
                  << " (err " << err << " > " << tol << ")" << std::endl;
        ++failures;
    }
}


// Every stepper, through the runtime factory, against cos(t).
static void run_steppers(){
    std::cout << "\n-- steppers via make_rich_vsolver --" << std::endl;

    const double t0 = 0, tmax = 2.0;
    std::array<double, 2> q0 = {1.0, 0.0};

    const std::pair<Stepper, const char*> methods[] = {
        {Stepper::Euler,  "Euler"},
        {Stepper::RK4,    "RK4"},
        {Stepper::RK23,   "RK23"},
        {Stepper::RK45,   "RK45"},
        {Stepper::DOP853, "DOP853"},
        {Stepper::BDF,    "BDF"},
    };

    for (const auto& [method, name] : methods){
        // Euler and RK4 are fixed-step, so they need a small step rather than a tolerance.
        const bool fixed_step = (method == Stepper::Euler || method == Stepper::RK4);
        const double stepsize = fixed_step ? 1e-5 : 0.0;
        const double tol      = fixed_step ? 1e-3 : 1e-6;

        BoxedRichSolver<double> solver = make_rich_vsolver<double>(
            method, ode_t<double>{.Rhs = rhs, .Jac = jac}, t0,
            View1D<double>{q0.data(), 2}, 1e-10, 1e-10, 0, 0, stepsize, 1);

        solver->do_advance_until(tmax);
        check_close(solver->get_vector()[0], std::cos(tmax), tol, std::string(name) + " x(2)");
    }
}


// The high-level driver, its dense output, and a null Jacobian falling back to finite differences.
static void run_ode_driver(){
    std::cout << "\n-- ODE driver --" << std::endl;

    std::array<double, 2> q0 = {1.0, 0.0};
    ODE<double> ode(ode_t<double>{.Rhs = rhs}, 0.0, View1D<double>{q0.data(), 2},
                    1e-12, 1e-12, 0, 0, 0, 1, {}, Stepper::RK45);

    OdeResult<double> result;
    check(ode.integrate_until(&result, 10.0), "integrate_until reported success");
    check(result.success(), "result.success()");
    check(!result.diverges(), "solution did not diverge");
    check_close(result.q(result.t().size() - 1, 0), std::cos(10.0), 1e-6, "x(10) with a null Jac");

    // Dense output: an OdeSolution answers at times that were never sampled.
    //
    // Built on a fresh ODE rather than reusing the one above via reset(). BaseSolver::Reset()
    // restores every state field except is_at_new_state_, so a reset solver reports that it is
    // not at a new state and rich_integrate_until() then builds a dense-output interval over
    // [t_old, t] == [t0, t0] and throws "Zero interval not allowed". That is a pre-existing
    // header-only bug, unrelated to the compiled interface -- it reproduces just as readily
    // through namespace ode.
    std::array<double, 2> q0b = {1.0, 0.0};
    ODE<double> ode_dense(ode_t<double>{.Rhs = rhs}, 0.0, View1D<double>{q0b.data(), 2},
                          1e-12, 1e-12, 0, 0, 0, 1, {}, Stepper::RK45);
    OdeSolution<double> solution;
    check(ode_dense.rich_integrate_until(solution, 10.0), "rich_integrate_until reported success");
    check_close(solution(3.7)[0], std::cos(3.7), 1e-6, "dense output at t=3.7");
    check_close(solution(8.25)[0], std::cos(8.25), 1e-6, "dense output at t=8.25");
}


// Events, including a mask, through the compiled makers.
static void run_events(){
    std::cout << "\n-- events --" << std::endl;

    std::array<double, 2> q0 = {1.0, 0.0};

    // x(t) = cos(t) crosses zero at pi/2 + k*pi.
    EventList<double> events = make_event_list<double>(
        make_precise_event<double>("zero_x", [](const double& /*t*/, const double* q){ return q[0]; }, 1e-14, 0),
        make_periodic_event<double>("every_2", 2.0)
    );

    ODE<double> ode(ode_t<double>{.Rhs = rhs, .Jac = jac}, 0.0, View1D<double>{q0.data(), 2},
                    1e-12, 1e-12, 0, 0, 0, 1, std::move(events), Stepper::RK45);

    OdeResult<double> result;
    ode.integrate_until(&result, 10.0);

    const OrbitData<double>& zeros = result.event_data().data("zero_x");
    const OrbitData<double>& ticks = result.event_data().data("every_2");

    // Zero crossings of cos in (0, 10]: pi/2, 3pi/2, 5pi/2 -- three of them, since 7pi/2 ~ 11.0.
    check(zeros.size() == 3, "found 3 zero crossings of x, got " + std::to_string(zeros.size()));
    if (zeros.size() > 0){
        check_close(zeros.t[0], M_PI / 2, 1e-8, "first zero crossing at pi/2");
    }
    // Periodic ticks at 2, 4, 6, 8, 10.
    check(ticks.size() == 5, "fired 5 periodic events, got " + std::to_string(ticks.size()));
}


// Variational integration: the largest Lyapunov exponent of a non-chaotic system tends to 0.
static void run_variational(){
    std::cout << "\n-- variational solver --" << std::endl;

    std::array<double, 2> q0 = {1.0, 0.0};
    std::array<double, 2> dq0 = {1e-8, 0.0};

    VariationalODE<double> vode(ode_t<double>{.Rhs = rhs, .Jac = jac}, 0.0,
                               View1D<double>{q0.data(), 2}, View1D<double>{dq0.data(), 2},
                               1.0, 1e-12, 1e-12, 0, 0, 0, 1, {}, Stepper::RK45);

    OdeResult<double> result;
    vode.integrate_until(&result, 200.0);

    const double lyap = vode.lyap_values().empty() ? 1.0 : vode.lyap_values().back();
    check(std::abs(lyap) < 1e-2, "Lyapunov exponent of the harmonic oscillator is ~0 (got "
                                 + std::to_string(lyap) + ")");
}


// The same code paths at the other two compiled precisions.
static void run_other_scalars(){
    std::cout << "\n-- float and long double --" << std::endl;

    std::array<float, 2> qf = {1.0f, 0.0f};
    ODE<float> ode_f(ode_t<float>{.Rhs = [](float* out, const float& , const float* q){
                         out[0] = q[1]; out[1] = -q[0];
                     }}, 0.0f, View1D<float>{qf.data(), 2}, 1e-6f, 1e-6f, 0, 0, 0, 1, {}, Stepper::RK45);
    OdeResult<float> rf;
    ode_f.integrate_until(&rf, 2.0f);
    check_close(rf.q(rf.t().size() - 1, 0), std::cos(2.0), 1e-3, "float x(2)");

    std::array<long double, 2> ql = {1.0L, 0.0L};
    ODE<long double> ode_l(ode_t<long double>{.Rhs = [](long double* out, const long double&, const long double* q){
                               out[0] = q[1]; out[1] = -q[0];
                           }}, 0.0L, View1D<long double>{ql.data(), 2}, 1e-15L, 1e-15L, 0, 0, 0, 1, {}, Stepper::RK45);
    OdeResult<long double> rl;
    ode_l.integrate_until(&rl, 2.0L);
    check_close(static_cast<double>(rl.q(rl.t().size() - 1, 0)), std::cos(2.0), 1e-9, "long double x(2)");
}


void test_compiled_interface(){
    std::cout << "\n========== Testing the compiled ode::crafted interface ==========" << std::endl;
    failures = 0;

    run_steppers();
    run_ode_driver();
    run_events();
    run_variational();
    run_other_scalars();

    if (failures == 0){
        std::cout << "\nAll compiled-interface checks passed." << std::endl;
    } else {
        std::cerr << "\n" << failures << " compiled-interface check(s) FAILED." << std::endl;
    }
}
