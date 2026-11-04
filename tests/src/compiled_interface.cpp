#include <odecraft/Crafted.hpp>

#include <cmath>
#include <iostream>

/*
Exercises the pre-compiled ode::crafted interface.

The point of this file is as much what it does NOT do as what it does: it includes only
Crafted headers, so if it links, no solver was instantiated here -- every symbol it needs
came out of src/. Any instantiation that went missing from src/ shows up as a link error,
and any that is duplicated shows up when this object is linked next to the library.
*/

using namespace ode::crafted;


// Harmonic oscillator: x'' = -x, so (x, v)' = (v, -x). Exact solution from x(0)=1, v(0)=0
// is x(t) = cos(t), which gives every check below a closed form to compare against.
static void rhs(State<double>& out, double /*t*/, const State<double>& q){
    out[0] = q[1];
    out[1] = -q[0];
}

// d(q')/dq: out(i, j) = df_i/dq_j.
static void jac(JacMat<double>& out, double /*t*/, const State<double>& /*q*/){
    out(0, 0) =  0; out(0, 1) = 1;
    out(1, 0) = -1; out(1, 1) = 0;
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
    State<double> q0 = {1.0, 0.0};
    State<double> tol_ = {1e-10, 1e-10};

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
            method, ode_t<double>{.Rhs = rhs, .Jac = jac}, t0, q0, tol_, tol_, 0, 0, stepsize, 1);

        solver->do_advance_until(tmax);
        auto qf = q0;
        solver->get_current_vector(qf);
        check_close(qf[0], std::cos(tmax), tol, std::string(name) + " x(2)");
    }
}


// The high-level driver, its dense output, and a null Jacobian falling back to finite differences.
static void run_ode_driver(){
    std::cout << "\n-- ODE driver --" << std::endl;

    State<double> q0 = {1.0, 0.0};
    State<double> tol = {1e-12, 1e-12};
    ODE<double> ode(ode_t<double>{.Rhs = rhs}, 0.0, q0, tol, tol, 0, 0, 0, 1, {}, Stepper::RK45);

    OdeResult<double> result;
    check(ode.integrate_until(&result, 10.0), "integrate_until reported success");
    check(result.success(), "result.success()");
    check_close(result.q().back()[0], std::cos(10.0), 1e-6, "x(10) with a null Jac");

    // Dense output: an OdeSolution answers at times that were never sampled.
    //
    // Built on a fresh ODE rather than reusing the one above, which has already integrated to
    // t=10: asking that one for dense output up to the same t=10 is a zero-length span, and
    // rich_integrate_until() then reports false instead of filling `solution`. Calling reset()
    // first would work too; a fresh object just keeps the two checks independent.
    ODE<double> ode_dense(ode_t<double>{.Rhs = rhs}, 0.0, q0, tol, tol, 0, 0, 0, 1, {}, Stepper::RK45);
    OdeSolution<double> solution;
    check(ode_dense.rich_integrate_until(solution, 10.0), "rich_integrate_until reported success");
    State<double> q = q0;
    solution(q, 3.7);
    check_close(q[0], std::cos(3.7), 1e-6, "dense output at t=3.7");
    solution(q, 8.25);
    check_close(q[0], std::cos(8.25), 1e-6, "dense output at t=8.25");
}


// Events, including a mask, through the compiled makers.
static void run_events(){
    std::cout << "\n-- events --" << std::endl;

    State<double> q0 = {1.0, 0.0};
    State<double> tol = {1e-12, 1e-12};

    // x(t) = cos(t) crosses zero at pi/2 + k*pi.
    EventList<double> events = make_event_list<double>(
        make_precise_event<double>("zero_x", [](const double& /*t*/, const State<double>& q){ return q[0]; }, 1e-14, 0),
        make_periodic_event<double>("every_2", 2.0)
    );

    ODE<double> ode(ode_t<double>{.Rhs = rhs, .Jac = jac}, 0.0, q0, tol, tol, 0, 0, 0, 1, std::move(events), Stepper::RK45);

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

    State<double> tol = {1e-12, 1e-12};
    VarState<double> q0{.main = {1.0, 0.0}, .variation = {1e-8, 0.0}};
    VarState<double> var_tol{.main = tol, .variation = tol};

    VariationalODE<double> vode(ode_t<double>{.Rhs = rhs, .Jac = jac}, 0.0, q0, 1.0, var_tol, var_tol, 0, 0, 0, 1, {}, Stepper::RK45);

    OdeResult<double, VarSD> result;
    vode.integrate_until(&result, 200.0);

    const double lyap = vode.lyap_values().empty() ? 1.0 : vode.lyap_values().back();
    check(std::abs(lyap) < 1e-2, "Lyapunov exponent of the harmonic oscillator is ~0 (got "
                                 + std::to_string(lyap) + ")");

    // Events on a variational solver act on the VarState, and fire as in run_events.
    EventList<double, VarState<double>> events = make_variational_event_list<double>(
        make_variational_precise_event<double>("zero_x", [](const double& /*t*/, const VarState<double>& q){ return q.main[0]; }, 1e-14, 0),
        make_variational_periodic_event<double>("every_2", 2.0)
    );

    VariationalODE<double> vode_events(ode_t<double>{.Rhs = rhs, .Jac = jac}, 0.0, q0, 1.0, var_tol, var_tol, 0, 0, 0, 1, std::move(events), Stepper::RK45);

    OdeResult<double, VarSD> result_events;
    vode_events.integrate_until(&result_events, 10.0);

    const OrbitData<double, VarSD>& zeros = result_events.event_data().data("zero_x");
    check(zeros.size() == 3, "variational: found 3 zero crossings of x, got " + std::to_string(zeros.size()));
    if (zeros.size() > 0){
        check_close(zeros.t[0], M_PI / 2, 1e-8, "variational: first zero crossing at pi/2");
    }
    const size_t n_ticks = result_events.event_data().data("every_2").size();
    check(n_ticks == 5, "variational: fired 5 periodic events, got " + std::to_string(n_ticks));
}


// A rotation field, linear in the coordinates, so that interpolation is exact on either kind of
// mesh and a streamline from (1, 0) runs along the unit circle.
static void run_vector_fields(){
    std::cout << "\n-- interpolators and vector fields --" << std::endl;

    constexpr size_t n = 21;
    grid_axes_t axes;
    for (size_t axis = 0; axis < 2; axis++){
        Array1D<double, 0> x(n);
        for (size_t i = 0; i < n; i++){
            x[i] = -2 + 4.0*i/(n - 1);
        }
        axes.push_back(x);
    }

    std::vector<double> points(2*n*n), values(2*n*n);
    for (size_t i = 0; i < n; i++){
        for (size_t j = 0; j < n; j++){
            const size_t p = 2*(i*n + j);
            points[p] = axes[0][i];
            points[p + 1] = axes[1][j];
            values[p] = -axes[1][j];
            values[p + 1] = axes[0][i];
        }
    }
    const field_values_t field(values.data(), n*n, size_t(2));

    const double x0[2] = {1.0, 0.0};
    const double point[2] = {0.33, -0.71};
    double out[2];

    RegularGridInterpolator regular(field, axes, true);
    check(regular.interp(out, point), "regular interpolation inside the grid");
    check_close(out[0], 0.71, 1e-12, "regular interpolation, first component");

    RegularVectorField regular_field(field, axes, CoordType::Cartesian, true);
    auto regular_line = regular_field.streamline(x0, M_PI, 1e-9, 1e-9, 0, 0, 0, 1, Stepper::RK45, false);
    check_close(regular_line->q().back()[0], -1.0, 1e-6, "regular streamline after half a turn");

    ScatteredNdInterpolator scattered(points.data(), field, 2, true);
    check(scattered.interp(out, point), "scattered interpolation inside the hull");
    check_close(out[0], 0.71, 1e-12, "scattered interpolation, first component");

    ScatteredVectorField scattered_field(points.data(), field, 2, true);
    auto scattered_line = scattered_field.streamline(x0, M_PI, 1e-9, 1e-9, 0, 0, 0, 1, Stepper::RK45, false);
    check_close(scattered_line->q().back()[0], -1.0, 1e-6, "scattered streamline after half a turn");
}


// The same code paths at the other two compiled precisions.
static void run_other_scalars(){
    std::cout << "\n-- float and long double --" << std::endl;

    State<float> qf = {1.0f, 0.0f};
    State<float> tol_f = {1e-6f, 1e-6f};
    ODE<float> ode_f(ode_t<float>{.Rhs = [](State<float>& out, float, const State<float>& q){
                         out[0] = q[1]; out[1] = -q[0];
                     }}, 0.0f, qf, tol_f, tol_f, 0, 0, 0, 1, {}, Stepper::RK45);
    OdeResult<float> rf;
    ode_f.integrate_until(&rf, 2.0f);
    check_close(rf.q().back()[0], std::cos(2.0), 1e-3, "float x(2)");

    State<long double> ql = {1.0L, 0.0L};
    State<long double> tol_l = {1e-15L, 1e-15L};
    ODE<long double> ode_l(ode_t<long double>{.Rhs = [](State<long double>& out, long double, const State<long double>& q){
                               out[0] = q[1]; out[1] = -q[0];
                           }}, 0.0L, ql, tol_l, tol_l, 0, 0, 0, 1, {}, Stepper::RK45);
    OdeResult<long double> rl;
    ode_l.integrate_until(&rl, 2.0L);
    check_close(static_cast<double>(rl.q().back()[0]), std::cos(2.0), 1e-9, "long double x(2)");
}


void test_compiled_interface(){
    std::cout << "\n========== Testing the compiled ode::crafted interface ==========" << std::endl;
    failures = 0;

    run_steppers();
    run_ode_driver();
    run_events();
    run_variational();
    run_vector_fields();
    run_other_scalars();

    if (failures == 0){
        std::cout << "\nAll compiled-interface checks passed." << std::endl;
    } else {
        std::cerr << "\n" << failures << " compiled-interface check(s) FAILED." << std::endl;
    }
}
