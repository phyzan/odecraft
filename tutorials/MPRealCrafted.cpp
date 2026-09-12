#include <odecraft/Compiled/odecraft.hpp>


using namespace ode::crafted;

template<typename T>
void solution_exact(T* out, const T& t, const T* q0){
    // q0 = [x(0), x'(0)]
    out[0] = q0[0]*mpfr::cos(mpfr::mpreal(t)) + q0[1]*mpfr::sin(mpfr::mpreal(t));
    out[1] = -q0[0]*mpfr::sin(mpfr::mpreal(t)) + q0[1]*mpfr::cos(mpfr::mpreal(t));
}

int main(){


    // Let's locate an event at extremely high precision
    // using the pre-built library

    // This must be done before anything else:
    // setting the working precision for mpreal_t, which is the arbitrary-precision scalar the library is compiled for.
    set_mpreal_prec(512); // 512 bits, ~154 decimal digits

    // Next, define the event
    pbox::Box<Event<mpreal_t>> event = make_precise_event<mpreal_t>(
        "crossing",
        [](const mpreal_t& /*t*/, const mpreal_t* q) -> mpreal_t {
            return q[0] - 1;
        },
        mpreal_t{0}, // machine precision tolerance
        1
    );

    // The initial state: q[0] = 2, q[1] = 0, 
    // i.e. x'' = -x with x(0) = 2, x'(0) = 0
    std::array<mpreal_t, 2> q0_array = {mpreal_t{2}, mpreal_t{0}};
    View1D<mpreal_t> q0{q0_array.data(), q0_array.size()};

    // Now the integrator
    BoxedRichSolver<mpreal_t> solver = make_rich_vsolver<mpreal_t>(
        Stepper::RK45,
        ode_t<mpreal_t>{
            .Rhs=[](mpreal_t* out, const mpreal_t& /*t*/, const mpreal_t* q){
                out[0] = q[1];
                out[1] = -q[0];
            }
        },
        mpreal_t{0}, // t0
        q0, // [2, 0]
        mpreal_t{1e-30}, // rtol
        mpreal_t{1e-30}, // atol
        mpreal_t{0}, // min_step
        mpreal_t{0}, // max_step
        mpreal_t{0}, // first step
        1, // direction
        make_event_list<mpreal_t>(std::move(event))
    );


    // Integrate in an interval we expect to contain the event, and print the result. The interval must be small for the example, because this is a very high-precision integration and the stepper will take very small steps, and relatively slow ones (compiled code using mpreal_t instead of built-in double precision has a big performance penalty).
    mpreal_t t_expected = 5*mpfr::const_pi()/3;
    mpreal_t epsilon = 1;
    mpreal_t t_start = t_expected - epsilon;


    // With q0 = [2, 0] at t = 0 the solution is x(t) = 2*cos(t), so the objective x - 1
    // vanishes when cos(t) = 1/2: at t = pi/3, where x is falling, and at t = 5*pi/3, where it
    // is rising. The event was built with dir = +1, so the solver reports the rising one.

    
    std::array<mpreal_t, 2> q_start_array;
    solution_exact(q_start_array.data(), t_start, q0_array.data());

    // We could have set t0 to t_start when constructing the solver, but let's do it now to show how to set new initial conditions in an already initialized solver.
    solver->do_set_ics(t_start, q_start_array.data());

    // We advance until the event is detected, and time the integration duration.
    auto start = std::chrono::high_resolution_clock::now();
    solver->do_advance_to_event(1000, {"crossing"});
    auto end = std::chrono::high_resolution_clock::now();
    // Print the result
    if (solver->get_at_event()){
        std::cout << "Event 'crossing' detected at t = " << solver->get_time() << std::endl;
        std::cout << "Expected t = " << t_expected << std::endl;
        std::cout << "Difference = " << solver->get_time() - t_expected << std::endl;
        std::cout << "Integration took " << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;
    } else {
        std::cout << "No event detected." << std::endl;
    }
}