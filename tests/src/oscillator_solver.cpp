#include <odecraft/Events/Events.hpp>
#include <odecraft/Core/VirtualBase.hpp>
#include <odecraft/odecraft.hpp>
#include <mpreal.h>

using namespace ode;

template<typename T>
T crossing(const T& /*t*/, const common::State<T, 2>& y) {
    return y[1] - 1;
}


template<typename Scalar>
void crossing_test(Stepper method) {
    using SD = contracts::common::SD<2>;
    using State = common::State<Scalar, 2>;

    // Initial conditions
    State y0 = {3, 0};


    // Define the y' = 1 crossing
    BoxedEvent<Scalar, State> event = make_precise_event<Scalar, State>("event", crossing<Scalar>, Scalar(0));

    // Create solver
    Scalar t = 0;
    State rtol = {1e-6, 1e-6};
    State atol = {1e-9, 1e-9};
    Scalar min_step = 0;
    Scalar max_step = 1;
    Scalar stepsize = 0;
    int dir = 1;


    pbox::Box<OdeRichSolver<Scalar, SD>> solver = common::make_solver<UtilPolicy::RichVirtual>(
        method,
        OdeData{.Rhs=
            ODECRAFT_LAMBDA(out, /*t*/, q) {
                const Scalar omega{1}; // angular frequency
                out[0] = q[1];
                out[1] = - omega*omega*q[0];
            },
        },
        t,
        y0, // q0
        rtol,
        atol,
        min_step,
        max_step,
        stepsize,
        dir,
        make_event_list<Scalar, State>(std::move(event))
    );

    // Advance until event is detected
    while (!solver->get_at_event()) {
        solver->do_advance();
    }

    std::cout << "Event detected at t = " << solver->get_time() << "\n";
    std::cout << "State at event: ";
    auto yf = y0;
    solver->get_current_vector(yf);
    for (size_t i = 0; i < 2; ++i) {
        std::cout << yf[i] << " ";
    }
    std::cout << "\n" << std::endl;
}



void test_oscillator_solver() {

    // Test with double precision
    std::cout << "Testing with double precision:\n";
    crossing_test<double>(Stepper::RK45);

    // Test with arbitrary precision
    std::cout << "\nTesting with arbitrary precision (mpreal):\n";
    mpfr::mpreal::set_default_prec(256); // Set precision to 256 bits

    crossing_test<mpfr::mpreal>(Stepper::RK45);
    std::cout << "Expected event at t = 3.48143\n";
}