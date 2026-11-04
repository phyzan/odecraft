#include <odecraft/odecraft.hpp>
#include <lazex/apps/lazex_mpreal.hpp>

using namespace ode;

void test_lazy_scalar(){
    using A = mpfr::mpreal;
    using T = lazex::LazyType<A>; //set T = A for performance comparison.
    using State = common::State<T, 2>;

    // mpfr::mpreal::set_default_prec(256);
    lazex::set_default_mpreal_prec(256);
    State y0 = {1, -3};
    State tol = {T{1e-40}, T{1e-40}};

    auto solver = common::getSolver<Stepper::BDF, SolverPolicy::Static>(
        OdeData{
            .Rhs=ODECRAFT_LAMBDA(out, t, y){ 
                out[0] = y[1];
                out[1] = -y[0];
            },
        },
        T{0.},
        y0,
        tol,
        tol
    );

    auto t_start = std::chrono::high_resolution_clock::now();
    solver.advance_until(0.001);
    auto t_end = std::chrono::high_resolution_clock::now();
    solver.fill_current_vector(y0);
    std::cout << "q: " << y0[0] << " " << y0[1] << std::endl;
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms" << std::endl;
}