#include "odecraft/Contracts/DenseDescriptor.hpp"
#include <odecraft/odecraft.hpp>


struct MyODE{


    static void Rhs(auto& dy_dt, auto /*t*/, const auto& q) {
        //3D lorenz system, args = {sigma, rho, beta}
        const double sigma = 10.0;
        const double rho = 28.0;
        const double beta = 8.0/3.0;
        dy_dt[0] = sigma*(q[1] - q[0]);
        dy_dt[1] = q[0]*(rho - q[2]) - q[1];
        dy_dt[2] = q[0]*q[1] - beta*q[2];
    }

};


using namespace ode;


void test_variational_solver(){

    std::cout << "\n---------- Testing VariationalSolver ------------------\n" << std::endl;

    std::cout << " (This might take too long if compiled in debug mode)" << std::endl;

    using T = double;

    using SD = contracts::CArrayDescriptor<3>;
    using State = SD::state_type<T>;


    State y0 = {1.0, 1.0, 1.0};
    State y0_var = {1.0, 1.0, 1.0};

    State rtol = {1e-9, 1e-9, 1e-9};
    State atol = {1e-12, 1e-12, 1e-12};

    chaos::VariationalSolver<Stepper::BDF, T, SD, MyODE, ode::SolverPolicy::Static> solver(
        MyODE{},
        0.0,
        {.main=y0, .variation=y0_var},
        0.1,
        {.main=rtol, .variation=rtol},
        {.main=atol, .variation=atol},
        0.0,
        0.0,
        0.0,
        1
    );


    auto t_start = std::chrono::high_resolution_clock::now();
    solver.advance_until(10000);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Expected Lyapunov exponent: ~0.905\n";
    std::cout <<"Computed Lyapunov exponent: " << solver.lyapunov_exponent() << "\n";
    std::cout <<"Computation completed in " << " " << std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count()*1000 << " ms\n";

}