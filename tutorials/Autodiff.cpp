#include <iostream>
#include <odecraft/odecraft.hpp>

using namespace ode;

using SD = contracts::common::SD<3>;
using State = common::State<double, 3>;
using JacMat = jacmat_t<double, SD>;

struct LorenzSimple{

    // Simple Rhs, using the plainest signature available for the solvers
    static void Rhs(State& dy_dt, double /*t*/, const State& q) {
        //3D lorenz system, args = {sigma, rho, beta}
        const double sigma = 10.0;
        const double rho = 28.0;
        const double beta = 8.0/3.0;
        dy_dt[0] = sigma*(q[1] - q[0]);
        dy_dt[1] = q[0]*(rho - q[2]) - q[1];
        dy_dt[2] = q[0]*q[1] - beta*q[2];
    }

};


struct LorenzJacobian : public LorenzSimple {


    // Providing both the Rhs (from inheritance) and the Jacobian for the Lorenz system
    static void Jac(JacMat& J, double /*t*/, const State& q) {
        constexpr double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
        // J(i, j) = df_i/dq_j, written column by column: J is column-major
        J(0, 0) = -sigma;  J(1, 0) = rho - q[2];  J(2, 0) = q[1];   // d/dx
        J(0, 1) = sigma;   J(1, 1) = -1.0;        J(2, 1) = q[0];   // d/dy
        J(0, 2) = 0.0;     J(1, 2) = -q[0];       J(2, 2) = -beta;  // d/dz
    }

};

struct LorenzAutodiff{

    // This generic Rhs function supports automatic differentiation via dual numbers if needed
    // Significantly speeds up the BDF solver when compared to using finite-difference approximation for the Jacobian
    static void Rhs(auto& dy_dt, const auto& /*t*/, const auto& q) {
        //3D lorenz system, args = {sigma, rho, beta}
        const double sigma = 10.0;
        const double rho = 28.0;
        const double beta = 8.0/3.0;
        dy_dt[0] = sigma*(q[1] - q[0]);
        dy_dt[1] = q[0]*(rho - q[2]) - q[1];
        dy_dt[2] = q[0]*q[1] - beta*q[2];
    }

};

template<typename OdeType>
void test_solver(const char* label) {

    using T = double;

    State y0 = {1.0, 1.0, 1.0};
    State y0_var = {1.0, 1.0, 1.0};
    State rtol = {1e-9, 1e-9, 1e-9};
    State atol = {1e-12, 1e-12, 1e-12};

    chaos::common::VarState<T, 3> q0{.main=y0, .variation=y0_var};
    chaos::common::VarState<T, 3> var_rtol{.main=rtol, .variation=rtol};
    chaos::common::VarState<T, 3> var_atol{.main=atol, .variation=atol};

    auto solver = chaos::common::getVariationalSolver<ode::SolverPolicy::Static, Stepper::BDF>(
        OdeType{},
        0.0,
        q0,
        0.1,
        var_rtol,
        var_atol,
        0.0,
        0.0,
        0.0,
        1
    );


    std::cout << "\n------- Running test for case: " << "\033[1m" << label << "\033[0m" << "-------" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    solver.advance_until(100);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Expected Lyapunov exponent: ~0.905\n";
    std::cout << "Computed Lyapunov exponent: " << solver.lyapunov_exponent() << "\n";
    std::cout << "Computation completed in " << "\033[1m"
              << std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count()*1000
              << " ms\033[0m\n\n";
}







int main(){

    std::cout << "\n---------- Testing VariationalSolver ------------------\n" << std::endl;

    std::cout << " (This might take too long if compiled in debug mode)" << std::endl;
    test_solver<LorenzSimple>("Finite differences for Jacobian");
    test_solver<LorenzJacobian>("Exact Jacobian");
    test_solver<LorenzAutodiff>("Automatic differentiation for Jacobian");

}

/*
g++ -std=c++20 -O3 -Iinclude -Iexternal/xdiff/include -Iexternal/xdiff/external/lazex/include -Iexternal/polybox/include -Iexternal/ndspan/include tutorials/Autodiff.cpp -o autodiff && ./autodiff
*/