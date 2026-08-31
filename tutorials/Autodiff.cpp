#include <iostream>
#include <odecraft/odecraft.hpp>

using namespace ode;

struct LorenzSimple{

    // Simple Rhs, using the plainest signature available for the solvers
    static void Rhs(double* dy_dt, double /*t*/, const double* q) {
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
    static void Jac(double* J, double /*t*/, const double* q) {
        constexpr double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
        // column-major: J[i + 3*j] = df_i/dq_j
        J[0] = -sigma;   J[1] = rho - q[2];  J[2] = q[1];   // d/dx
        J[3] = sigma;    J[4] = -1.0;    J[5] = q[0];   // d/dy
        J[6] = 0.0;  J[7] = -q[0];       J[8] = -beta;  // d/dz
    }

};

struct LorenzAutodiff{

    // This generic Rhs function supports automatic differentiation via dual numbers if needed
    // Significantly speeds up the BDF solver when compared to using finite-difference approximation for the Jacobian
    static void Rhs(auto* dy_dt, const auto& /*t*/, auto q) {
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

    static constexpr size_t NSYS = 3;

    using T = double;

    std::array<T, 3> y0 = {1.0, 1.0, 1.0};
    std::array<T, 3> y0_var = {1.0, 1.0, 1.0};

    chaos::VariationalSolver<Stepper::BDF, T, NSYS, ode::SolverPolicy::Static, OdeType> solver(
        OdeType{},
        0.0,
        View1D<T, NSYS>{y0.data()},
        View1D<T, NSYS>{y0_var.data()},
        0.1,
        1e-9,
        1e-12,
        0.0,
        0.0,
        0.0,
        1
    );


    std::cout << "\n------- Running test for case: " << "\033[1m" << label << "\033[0m" << "-------" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    solver.advance_until(100);
    auto t_end = std::chrono::high_resolution_clock::now();
    print("Expected Lyapunov exponent: ~0.905");
    print("Computed Lyapunov exponent: ", solver.lyapunov_exponent());
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
g++ -std=c++20 -O3 -Iinclude -Iexternal/xdiff/include -Iexternal/xdiff/external/lazy/include -Iexternal/polybox/include -Iexternal/ndspan/include tutorials/Autodiff.cpp -o autodiff && ./autodiff
*/