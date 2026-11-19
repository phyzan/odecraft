// The state descriptor contracts: what VectorDescriptor, TensorDescriptor and their Jacobian
// descriptor do on their own, and that the solvers work with all of them, and with a descriptor
// seen through the StateDescriptor contract template.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

#include <odecraft/Contracts/ContractTemplate.hpp>

#include <sstream>

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::contracts::TensorDescriptor;
using ndspan::Layout;


TEMPLATE_TEST_CASE("State descriptors visit check and print their states", "[contracts]", ODECRAFT_TEST_SCALARS){
    using T = TestType;

    for_each_type<StateKinds<3>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            auto a = make_state<SD, T>({1, 2, 3});
            const auto b = make_state<SD, T>({10, 20, 30});

            // for_each walks several states in lockstep, writing through the first
            SD::for_each([](T& a_i, const T& b_i){ a_i += b_i; }, a, b);
            CHECK(same_state<SD>(a, make_state<SD, T>({11, 22, 33})));

            CHECK(SD::all_true([](const T& x){ return x > 10; }, a));
            CHECK_FALSE(SD::all_true([](const T& x){ return x > 11; }, a));
            CHECK(to_double(SD::template norm_sqr<T>(T(-3))) == 9);

            state_t<T, SD> resized{};
            SD::resize_like(resized, a);
            CHECK(resized.size() == 3);

            std::ostringstream out;
            SD::console(out, a);
            CHECK(out.str() == "11 22 33");
        }
    });
}

TEMPLATE_TEST_CASE("Tensor descriptors keep the shape of their states", "[contracts]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = TensorDescriptor<Layout::C>;
    STATIC_REQUIRE(SD::SIZE == 0);
    STATIC_REQUIRE(TensorDescriptor<Layout::C, 2, 3>::SIZE == 6);

    state_t<T, SD> ref(2, 3);
    ref.fill(T(1));
    state_t<T, SD> resized{};
    SD::resize_like(resized, ref);
    REQUIRE(resized.ndim() == 2);
    CHECK((resized.shape(0) == 2 && resized.shape(1) == 3));

    T sum = 0;
    SD::for_each([&sum](const T& x){ sum += x; }, ref);
    CHECK(to_double(sum) == 6);
}


/// @brief A 3x3 system that needs a row exchange: its leading entry is zero.
template<typename SD, typename T>
void fill_pivoting_matrix(jacmat_t<T, SD>& A){
    const int rows[3][3] = {{0, 2, 1}, {1, 1, 1}, {2, 1, 0}};
    for (size_t i = 0; i < 3; i++){
        for (size_t j = 0; j < 3; j++){
            A(i, j) = rows[i][j];
        }
    }
}

TEMPLATE_TEST_CASE("The Jacobian descriptor solves and multiplies with its matrices", "[contracts][linalg]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    const double tol = 100 * epsilon<T>();

    for_each_type<StateKinds<3>>([&]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            using JD = typename SD::JD;
            const auto x_exact = make_state<SD, T>({1, -2, 3});
            auto A = make_jacmat<SD, T>(x_exact);
            fill_pivoting_matrix<SD, T>(A);
            // b = A * x_exact = (-1, 2, 0)
            const auto b = make_state<SD, T>({-1, 2, 0});

            SECTION("dot and dot_add multiply with the matrix"){
                auto out = make_state<SD, T>({0, 0, 0});
                JD::linalg::dot(out, A, x_exact);
                CHECK_THAT(max_error<SD>(out, b), IsAtMost(tol));

                JD::linalg::dot_add(out, T(2), A, x_exact);
                CHECK_THAT(max_error<SD>(out, make_state<SD, T>({-3, 6, 0})), IsAtMost(tol));
            }
            SECTION("identity_minus forms I - c*J, in place too"){
                auto M = A;
                JD::linalg::identity_minus(M, T(2), A);
                CHECK_THAT(abs_error(M(0, 0), T(1)), IsAtMost(tol));
                CHECK_THAT(abs_error(M(0, 1), T(-4)), IsAtMost(tol));
                CHECK_THAT(abs_error(M(2, 2), T(1)), IsAtMost(tol));

                JD::linalg::identity_minus(M, T(1), M);
                CHECK_THAT(abs_error(M(0, 0), T(0)), IsAtMost(tol));
                CHECK_THAT(abs_error(M(0, 1), T(4)), IsAtMost(tol));
            }
            SECTION("scale multiplies every entry, in place too"){
                auto M = A;
                JD::scale(M, T(-3), A);
                CHECK_THAT(abs_error(M(1, 2), T(-3)), IsAtMost(tol));
                JD::scale(M, T(2), M);
                CHECK_THAT(abs_error(M(1, 2), T(-6)), IsAtMost(tol));
                CHECK(JD::all_true([](const T& x){ return x <= 0; }, M));
            }
            SECTION("lu_factor and lu_solve solve the system, in place too"){
                typename JD::linalg::pivots_type piv{};
                auto LU = A;
                REQUIRE(JD::linalg::lu_factor(LU, piv));

                auto x = make_state<SD, T>({0, 0, 0});
                JD::linalg::lu_solve(x, LU, piv, b);
                CHECK_THAT(max_error<SD>(x, x_exact), IsAtMost(tol));

                auto in_place = b;
                JD::linalg::lu_solve(in_place, LU, piv, in_place);
                CHECK_THAT(max_error<SD>(in_place, x_exact), IsAtMost(tol));
            }
            SECTION("lu_factor reports a singular matrix"){
                auto S = A;
                for (size_t j = 0; j < 3; j++){
                    S(2, j) = S(0, j) + S(1, j);
                }
                typename JD::linalg::pivots_type piv{};
                CHECK_FALSE(JD::linalg::lu_factor(S, piv));
            }
        }
    });
}

TEMPLATE_TEST_CASE("The Jacobian descriptor fills matrices by rows and by columns", "[contracts]", ODECRAFT_TEST_SCALARS){
    using T = TestType;

    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            using JD = typename SD::JD;
            const auto like = make_state<SD, T>({0, 0});
            auto J = make_jacmat<SD, T>(like);
            auto K = make_jacmat<SD, T>(like);

            // Row i of the result is (10*i, 10*i + 1); the states visited are the row indices
            const auto index = make_state<SD, T>({0, 1});
            auto row = like;
            JD::for_each_row([](T& J_ij, const T& r_j){ J_ij = r_j; }, [&row](const T& i){
                row[0] = 10*i;
                row[1] = 10*i + 1;
                return std::tie(row);
            }, std::tie(J), index);
            CHECK(to_double(J(1, 0)) == 10);
            CHECK(to_double(J(1, 1)) == 11);
            CHECK(to_double(J(0, 1)) == 1);

            // By columns, into two matrices at once: the first gets the column, the second its double
            auto col = like, col2 = like;
            JD::for_each_column([](T& M_ij, const T& c_i){ M_ij = c_i; }, [&](const T& j){
                col[0] = j;
                col[1] = j + 100;
                col2[0] = 2*col[0];
                col2[1] = 2*col[1];
                return std::tie(col, col2);
            }, std::tie(J, K), index);
            CHECK(to_double(J(0, 1)) == 1);
            CHECK(to_double(J(1, 1)) == 101);
            CHECK(to_double(K(1, 0)) == 200);
        }
    });
}


// Two independent oscillators in the columns of a 2x2 state: the flat components 0-1 and 2-3
struct TwoOscillators {
    static void Rhs(auto& out, const auto& /*t*/, const auto& q){
        out[0] = q[1]; out[1] = -q[0];
        out[2] = q[3]; out[3] = -q[2];
    }
};

template<typename SD, typename T>
state_t<T, SD> two_oscillators_ics(){
    state_t<T, SD> q0 = [](){
        if constexpr (is_runtime_sized<SD>){
            return state_t<T, SD>(2, 2);
        } else {
            return state_t<T, SD>{};
        }
    }();
    // (cos t, -sin t) in the first oscillator, (sin t, cos t) in the second
    q0[0] = 1; q0[1] = 0; q0[2] = 0; q0[3] = 1;
    return q0;
}

TEMPLATE_TEST_CASE("Steppers integrate multi-dimensional states of any layout", "[contracts][steppers]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using Tensors = std::tuple<
        TensorDescriptor<Layout::C, 2, 2>,  // static shape
        TensorDescriptor<Layout::F, 2, 2>,  // static shape, column-major
        TensorDescriptor<Layout::C>         // runtime shape and number of dims
    >;
    for_each_type<Tensors>([]<typename SD>(){
        const bool column_major = std::is_same_v<SD, std::tuple_element_t<1, Tensors>>;
        DYNAMIC_SECTION(kind_name<SD> << (column_major ? " in column-major layout" : "")){
            for_each_type<AllSteppers>([]<typename Stepper>(){
                DYNAMIC_SECTION(Stepper::name){
                    auto solver = make_solver<Stepper, SD, SolverPolicy::Static>(TwoOscillators{}, T(0), two_oscillators_ics<SD, T>());
                    REQUIRE(solver.advance_until(T(2)));

                    const auto q = current_state(solver);
                    REQUIRE(q.ndim() == 2);
                    CHECK((q.shape(0) == 2 && q.shape(1) == 2));
                    const T c = cos(T(2));
                    const T s = sin(T(2));
                    const double err = settings<Stepper, T>().error;
                    CHECK_THAT(abs_error(q[0], c), IsAtMost(err));
                    CHECK_THAT(abs_error(q[1], T(-s)), IsAtMost(err));
                    CHECK_THAT(abs_error(q[2], s), IsAtMost(err));
                    CHECK_THAT(abs_error(q[3], c), IsAtMost(err));
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("A descriptor seen through the contract template drives every stepper", "[contracts][steppers]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    // ode::StateDescriptor forwards every requirement of the contract to the descriptor it wraps
    using SD = ode::StateDescriptor<StaticSize<2>>;
    STATIC_REQUIRE(std::is_same_v<state_t<T, SD>, std::array<T, 2>>);

    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            auto solver = make_solver<Stepper, SD, SolverPolicy::Static>(Oscillator{}, T(0), Oscillator::at<SD>(T(0)));
            REQUIRE(solver.advance_until(T(3)));
            CHECK_THAT(max_error<SD>(current_state(solver), Oscillator::at<SD>(T(3))), IsAtMost(settings<Stepper, T>().error));
        }
    });
}

} // namespace
} // namespace odetest
