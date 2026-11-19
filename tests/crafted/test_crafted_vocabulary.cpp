// The vocabulary of the compiled interface: its states and callables, its scalar types and
// their precision, the stepper catalog, and the names it re-exports, which must be the very
// ones of the header-only interface they stand for.

#include "crafted_test.hpp"

#include <functional>
#include <type_traits>

namespace odetest {
namespace {

using namespace ode::crafted;

TEMPLATE_TEST_CASE("A state has a runtime shape of any rank", "[crafted][vocabulary]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    STATIC_REQUIRE(std::is_same_v<SD, ode::contracts::TensorDescriptor<ndspan::Layout::C>>);
    STATIC_REQUIRE(std::is_same_v<VarSD, ode::contracts::VarSD<SD>>);
    STATIC_REQUIRE(std::is_same_v<State<T>, ode::state_t<T, SD>>);
    STATIC_REQUIRE(std::is_same_v<VarState<T>, ode::state_t<T, VarSD>>);
    STATIC_REQUIRE(std::is_same_v<JacMat<T>, ode::jacmat_t<T, SD>>);

    const State<T> line = {T(1), T(2), T(3)};
    CHECK(line.ndim() == 1);
    CHECK(line.size() == 3);
    CHECK(line[2] == T(3));

    State<T> table(2, 3);
    CHECK(table.ndim() == 2);
    CHECK(table.shape(0) == 2);
    CHECK(table.shape(1) == 3);
    CHECK(table.size() == 6);
    table(1, 2) = T(5);
    CHECK(table[5] == T(5)); // in C order

    const VarState<T> both{.main = line, .variation = filled_like(line, T(0))};
    CHECK(both.main.size() == 3);
    CHECK(both.variation[1] == T(0));
}

TEMPLATE_TEST_CASE("Callables are type-erased", "[crafted][vocabulary]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    STATIC_REQUIRE(std::is_same_v<rhs_t<T>, std::function<void(State<T>&, T, const State<T>&)>>);
    STATIC_REQUIRE(std::is_same_v<rhs_t<T, VarState<T>>, std::function<void(VarState<T>&, T, const VarState<T>&)>>);
    STATIC_REQUIRE(std::is_same_v<jac_t<T>, std::function<void(JacMat<T>&, T, const State<T>&)>>);
    STATIC_REQUIRE(std::is_same_v<objfun_t<T>, std::function<T(const T&, const State<T>&)>>);
    STATIC_REQUIRE(std::is_same_v<interp_t<T>, std::function<void(State<T>&, const T&)>>);
    STATIC_REQUIRE(std::is_same_v<observer_t<T>, std::function<bool(const T&, const State<T>&, const T*)>>);
    STATIC_REQUIRE(std::is_same_v<ode_t<T>, ode::OdeData<rhs_t<T>, jac_t<T>>>);

    // A system is written inline, and may leave its Jacobian out. Two components, as GCC 13 takes
    // the copy of a one-element array for out of bounds
    const ode_t<T> system{.Rhs = [](State<T>& out, T t, const State<T>& q){ out[0] = t * q[0]; out[1] = q[0] + q[1]; }};
    CHECK(system.Jac == nullptr);
    State<T> out = {T(0), T(0)};
    system.Rhs(out, T(2), State<T>{T(3), T(4)});
    CHECK(out[0] == T(6));
    CHECK(out[1] == T(7));
}

TEST_CASE("The multiprecision type is the one the library was built with", "[crafted][vocabulary][precision]"){
#ifdef ODECRAFT_USE_LAZY_MPREAL
    STATIC_REQUIRE(std::is_same_v<mpreal_t, lazex::LazyType<mpfr::mpreal>>);
#else
    STATIC_REQUIRE(std::is_same_v<mpreal_t, mpfr::mpreal>);
#endif
    STATIC_REQUIRE(is_multiprecision<mpreal_t>);
}

TEST_CASE("set_mpreal_prec sets the precision of new multiprecision values", "[crafted][vocabulary][precision]"){
    // Restores the precision of the other tests however this one ends
    struct Restore {
        mpfr_prec_t bits = get_default_prec();
        ~Restore(){ set_mpreal_prec(bits); }
    } restore;
    CHECK(restore.bits == mpreal_bits);

    set_mpreal_prec(200);
    CHECK(get_default_prec() == 200);
    const mpreal_t third = mpreal_t(1) / mpreal_t(3);
    CHECK(static_cast<const mpfr::mpreal&>(third).get_prec() == 200);
    // A third is now right to 200 bits, well past what 128 can hold
    CHECK(epsilon<mpreal_t>() < 1e-59);
}

TEST_CASE("The re-exported names are those of the header-only interface", "[crafted][vocabulary]"){
    STATIC_REQUIRE(std::is_same_v<StepResult, ode::StepResult>);
    STATIC_REQUIRE(std::is_same_v<JacPolicy, ode::JacPolicy>);
    STATIC_REQUIRE(std::is_same_v<RootPolicy, ode::RootPolicy>);
    STATIC_REQUIRE(std::is_same_v<EventOptions, ode::EventOptions>);
    STATIC_REQUIRE(std::is_same_v<EventPolicy, ode::EventPolicy>);
    STATIC_REQUIRE(std::is_same_v<EventCounter<double, SD>, ode::EventCounter<double, SD>>);
    STATIC_REQUIRE(std::is_same_v<Interval<double>, ode::interp::uni::Interval<double>>);
    STATIC_REQUIRE(std::is_same_v<Steppers, ode::Steppers>);
    STATIC_REQUIRE(std::is_same_v<tags::RK45, ode::tags::RK45>);
    STATIC_REQUIRE(std::is_same_v<OdeData<rhs_t<double>, jac_t<double>>, ode::OdeData<rhs_t<double>, jac_t<double>>>);
    STATIC_REQUIRE(std::is_same_v<Vector<double, 3>, ode::Vector<double, 3>>);

    STATIC_REQUIRE(std::is_same_v<Allocation, ndspan::Allocation>);
    STATIC_REQUIRE(std::is_same_v<Layout, ndspan::Layout>);
    STATIC_REQUIRE(std::is_same_v<Array<double>, ndspan::Array<double>>);
    STATIC_REQUIRE(std::is_same_v<Array1D<double, 0>, ndspan::Array1D<double, 0>>);
    STATIC_REQUIRE(std::is_same_v<Array2D<double, 0, 0, Layout::C>, ndspan::Array2D<double, 0, 0, Layout::C>>);
    STATIC_REQUIRE(std::is_same_v<View<const double>, ndspan::View<const double>>);
    STATIC_REQUIRE(std::is_same_v<View1D<double, 0, Layout::C>, ndspan::View1D<double, 0, Layout::C>>);
    STATIC_REQUIRE(std::is_same_v<View2D<double, 0, 0, Layout::C>, ndspan::View2D<double, 0, 0, Layout::C>>);

    const auto v = make_vector<double>(1.0, 2.0, 3.0);
    CHECK(v.size() == 3);
    CHECK(v[2] == 3.0);
}

TEST_CASE("The stepper catalog finds every compiled stepper by its name and its id", "[crafted][vocabulary][catalog]"){
    for (const std::string& name : stepper_names){
        CAPTURE(name);
        const Steppers::Id id = Steppers::getID(name);
        CHECK(Steppers::getName(id) == name);
    }
    CHECK(Steppers::id<tags::Euler> == Steppers::getID("Euler"));
    CHECK(Steppers::id<tags::RK4> == Steppers::getID("RK4"));
    CHECK(Steppers::id<tags::RK23> == Steppers::getID("RK23"));
    CHECK(Steppers::id<tags::RK45> == Steppers::getID("RK45"));
    CHECK(Steppers::id<tags::DOP853> == Steppers::getID("DOP853"));
    CHECK(Steppers::id<tags::BDF> == Steppers::getID("BDF"));
    CHECK(Steppers::id<tags::RK45> != Steppers::id<tags::DOP853>);
}

} // namespace
} // namespace odetest
