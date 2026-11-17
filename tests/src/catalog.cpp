#include "odecraft/Contracts/DenseDescriptor.hpp"
#include <odecraft/odecraft.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

/*
Steppers are types, and a Catalog of them is what chooses one at runtime, so that the set of
steppers is open: one written outside the library is declared with ODECRAFT_DECLARE_STEPPER,
listed in a catalog next to the built-in ones, and used through every factory like them.

The steppers below live in a namespace of their own on purpose: the tag machinery has to work
for steppers outside namespace ode, whose tags still go in ode::tags.
*/

namespace mylib{

// Explicit midpoint method (2nd order, fixed step), a stepper the library does not have
template<typename T, typename SD, typename OdeType, ode::SolverPolicy SP, typename Derived = void>
class Midpoint : public ode::detail::BaseDispatcher<ode::GetDerived<Midpoint<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>{

    ODECRAFT_DERIVED_STEPPER;

    using Base = ode::detail::BaseDispatcher<ode::GetDerived<Midpoint<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>;

public:
    using State = ode::state_t<T, SD>;
    using RealState = ode::real_state_t<T, SD>;

    static constexpr int ERR_EST_ORDER = 2;
    static constexpr bool IS_IMPLICIT = false;

    Midpoint(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T /*min_step*/=0, T /*max_step*/=0, T stepsize=0, int dir=1, ode::EventList<T, State> events = {})
        : Base(std::move(ode), std::move(t0), q0, std::move(rtol), std::move(atol), 0, 0, stepsize, dir, std::move(events)), k_(q0) {}

protected:

    ode::StepResult adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q){
        const T h = habs * this->direction();
        t_new = t + h;
        habs_new = habs;
        // q_new = q + h * f(t + h/2, q + h/2 * f(t, q))
        this->rhs(k_, t, q);
        SD::for_each([&h](auto& k_i, const auto& q_i){
            k_i = q_i + k_i * (h/2);
        }, k_, q);
        this->rhs(q_new, t + h/2, k_);
        SD::for_each([&h](auto& q_new_i, const auto& q_i){
            q_new_i = q_i + q_new_i * h;
        }, q_new, q);
        return ode::all_are_finite<T, SD>(q_new) ? ode::StepResult::Success : ode::StepResult::NonFiniteError;
    }

    auto local_interp_impl() const{
        return ode::make_light_functor([](State& out, const T& t, const auto& t1, const auto& t2, const auto& y1, const auto& y2){
            return ode::interp::uni::lin_interp<T, SD>(out, t, t1, t2, y1, y2);
        }, this->t_old(), this->t_new(), this->vector_old(), this->vector_new());
    }

private:
    State k_;
};


// A stepper derived from a built-in one, under a tag of its own
template<typename T, typename SD, typename OdeType, ode::SolverPolicy SP, typename Derived = void>
class TracedRK45 : public ode::RK45<T, SD, OdeType, SP, ode::GetDerived<TracedRK45<T, SD, OdeType, SP, Derived>, Derived>>{

    ODECRAFT_DERIVED_STEPPER;

    using Base = ode::RK45<T, SD, OdeType, SP, ode::GetDerived<TracedRK45<T, SD, OdeType, SP, Derived>, Derived>>;

public:
    using Base::Base;
};

} // namespace mylib

// Their tags go in ode::tags, next to those of the built-in steppers
ODECRAFT_DECLARE_STEPPER(Midpoint, mylib::Midpoint)
ODECRAFT_DECLARE_STEPPER(TracedRK45, mylib::TracedRK45)


using namespace ode;

namespace {

using T = double;
using SD = contracts::VectorDescriptor<2>;
using State = state_t<T, SD>;

// Harmonic oscillator: from (1, 0), x(t) = cos(t)
struct Oscillator{
    void Rhs(State& out, const T& /*t*/, const State& q) const {
        out[0] = q[1];
        out[1] = -q[0];
    }
};

// The built-in steppers, a user-defined one and one derived from a built-in one
using MySteppers = Catalog<tags::RK45, tags::Midpoint, tags::BDF, tags::TracedRK45>;

int failures = 0;

void check(bool ok, const std::string& what){
    if (ok){
        std::cout << "  ok   : " << what << std::endl;
    } else {
        std::cerr << "  FAIL : " << what << std::endl;
        ++failures;
    }
}

template<typename Exception, typename Callable>
void check_throws(Callable&& f, const std::string& what){
    try {
        f();
        check(false, what);
    } catch (const Exception&) {
        check(true, what);
    }
}


// The ids are compile-time constants, in the order the steppers are listed
static_assert(MySteppers::id<tags::RK45>.index() == 0);
static_assert(MySteppers::id<tags::Midpoint>.index() == 1);
static_assert(MySteppers::id<tags::BDF>.index() == 2);
static_assert(MySteppers::id<tags::TracedRK45>.index() == 3);
static_assert(Steppers::id<tags::Euler>.index() == 0 && Steppers::id<tags::BDF>.index() == 5);
static_assert(MySteppers::id<tags::BDF> != MySteppers::id<tags::RK45>);

// An id is not an integer: only its catalog makes one, so it always stands for one of its
// steppers, and each catalog has an Id type of its own
template<typename C, typename Arg>
concept takes_as_id = requires(const Arg& arg) { C::getName(arg); };

static_assert(takes_as_id<MySteppers, MySteppers::Id>);
static_assert(!takes_as_id<MySteppers, int> && !takes_as_id<MySteppers, size_t>);
static_assert(!takes_as_id<MySteppers, Steppers::Id> && !takes_as_id<Steppers, MySteppers::Id>);
static_assert(!std::is_constructible_v<MySteppers::Id, size_t> && !std::is_default_constructible_v<MySteppers::Id>);
static_assert(!std::is_convertible_v<MySteppers::Id, size_t>);
static_assert(std::is_trivially_copyable_v<MySteppers::Id> && sizeof(MySteppers::Id) == sizeof(size_t));

static_assert(detail::isStepper<tags::Midpoint> && detail::isStepper<tags::RK45> && !detail::isStepper<Steppers::Id>);

// The tag of a stepper has its name, and is in ode::tags wherever the stepper is
static_assert(std::is_same_v<tags::RK45, detail::StepperTag<RK45, "RK45">>);
static_assert(std::is_same_v<tags::Midpoint, detail::StepperTag<mylib::Midpoint, "Midpoint">>);
static_assert(tags::Midpoint::name == "Midpoint" && tags::DOP853::name == "DOP853");

// clone() points to the solver itself for the static policies, and to OdeSolver for the virtual
// ones, while virtual_base is the most specific interface
using StaticRK45 = RK45<T, SD, Oscillator, SolverPolicy::Static>;
using RichRK45 = RK45<T, SD, Oscillator, SolverPolicy::RichVirtual>;
static_assert(std::is_void_v<StaticRK45::virtual_base>);
static_assert(std::is_same_v<StaticRK45::clone_type, StaticRK45>);
static_assert(std::is_same_v<RichRK45::virtual_base, OdeRichSolver<T, SD>>);
static_assert(std::is_same_v<RichRK45::clone_type, OdeSolver<T, SD>>);

// A variational solver is reached through ChaoticSolver, whatever stepper it is built on
using VarMidpoint = chaos::VariationalSolver<tags::Midpoint, T, SD, Oscillator, SolverPolicy::Virtual>;
static_assert(std::is_same_v<VarMidpoint::virtual_base, chaos::ChaoticSolver<T, SD, UtilPolicy::Virtual>>);
static_assert(std::is_base_of_v<chaos::ChaoticSolver<T, SD, UtilPolicy::Virtual>, VarMidpoint>);
static_assert(std::is_same_v<VarMidpoint::clone_type, OdeSolver<T, chaos::VarSD<SD>>>);
static_assert(std::is_void_v<chaos::VariationalSolver<tags::RK45, T, SD, Oscillator, SolverPolicy::Static>::virtual_base>);


void run_names(){
    std::cout << "\n-- ids and names --" << std::endl;

    for (const std::string name : {"Euler", "RK4", "RK23", "RK45", "DOP853", "BDF"}){
        check(Steppers::getName(Steppers::getID(name)) == name, name + " round-trips through Steppers");
    }
    check(MySteppers::getID("Midpoint") == MySteppers::id<tags::Midpoint>, "a user-defined stepper is found by name");
    check(MySteppers::getName(MySteppers::id<tags::TracedRK45>) == "TracedRK45", "a derived stepper keeps the name of its own tag");
    check(MySteppers::from_stepper<std::string_view>(MySteppers::id<tags::Midpoint>, []<typename S>(){ return S::name; }) == "Midpoint",
          "from_stepper hands over the tag of the id");

    check_throws<std::invalid_argument>([]{ (void)MySteppers::getID("Euler"); }, "a stepper not in the catalog is not found by name");
}


// Every stepper of the catalog, chosen at runtime, against cos(t)
void run_runtime_choice(){
    std::cout << "\n-- runtime choice through a catalog --" << std::endl;

    const State q0 = {1.0, 0.0};
    const State tol = {1e-10, 1e-10};
    const T tmax = 2.0;

    const std::array ids{MySteppers::id<tags::RK45>, MySteppers::id<tags::Midpoint>,
                         MySteppers::id<tags::BDF>, MySteppers::id<tags::TracedRK45>};
    for (const MySteppers::Id id : ids){
        const std::string name(MySteppers::getName(id));
        BoxedSolver<T, SD, UtilPolicy::RichVirtual> solver = MySteppers::make<UtilPolicy::RichVirtual, SD, T>(
            id, Oscillator{}, 0.0, q0, tol, tol, 0.0, 0.0, 1e-4, 1);
        check(solver->get_method() == name, "the solver made from id " + std::to_string(id.index()) + " is " + name);

        solver->do_advance_until(tmax);
        State q = q0;
        solver->get_current_vector(q);
        check(std::abs(q[0] - std::cos(tmax)) < 1e-6, name + " integrates x(2) = cos(2)");

        // A clone is the same stepper
        std::unique_ptr<OdeSolver<T, SD>> copy = solver->clone();
        check(copy->get_method() == name && copy->get_time() == solver->get_time(), name + " clones into the same stepper");
    }
}


// The tag of a user-defined stepper, at compile time, through each factory
void run_compile_time_choice(){
    std::cout << "\n-- a user-defined stepper through the factories --" << std::endl;

    const State q0 = {1.0, 0.0};
    const State tol = {1e-10, 1e-10};

    auto solver = getSolver<tags::Midpoint, SolverPolicy::Static, T, SD>(Oscillator{}, 0.0, q0, tol, tol, 0.0, 0.0, 1e-3, 1);
    check(solver.method() == "Midpoint", "getSolver builds the user-defined stepper");
    std::unique_ptr<decltype(solver)> copy = solver.clone();
    check(copy->method() == "Midpoint", "a static solver clones into its own type");

    auto boxed = make_rich_vsolver<SD>(tags::Midpoint{}, Oscillator{}, 0.0, q0, tol, tol, 0.0, 0.0, 1e-3, 1);
    check(boxed->get_method() == "Midpoint", "make_rich_vsolver takes the tag of a user-defined stepper");

    auto derived = make_vsolver<SD>(tags::TracedRK45{}, Oscillator{}, 0.0, q0, tol, tol);
    check(derived->get_method() == "TracedRK45", "a stepper derived from RK45 reports its own name");

    ODE<T, SD> ode(tags::Midpoint{}, Oscillator{}, 0.0, q0, tol, tol, 0.0, 0.0, 1e-3, 1);
    check(ode.solver()->get_method() == "Midpoint", "the ODE driver takes the tag of a user-defined stepper");

    // A variational solver built on it reports it, and is reached through ChaoticSolver
    using VarState = state_t<T, chaos::VarSD<SD>>;
    const VarState vq0{.main = q0, .variation = {1.0, 0.0}};
    const VarState vtol{.main = tol, .variation = tol};
    pbox::Box<chaos::ChaoticSolver<T, SD, UtilPolicy::Virtual>> var = chaos::make_variational_solver<UtilPolicy::Virtual, SD>(
        tags::Midpoint{}, Oscillator{}, 0.0, vq0, 1.0, vtol, vtol, 0.0, 0.0, 1e-3, 1);
    check(var->get_method() == "Midpoint", "a variational solver reports the stepper it is built on");
    var->do_advance_until(10.0);
    check(std::abs(var->get_lyapunov_exponent()) < 1e-2, "the Lyapunov exponent of the oscillator is ~0 with the user-defined stepper");

    auto var_static = chaos::getVariationalSolver<SolverPolicy::Static, SD>(tags::RK45{}, Oscillator{}, 0.0, vq0, 1.0, vtol, vtol);
    check(var_static.method() == "RK45", "getVariationalSolver takes the tag of its stepper");

    // So does a solver with compile-time objectives
    auto obj = [](const T& /*t*/, const State& q) -> T { return q[0]; };
    auto event_stepper = getEventStepper<tags::Midpoint, SolverPolicy::Static, T, SD>(
        std::tuple{ObjFunData<T, decltype(obj)>{.func = obj}}, Oscillator{}, 0.0, q0, tol, tol, 0.0, 0.0, 1e-3, 1);
    check(event_stepper.method() == "Midpoint", "a static event stepper reports the stepper it is built on");
}

} // namespace


void test_catalog(){
    std::cout << "\n========== Testing stepper catalogs ==========" << std::endl;
    failures = 0;

    run_names();
    run_runtime_choice();
    run_compile_time_choice();

    if (failures == 0){
        std::cout << "\nAll catalog checks passed." << std::endl;
    } else {
        std::cerr << "\n" << failures << " catalog check(s) FAILED." << std::endl;
    }
}
