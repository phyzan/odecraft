// Static events: objective functions fixed at compile time into a StaticEventStepper, located in
// every direction and at every tolerance, several within one step, and through each way of
// building such a stepper.

#include "odecraft_test.hpp"

#include <utility>
#include <vector>

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::ObjFunData;

/// @brief An objective the solver stopped at: its index, and when.
template<typename T>
struct ObjectiveHit {
    int objective;
    T t;
    size_t step;
};

/// @brief Integrates until t_end, collecting every objective met on the way, with the state there.
template<typename Solver, typename T, typename OnHit>
std::vector<ObjectiveHit<T>> record_objectives(Solver& solver, const T& t_end, OnHit&& on_hit){
    // A StaticEventStepper keeps the State alias of its stepper private, hence the auto
    std::vector<ObjectiveHit<T>> hits;
    REQUIRE(solver.advance_until(t_end, [&](const T& t, const auto& q, const T* /*t_ptr*/){
        if (solver.is_at_objective()){
            hits.push_back({solver.current_objective(), t, solver.step_count()});
            on_hit(t, q);
        }
        return true;
    }));
    CHECK(solver.t() == t_end);
    return hits;
}

template<typename Solver, typename T>
std::vector<ObjectiveHit<T>> record_objectives(Solver& solver, const T& t_end){
    return record_objectives(solver, t_end, [](const T&, const auto&){});
}

template<typename T>
std::vector<int> objectives_of(const std::vector<ObjectiveHit<T>>& hits){
    std::vector<int> out;
    for (const auto& hit : hits){
        out.push_back(hit.objective);
    }
    return out;
}

template<typename SD, typename T>
struct LevelOf {
    int component;
    double level;
    T operator()(const T& /*t*/, const state_t<T, SD>& q) const { return q[component] - T(level); }
};

/// @brief A static event stepper of the oscillator with the given objectives.
template<typename Stepper, typename SD, typename T, typename... ObjFun>
auto oscillator_stepper(const T& t0, int dir, ObjFunData<T, ObjFun>... objectives){
    const Settings<T> s = settings<Stepper, T>();
    const auto q0 = Oscillator::at<SD>(t0);
    return ode::getEventStepper<Stepper, SolverPolicy::Static, T, SD>(std::tuple{objectives...}, Oscillator{}, t0, q0,
        uniform_like<SD, T>(q0, s.rtol), uniform_like<SD, T>(q0, s.atol), T(0), T(0), s.stepsize, dir);
}

/// @brief A static event stepper of uniform motion with unit steps, which every stepper integrates exactly.
template<typename Stepper, typename SD, typename T, typename... ObjFun>
auto unit_step_stepper(int dir, ObjFunData<T, ObjFun>... objectives){
    const auto q0 = make_state<SD, T>({0, 0});
    const auto loose = uniform_like<SD, T>(q0, T(1));
    return ode::getEventStepper<Stepper, SolverPolicy::Static, T, SD>(std::tuple{objectives...}, UniformMotion{}, T(0), q0,
        loose, loose, T(0), T(1), T(1), dir);
}

template<typename T>
std::vector<double> objective_tolerances(){
    if constexpr (std::is_same_v<T, float>){
        return {0, 1e-4};
    } else {
        return {0, 1e-9, 1e-4};
    }
}


TEMPLATE_TEST_CASE("Static objectives locate every crossing in the requested direction", "[events][static]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for_each_type<StateKinds<2>>([]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    const double bound = settings<Stepper, T>().error;
                    const T pi = T(acos(T(-1)));
                    for (int dir : {1, -1}){
                        for (int crossing : {1, -1, 0}){
                            for (double tol : objective_tolerances<T>()){
                                CAPTURE(dir, crossing, tol);
                                // x = cos t falls through zero at pi/2 and rises through it at 3pi/2
                                std::vector<T> expected;
                                if (crossing != 1){
                                    expected.push_back(pi/2);
                                }
                                if (crossing != -1){
                                    expected.push_back(3*pi/2);
                                }
                                if (dir == -1){
                                    std::reverse(expected.begin(), expected.end());
                                }

                                auto solver = oscillator_stepper<Stepper, SD>(
                                    dir == 1 ? T(0) : T(5), dir, ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.0}, .ftol = T(tol), .dir = crossing});
                                std::vector<state_t<T, SD>> states;
                                const auto hits = record_objectives(solver, dir == 1 ? T(5) : T(0), [&states](const T&, const auto& q){
                                    states.push_back(q);
                                });

                                REQUIRE(hits.size() == expected.size());
                                for (size_t i = 0; i < hits.size(); i++){
                                    CAPTURE(i);
                                    CHECK(hits[i].objective == 0);
                                    CHECK_THAT(abs_error(hits[i].t, expected[i]), IsAtMost(bound + tol));
                                    CHECK_THAT(abs_error(states[i][0], T(0)), IsAtMost(tol + 100 * epsilon<T>()));
                                    CHECK_THAT(max_error<SD>(states[i], Oscillator::at<SD>(hits[i].t)), IsAtMost(bound));
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("Static objectives within one step are reported in time order", "[events][static][same-step]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const double tol = uniform_motion_tolerance<Stepper, T>();
            for_each_type<StateKinds<2>>([&]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    for (int dir : {1, -1}){
                        CAPTURE(dir);
                        // Unit steps, with x = t and y = -t: x reaches 0.2 and 1.1, y reaches -0.7, each times dir
                        auto solver = unit_step_stepper<Stepper, SD, T>(dir,
                            ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.2 * dir}},
                            ObjFunData<T, LevelOf<SD, T>>{.func = {1, -0.7 * dir}},
                            ObjFunData<T, LevelOf<SD, T>>{.func = {0, 1.1 * dir}});
                        const auto hits = record_objectives(solver, T(2 * dir));

                        REQUIRE(objectives_of(hits) == std::vector<int>{0, 1, 2});
                        const std::vector<double> times = {0.2, 0.7, 1.1};
                        const std::vector<size_t> steps = {1, 1, 2};
                        for (size_t i = 0; i < hits.size(); i++){
                            CAPTURE(i);
                            CHECK_THAT(abs_error(hits[i].t, T(times[i] * dir)), IsAtMost(tol));
                            CHECK(hits[i].step == steps[i]);
                        }
                    }
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("Static objectives that vanish together are each reported", "[events][static][same-step]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            auto solver = unit_step_stepper<Stepper, SD, T>(1,
                ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.5}},
                ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.5}},
                ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.75}});
            const auto hits = record_objectives(solver, T(2));
            REQUIRE(objectives_of(hits) == std::vector<int>{0, 1, 2});
            CHECK(hits[0].t == hits[1].t);
        }
    });
}

TEMPLATE_TEST_CASE("A static crossing just past the end of a step is not missed", "[events][static]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            // The first unit step ends at x = 1, within the tolerance of the root at 1.0001, but short of it
            auto solver = unit_step_stepper<Stepper, SD, T>(1, ObjFunData<T, LevelOf<SD, T>>{.func = {0, 1.0001}, .ftol = T(1e-3), .dir = 1});
            const auto hits = record_objectives(solver, T(3));
            REQUIRE(hits.size() == 1);
            CHECK_THAT(abs_error(hits[0].t, T(1.0001)), IsAtMost(1e-3));
        }
    });
}


TEMPLATE_TEST_CASE("A static event stepper reports its objectives", "[events][static]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto solver = unit_step_stepper<tags::RK45, SD, T>(1,
        ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.3}, .ftol = T(1e-6), .dir = 1},
        ObjFunData<T, LevelOf<SD, T>>{.func = {1, -0.6}, .dir = -1});
    using Solver = decltype(solver);
    STATIC_REQUIRE(Solver::NOBJ == 2);

    CHECK(solver.template objective<0>().ftol == T(1e-6));
    CHECK(solver.template objective<0>().dir == 1);
    CHECK(solver.template objective<1>().dir == -1);
    CHECK(solver.template objective<1>().func.component == 1);
    CHECK(solver.method() == "RK45");

    CHECK_FALSE(solver.is_at_objective());
    CHECK(solver.current_objective() == -1);
    REQUIRE(solver.advance_until(T(0.4)));
    CHECK_FALSE(solver.is_at_objective());   // the step that met the objective has moved on

    solver.Reset();
    const auto hits = record_objectives(solver, T(1));
    CHECK(objectives_of(hits) == std::vector<int>{0, 1});

    auto copy = solver.clone();
    solver.Reset();
    copy->Reset();
    CHECK(objectives_of(record_objectives(*copy, T(1))) == objectives_of(record_objectives(solver, T(1))));
}

TEMPLATE_TEST_CASE("A single objective stepper takes its objective with its settings or a direction alone", "[events][static]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using Objective = LevelOf<SD, T>;
    using Single = ode::SingleStaticEventStepper<tags::RK45, T, SD, Oscillator, SolverPolicy::Static, Objective>;
    const Settings<T> s = settings<tags::RK45, T>();
    const auto q0 = Oscillator::at<SD>(T(0));
    const auto rtol = uniform_like<SD, T>(q0, s.rtol), atol = uniform_like<SD, T>(q0, s.atol);
    // x - 0.5 = cos t - 0.5 falls through zero at pi/3 and rises through it at 5pi/3
    const T pi = T(acos(T(-1)));

    SECTION("with its settings: only the crossings in its direction"){
        Single solver(ObjFunData<T, Objective>{.func = {0, 0.5}, .ftol = T(0), .dir = 1}, Oscillator{}, T(0), q0, rtol, atol);
        const auto hits = record_objectives(solver, T(6));
        REQUIRE(hits.size() == 1);
        CHECK_THAT(abs_error(hits[0].t, T(5*pi/3)), IsAtMost(s.error));
    }
    SECTION("with a direction alone: the crossings in that direction, to machine precision"){
        const std::vector<std::pair<int, std::vector<T>>> crossings = {
            {-1, {T(pi / 3)}},
            {+1, {T(5 * pi / 3)}},
            {0, {T(pi / 3), T(5 * pi / 3)}},
        };
        for (const auto& [dir, times] : crossings){
            CAPTURE(dir);
            Single solver(Objective{0, 0.5}, dir, Oscillator{}, T(0), q0, rtol, atol);
            CHECK(solver.template objective<0>().ftol == T(0));
            CHECK(solver.template objective<0>().dir == dir);
            const auto hits = record_objectives(solver, T(6));
            REQUIRE(hits.size() == times.size());
            for (size_t i = 0; i < hits.size(); i++){
                CHECK_THAT(abs_error(hits[i].t, times[i]), IsAtMost(s.error));
            }
        }
    }
}


// A solver with its system and objectives built in, as a class of its own
template<typename T, typename SD>
class Bouncer : public ode::StaticEventStepper<tags::RK45, T, SD, Oscillator, SolverPolicy::Static, LevelOf<SD, T>, LevelOf<SD, T>> {
    using Base = ode::StaticEventStepper<tags::RK45, T, SD, Oscillator, SolverPolicy::Static, LevelOf<SD, T>, LevelOf<SD, T>>;
public:
    // x = +-0.5, in either direction
    template<typename... Args>
    Bouncer(Args&&... args) : Base(std::tuple{ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.5}}, ObjFunData<T, LevelOf<SD, T>>{.func = {0, -0.5}}},
                                   Oscillator{}, std::forward<Args>(args)...) {}
};

TEMPLATE_TEST_CASE("A stepper class can build its objectives in", "[events][static]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const Settings<T> s = settings<tags::RK45, T>();
    const auto q0 = Oscillator::at<SD>(T(0));
    Bouncer<T, SD> solver(T(0), q0, uniform_like<SD, T>(q0, s.rtol), uniform_like<SD, T>(q0, s.atol));
    // cos t = 0.5 at pi/3, -0.5 at 2pi/3 and 4pi/3, 0.5 again at 5pi/3
    const auto hits = record_objectives(solver, T(6));
    CHECK(objectives_of(hits) == std::vector<int>{0, 1, 1, 0});
}

TEMPLATE_TEST_CASE("A static event stepper works through the virtual interface", "[events][static][virtual]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q0 = make_state<SD, T>({0, 0});
    const auto loose = uniform_like<SD, T>(q0, T(1));
    auto solver = ode::getEventStepper<tags::RK45, SolverPolicy::Virtual, T, SD>(
        std::tuple{ObjFunData<T, LevelOf<SD, T>>{.func = {0, 0.3}}}, UniformMotion{}, T(0), q0, loose, loose, T(0), T(1), T(1), 1);
    ode::OdeSolver<T, SD>& base = solver;

    REQUIRE(base.do_advance());
    CHECK(solver.is_at_objective());
    CHECK_THAT(abs_error(base.get_time(), T(0.3)), IsAtMost(20 * epsilon<T>()));
    REQUIRE(base.do_advance());
    CHECK_FALSE(solver.is_at_objective());
    CHECK(base.get_time() == T(1));
}

} // namespace
} // namespace odetest
