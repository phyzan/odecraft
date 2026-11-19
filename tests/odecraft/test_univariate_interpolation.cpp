// The building blocks of dense output: intervals with open, closed and linking ends, the
// interpolator of a single step, and the chain of them that covers a whole integration.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

#include <stdexcept>

namespace odetest {
namespace {

using ode::interp::uni::Interval;
using ode::interp::uni::Interpolator;
using ode::interp::uni::LocalInterpolator;
using ode::interp::uni::CustomLocalInterpolator;
using ode::interp::uni::LinkedInterpolator;
using ode::interp::uni::BoxedInterp;

// Boundary flags: a closed end belongs to the interval, an open one does not, and a linking one
// is decided when the interval is linked to its neighbour
constexpr int closed = 1, open = -1, link = 0;


TEMPLATE_TEST_CASE("Intervals contain their interior and the ends they close", "[interpolation][interval]", ODECRAFT_TEST_SCALARS){
    using T = TestType;

    SECTION("a closed interval contains both ends"){
        const Interval<T> I(T(0), T(1), closed, closed);
        CHECK(I.contains(T(0)));
        CHECK(I.contains(T(0.5)));
        CHECK(I.contains(T(1)));
        CHECK_FALSE(I.contains(T(-0.1)));
        CHECK_FALSE(I.contains(T(1.1)));
        CHECK(I.signature() == "[0, 1]");
    }
    SECTION("an open interval contains neither"){
        const Interval<T> I(T(0), T(1), open, open);
        CHECK_FALSE(I.contains(T(0)));
        CHECK(I.contains(T(0.5)));
        CHECK_FALSE(I.contains(T(1)));
        CHECK(I.signature() == "(0, 1)");
    }
    SECTION("a linking end counts as contained until it is linked"){
        const Interval<T> I(T(0), T(1), link, link);
        CHECK(I.contains(T(0)));
        CHECK(I.contains(T(1)));
        CHECK(I.signature() == "|0, 1|");
    }
    SECTION("a reversed interval runs backward"){
        const Interval<T> I(T(1), T(0), closed, open);
        CHECK(I.dir() == -1);
        CHECK(I.h() == T(-1));
        CHECK(I.contains(T(1)));
        CHECK(I.contains(T(0.25)));
        CHECK_FALSE(I.contains(T(0)));
        CHECK_FALSE(I.contains(T(1.5)));
    }
    SECTION("a point contains itself only"){
        const Interval<T> I(T(2));
        CHECK(I.is_point());
        CHECK(I.contains(T(2)));
        CHECK_FALSE(I.contains(T(2.5)));
        CHECK(I.signature() == "[2]");
        CHECK((I.start() == T(2) && I.end() == T(2)));
    }
    SECTION("an empty interval is rejected"){
        CHECK_THROWS_AS(Interval<T>(T(1), T(1)), std::runtime_error);
    }
}

TEMPLATE_TEST_CASE("Intervals resize and link at their ends", "[interpolation][interval]", ODECRAFT_TEST_SCALARS){
    using T = TestType;

    SECTION("an end moves only within the interval's direction"){
        Interval<T> I(T(0), T(2), closed, closed);
        I.adjust_start(T(0.5));
        I.adjust_end(T(1.5));
        CHECK((I.start() == T(0.5) && I.end() == T(1.5) && I.h() == T(1)));
        CHECK_THROWS_AS(I.adjust_start(T(1.5)), std::runtime_error);
        CHECK_THROWS_AS(I.adjust_end(T(0.5)), std::runtime_error);
    }
    SECTION("ends open and close"){
        Interval<T> I(T(0), T(1), open, closed);
        I.open_end();
        CHECK_FALSE(I.contains(T(1)));
        I.close_end();
        I.close_start();
        CHECK((I.contains(T(0)) && I.contains(T(1))));
        Interval<T> point(T(3));
        CHECK_THROWS_AS(point.open_end(), std::runtime_error);
    }
    SECTION("a linking end gives the shared time to exactly one of the two intervals"){
        Interval<T> left(T(0), T(1), closed, link);
        Interval<T> right(T(1), T(2), closed, closed);
        REQUIRE(left.can_link_with(right));
        left.link_with(right);
        CHECK_FALSE(left.contains(T(1)));
        CHECK(right.contains(T(1)));
    }
    SECTION("intervals that would share or lose the time between them do not link"){
        const Interval<T> a(T(0), T(1), closed, closed);
        CHECK_FALSE(a.can_link_with(Interval<T>(T(1), T(2), closed, closed)));   // both claim t = 1
        const Interval<T> b(T(0), T(1), closed, open);
        CHECK_FALSE(b.can_link_with(Interval<T>(T(1), T(2), open, closed)));     // neither does
        const Interval<T> c(T(0), T(1), closed, link);
        CHECK_FALSE(c.can_link_with(Interval<T>(T(1), T(2), link, closed)));     // no one decides
        CHECK_FALSE(c.can_link_with(Interval<T>(T(1.5), T(2), closed, closed))); // a gap
        CHECK_FALSE(c.can_link_with(Interval<T>(T(1), T(0), closed, closed)));   // turns back
        Interval<T> d(T(0), T(1), closed, closed);
        Interval<T> e(T(1), T(2), closed, closed);
        CHECK_THROWS_AS(d.link_with(e), std::runtime_error);
    }
}


TEMPLATE_TEST_CASE("A local interpolator is exact at its ends and linear between them", "[interpolation][local]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    const double tol = 10 * epsilon<T>();

    for_each_type<StateKinds<2>>([&]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            using Local = LocalInterpolator<T, SD>;
            const auto y1 = make_state<SD, T>({0, 10});
            const auto y2 = make_state<SD, T>({2, 30});
            auto out = y1;

            SECTION("a segment"){
                const Local f(T(1), T(3), y1, y2, closed, closed);
                f.call(out, T(2));
                CHECK_THAT(max_error<SD>(out, make_state<SD, T>({1, 20})), IsAtMost(tol));
                f.call(out, T(1));
                CHECK(same_state<SD>(out, y1));
                f.call(out, T(3));
                CHECK(same_state<SD>(out, y2));
                CHECK_THROWS_AS(f.call(out, T(3.5)), std::runtime_error);

                CHECK(f.order() == 1);
                CHECK(f.dir() == 1);
                CHECK((f.t_start() == T(1) && f.t_end() == T(3)));
                CHECK(same_state<SD>(f.q_start(), y1));
                CHECK(same_state<SD>(f.q_end(), y2));
                CHECK(f.is_out_of_bounds(T(0)));
                CHECK_FALSE(f.is_out_of_bounds(T(1)));
            }
            SECTION("a segment with an open end"){
                const Local f(T(1), T(3), y1, y2, closed, open);
                CHECK_THROWS_AS(f.call(out, T(3)), std::runtime_error);
                CHECK(f.is_out_of_bounds(T(3)));
                CHECK(f.end_bdr() == open);
            }
            SECTION("a backward segment"){
                const Local f(T(3), T(1), y2, y1, closed, closed);
                CHECK(f.dir() == -1);
                f.call(out, T(2.5));
                CHECK_THAT(max_error<SD>(out, make_state<SD, T>({1.5, 25})), IsAtMost(tol));
            }
            SECTION("a point"){
                const Local f(T(5), y2);
                f.call(out, T(5));
                CHECK(same_state<SD>(out, y2));
                CHECK_THROWS_AS(f.call(out, T(5.1)), std::runtime_error);
                CHECK(f.interval().is_point());
            }
            SECTION("narrowed within its step, it still evaluates the same polynomial"){
                Local f(T(1), T(3), y1, y2, closed, closed);
                f.adjust_start(T(1.5));
                f.adjust_end(T(2.5));
                CHECK_THROWS_AS(f.call(out, T(1)), std::runtime_error);
                f.call(out, T(2));
                CHECK_THAT(max_error<SD>(out, make_state<SD, T>({1, 20})), IsAtMost(tol));
                // It can grow back up to its step, never beyond
                f.adjust_start(T(1));
                CHECK_THROWS_AS(f.adjust_end(T(3.5)), std::runtime_error);
                CHECK_THROWS_AS(f.adjust_start(T(0.5)), std::runtime_error);
            }
            SECTION("a clone is independent of the original"){
                Local f(T(1), T(3), y1, y2, closed, closed);
                std::unique_ptr<Interpolator<T, SD>> copy = f.clone();
                copy->adjust_end(T(2));
                CHECK(copy->t_end() == T(2));
                CHECK(f.t_end() == T(3));
            }
        }
    });
}

TEMPLATE_TEST_CASE("A custom local interpolator evaluates its callable inside the step", "[interpolation][local]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<1>;
    // y = t^2 on [0, 2], which the linear default would get wrong in between
    auto square = [](state_t<T, SD>& out, const T& t){ out[0] = t*t; };
    using Custom = CustomLocalInterpolator<T, SD, decltype(square)>;
    const Custom f(square, T(0), T(2), make_state<SD, T>({0}), make_state<SD, T>({4}), closed, closed);

    auto out = make_state<SD, T>({0});
    f.call(out, T(0.5));
    CHECK(out[0] == T(0.25));
    f.call(out, T(2));
    CHECK(out[0] == T(4));

    auto copy = f.clone();
    copy->call(out, T(1.5));
    CHECK(out[0] == T(2.25));
}


/// @brief A chain of n segments of y = (t, 2t) on [0, n], built the way a solver builds its dense
/// output: a point, then each step joined to the previous one.
template<typename SD, typename T>
LinkedInterpolator<T, SD> piecewise_linear_chain(size_t n){
    const auto at = [](const T& t){ return make_state<SD, T>({t, T(2*t)}); };
    LinkedInterpolator<T, SD> chain(T(0), at(T(0)));
    for (size_t i = 0; i < n; i++){
        const T a = T(i), b = T(i + 1);
        // The first segment replaces the starting point, the others link to the one before
        chain.expand(pbox::make_box<LocalInterpolator<T, SD>>(a, b, at(a), at(b), i == 0 ? closed : link, closed));
    }
    return chain;
}

TEMPLATE_TEST_CASE("A linked interpolator chains steps into one dense output", "[interpolation][linked]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    const double tol = 100 * epsilon<T>();

    for_each_type<StateKinds<2>>([&]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            constexpr size_t n = 40;
            LinkedInterpolator<T, SD> chain = piecewise_linear_chain<SD, T>(n);
            auto out = make_state<SD, T>({0, 0});

            SECTION("each time is routed to the segment that holds it"){
                REQUIRE(chain.size() == n);
                for (double t : {0.0, 0.25, 1.0, 7.5, 13.0, 21.125, 39.99, 40.0}){
                    CAPTURE(t);
                    chain.call(out, T(t));
                    CHECK_THAT(max_error<SD>(out, make_state<SD, T>({T(t), T(2*t)})), IsAtMost(tol));
                }
                CHECK_THROWS_AS(chain.call(out, T(40.5)), std::runtime_error);
                CHECK_THROWS_AS(chain.call(out, T(-0.5)), std::runtime_error);
            }
            SECTION("it reports the span and the states at its ends"){
                CHECK((chain.t_start() == T(0) && chain.t_end() == T(n)));
                CHECK(chain.dir() == 1);
                CHECK(chain.interval().signature() == "[0, 40]");
                CHECK(same_state<SD>(chain.q_end(), make_state<SD, T>({T(n), T(2*n)})));
                CHECK(same_state<SD>(chain.q_start(), make_state<SD, T>({0, 0})));
                CHECK(chain.interpolant(3).t_start() == T(3));
                CHECK(chain.last_interpolant().t_end() == T(n));
                CHECK_THROWS_AS(chain.interpolant(n), std::out_of_range);
                CHECK(chain.is_out_of_bounds(T(41)));
                CHECK_FALSE(chain.is_out_of_bounds(T(40)));
            }
            SECTION("a linking start yields the shared time to the end that closes it"){
                CHECK(chain.interpolant(0).end_bdr() == closed);
                CHECK(chain.interpolant(1).start_bdr() == open);
            }
            SECTION("a segment that does not start where the chain ends is refused"){
                const auto gap = pbox::make_box<LocalInterpolator<T, SD>>(T(n + 1), T(n + 2), out, out, link, closed);
                CHECK_THROWS_AS(chain.expand(pbox::make_box<LocalInterpolator<T, SD>>(T(n + 1), T(n + 2), out, out, link, closed)), std::runtime_error);
                CHECK_FALSE(chain.can_link_with(*gap));
            }
            SECTION("its ends move, open and close"){
                chain.adjust_start(T(0.5));
                chain.adjust_end(T(39.5));
                CHECK((chain.t_start() == T(0.5) && chain.t_end() == T(39.5)));
                chain.open_end();
                CHECK(chain.is_out_of_bounds(T(39.5)));
                chain.close_end();
                CHECK_FALSE(chain.is_out_of_bounds(T(39.5)));
            }
            SECTION("closing its start makes the first time part of it"){
                auto open_start = LinkedInterpolator<T, SD>(pbox::make_box<LocalInterpolator<T, SD>>(T(0), T(1), out, out, open, closed));
                CHECK(open_start.is_out_of_bounds(T(0)));
                open_start.close_start();
                CHECK_FALSE(open_start.is_out_of_bounds(T(0)));
                CHECK(open_start.start_bdr() == closed);
            }
            SECTION("a clone is a deep copy"){
                std::unique_ptr<Interpolator<T, SD>> copy = chain.clone();
                copy->adjust_end(T(39.5));
                CHECK(copy->t_end() == T(39.5));
                CHECK(chain.t_end() == T(n));
                copy->call(out, T(10.5));
                CHECK_THAT(max_error<SD>(out, make_state<SD, T>({10.5, 21})), IsAtMost(tol));
            }
        }
    });
}

TEMPLATE_TEST_CASE("A linked interpolator chains backward steps as well", "[interpolation][linked]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<1>;
    const auto at = [](const T& t){ return make_state<SD, T>({T(-t)}); };

    LinkedInterpolator<T, SD> chain(T(0), at(T(0)));
    for (int i = 0; i < 5; i++){
        const T a = T(-i), b = T(-i - 1);
        chain.expand(pbox::make_box<LocalInterpolator<T, SD>>(a, b, at(a), at(b), i == 0 ? closed : link, closed));
    }
    CHECK(chain.dir() == -1);
    CHECK((chain.t_start() == T(0) && chain.t_end() == T(-5)));
    auto out = at(T(0));
    chain.call(out, T(-3.25));
    CHECK_THAT(abs_error(out[0], T(3.25)), IsAtMost(10 * epsilon<T>()));
    CHECK_THROWS_AS(chain.call(out, T(0.5)), std::runtime_error);
}


TEMPLATE_TEST_CASE("The interpolation formulas of dense output", "[interpolation]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const double tol = 10 * epsilon<T>();
    const auto y1 = make_state<SD, T>({1, -1});
    const auto y2 = make_state<SD, T>({3, -5});
    auto out = y1;

    SECTION("lin_interp draws the line between the two ends"){
        ode::interp::uni::lin_interp<T, SD>(out, T(1.5), T(1), T(2), y1, y2);
        CHECK_THAT(max_error<SD>(out, make_state<SD, T>({2, -3})), IsAtMost(tol));
    }
    SECTION("coef_mat_interp sums the polynomial y1 + h * sum_j c_j x^(j+1)"){
        // On [0, 2], c_0 = (1, 0) and c_1 = (0, 1): y(t) = y1 + 2*(x, x^2) with x = t/2
        const std::array<state_t<T, SD>, 2> coefs = {make_state<SD, T>({1, 0}), make_state<SD, T>({0, 1})};
        ode::interp::uni::coef_mat_interp<T, SD>(out, T(1), T(0), T(2), y1, y2, coefs.data(), 2);
        CHECK_THAT(max_error<SD>(out, make_state<SD, T>({2, -0.5})), IsAtMost(tol));
    }
}

} // namespace
} // namespace odetest
