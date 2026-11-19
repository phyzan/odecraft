// Vector fields: interpolators whose values are vectors, so they define a flow, and the
// streamlines they integrate along it, forward and backward, with any stepper. The flow of the
// tests is a rigid rotation, whose streamlines are circles: linear, so a grid or a triangulation
// interpolates it exactly, and its streamlines are known in closed form.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep
#include "../support/grids.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace odetest {
namespace {

namespace rgi = ode::interp::rgi;
namespace sci = ode::interp::sci;
using ode::interp::VirtualNdInterpolator;
using ode::interp::VirtualVectorField;
using rgi::CoordType;
using std::numbers::pi;

// A field's dimension is fixed at compile time or decided at runtime (0), and it is reached
// directly or through VirtualVectorField (true)
using RegularField2 = rgi::RegularVectorField<2>;
using RegularFieldN = rgi::RegularVectorField<0, true>;
using ScatteredField2 = sci::ScatteredVectorField<2, false>;
using ScatteredFieldN = sci::ScatteredVectorField<0, true>;

template<typename F>
struct FieldTraits;

template<int N, bool V>
struct FieldTraits<rgi::RegularVectorField<N, V>> {
    static constexpr bool regular = true;
    static constexpr size_t ndim = N;
};

template<int N, bool V>
struct FieldTraits<sci::ScatteredVectorField<N, V>> {
    static constexpr bool regular = false;
    static constexpr size_t ndim = N;
};

/// @brief What a streamline of the field is stored in.
template<typename F>
using Streamline = ode::OdeResult<double, ode::interp::FieldSD<FieldTraits<F>::ndim>>;

const ode::Steppers::Id rk45 = ode::Steppers::id<tags::RK45>;

/// @brief The rotation sampled on the custom grid, read as a grid or as scattered points.
template<typename F>
F rotation_field(){
    const Rectilinear grid = custom_grid();
    const auto points = grid.points();
    const auto values = sample(fields::rotation, points, 2, 2);
    if constexpr (FieldTraits<F>::regular){
        return F(view_of(values, grid.size(), 2), grid.axes, CoordType::Cartesian, true);
    } else {
        return F(points.data(), view_of(values, grid.size(), 2), 2, true);
    }
}

/// @brief The largest distance from the points of a streamline to where the rotation takes it
/// from (r, 0): the circle of radius r, run at angular speed w.
template<typename Result>
Magnitude distance_to_circle(const Result& line, double r, double w){
    if (line.t().empty() || line.t().size() != line.q().size()){
        return {std::numeric_limits<double>::quiet_NaN()};
    }
    Magnitude err{0};
    for (size_t i = 0; i < line.t().size(); i++){
        const auto& q = line.q()[i];
        const double angle = w * line.t()[i];
        err = worst(abs_error(q[0], r * std::cos(angle)), err);
        err = worst(abs_error(q[1], r * std::sin(angle)), err);
    }
    return err;
}

// From (r, 0) the rotation turns at angular speed 1, and at 1/r once normalized to unit speed
constexpr double r0 = 1.2;
constexpr double start[] = {r0, 0.0};

double angular_speed(bool normalized){
    return normalized ? 1 / r0 : 1.0;
}


TEMPLATE_TEST_CASE("A streamline ODE runs along the field or against it with any stepper", "[vector-field][streamline]", RegularField2, RegularFieldN, ScatteredField2, ScatteredFieldN){
    const auto field = rotation_field<TestType>();
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const Settings<double> s = settings<Stepper, double>();
            for (int dir : {1, -1}){
                for (bool normalized : {false, true}){
                    CAPTURE(dir, normalized);
                    auto ode = field.get_streamline_ode(start, s.rtol, s.atol, 0, 0, s.stepsize, dir, ode::Steppers::id<Stepper>, normalized);
                    REQUIRE(ode->solver()->get_method() == Stepper::name);
                    Streamline<TestType> line;
                    REQUIRE(ode->integrate_until(&line, dir * pi / 2));
                    CHECK(line.t().back() == dir * pi / 2);
                    CHECK_THAT(distance_to_circle(line, r0, angular_speed(normalized)), IsAtMost(s.error));
                }
            }
        }
    });
}

TEMPLATE_TEST_CASE("A streamline runs along the field or against it", "[vector-field][streamline]", RegularField2, RegularFieldN, ScatteredField2, ScatteredFieldN){
    const auto field = rotation_field<TestType>();
    const double length = 2.0;
    for (int dir : {1, -1}){
        for (bool normalized : {false, true}){
            CAPTURE(dir, normalized);
            const auto line = field.streamline(start, length, 1e-10, 1e-10, 0, 0, 0, dir, rk45, normalized);
            REQUIRE_FALSE(line->t().empty());
            CHECK(line->t().back() == dir * length);
            CHECK_THAT(distance_to_circle(*line, r0, angular_speed(normalized)), IsAtMost(1e-8));
        }
    }
}

TEMPLATE_TEST_CASE("A streamline is stored at the requested times only", "[vector-field][streamline][t-eval]", RegularField2, RegularFieldN, ScatteredField2, ScatteredFieldN){
    const auto field = rotation_field<TestType>();
    const std::vector<double> t_eval = {0.3, 0.9, 1.4};
    const auto line = field.streamline(start, 1.5, 1e-10, 1e-10, 0, 0, 0, 1, rk45, false, t_eval);
    CHECK(line->t() == t_eval);
    CHECK_THAT(distance_to_circle(*line, r0, 1.0), IsAtMost(1e-8));
}

TEMPLATE_TEST_CASE("A vector field vanishes outside its domain", "[vector-field]", RegularField2, RegularFieldN, ScatteredField2, ScatteredFieldN){
    const auto field = rotation_field<TestType>();
    const double inside[] = {0.5, -0.3}, outside[] = {2.5, 0.0};
    double out[2] = {7, 7};

    CHECK(field.ndim() == 2);
    CHECK(field.contains(inside));
    CHECK_FALSE(field.contains(outside));
    REQUIRE(field.interp(out, inside));
    CHECK_THAT(abs_error(out[0], 0.3), IsAtMost(1e-14));
    CHECK_THAT(abs_error(out[1], 0.5), IsAtMost(1e-14));

    SECTION("as the right-hand side of its streamlines"){
        field.OdeFunc(out, 0, inside);
        CHECK_THAT(abs_error(out[0], 0.3), IsAtMost(1e-14));
        CHECK_THAT(abs_error(out[1], 0.5), IsAtMost(1e-14));
        field.OdeFunc(out, 0, outside);
        CHECK(out[0] == 0);
        CHECK(out[1] == 0);
    }
    SECTION("normalized to unit speed"){
        field.OdeFuncNorm(out, 0, inside);
        CHECK_THAT(abs_error(out[0], 0.3 / std::hypot(0.3, 0.5)), IsAtMost(1e-14));
        CHECK_THAT(abs_error(out[1], 0.5 / std::hypot(0.3, 0.5)), IsAtMost(1e-14));
        field.OdeFuncNorm(out, 0, outside);
        CHECK(out[0] == 0);
        CHECK(out[1] == 0);
    }
    SECTION("so a streamline that leaves it stops at its edge"){
        // The circle of radius 1.9 leaves the domain through its upper side, y = 1.5
        const double far[] = {1.9, 0.0};
        const auto line = field.streamline(far, 3.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, false);
        REQUIRE_FALSE(line->q().empty());
        const auto& end = line->q().back();
        CHECK_THAT(abs_error(end[0], std::sqrt(1.9 * 1.9 - 1.5 * 1.5)), IsAtMost(1e-6));
        CHECK_THAT(abs_error(end[1], 1.5), IsAtMost(1e-6));
    }
}

TEMPLATE_TEST_CASE("A vector field is reached through VirtualVectorField", "[vector-field][virtual]", RegularFieldN, ScatteredFieldN){
    const auto field = rotation_field<TestType>();
    const VirtualVectorField<0>& base = field;
    const double inside[] = {0.5, -0.3}, outside[] = {2.5, 0.0};
    CHECK(base.ndim() == 2);
    CHECK(base.contains(inside));
    CHECK_FALSE(base.contains(outside));

    const auto line = base.streamline(start, 1.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, false);
    CHECK(line->t().back() == 1.0);
    CHECK_THAT(distance_to_circle(*line, r0, 1.0), IsAtMost(1e-8));

    const std::vector<double> t_eval = {0.5, 1.0};
    const auto sampled = base.streamline(start, 1.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, true, t_eval);
    CHECK(sampled->t() == t_eval);
    CHECK_THAT(distance_to_circle(*sampled, r0, 1 / r0), IsAtMost(1e-8));

    auto ode = base.get_streamline_ode(start, 1e-10, 1e-10, 0, 0, 0, -1, rk45, false);
    Streamline<TestType> backward;
    REQUIRE(ode->integrate_until(&backward, -1.0));
    CHECK_THAT(distance_to_circle(backward, r0, 1.0), IsAtMost(1e-8));

    // And as the interpolator it is
    const VirtualNdInterpolator& interp = field;
    double out[2];
    REQUIRE(interp.interp(out, inside));
    CHECK_THAT(abs_error(out[0], 0.3), IsAtMost(1e-14));
    CHECK(interp.nvals_per_point() == 2);
}


// ============================================================================================
//                                       streamplot_data
// ============================================================================================

/// @brief What the streamlines of a streamplot add up to.
struct StreamplotSummary {
    size_t lines = 0;
    size_t malformed = 0;      // not of shape (2, n >= 2)
    size_t repeats = 0;        // points that coincide with the one before them
    size_t closed_circles = 0; // lines that never come near the edge of the domain
    Magnitude worst_gap{0};    // the longest step between consecutive points, over ds
    Magnitude worst_drift{0};  // the largest change of radius along a closed circle
    Magnitude worst_escape{0}; // how far beyond the domain any point lies
};

/// @brief Summarizes the streamlines of a rotation over the custom grid, in Cartesian (x, y)
/// coordinates or in polar (r, phi) ones, whose physical distance is r dphi.
template<typename Lines>
StreamplotSummary summarize(const Lines& lines, double ds, bool polar, double x_lo, double x_hi, double y_lo, double y_hi){
    StreamplotSummary out;
    out.lines = lines.size();
    for (const auto& line : lines){
        if (line.shape(0) != 2 || line.shape(1) < 2){
            ++out.malformed;
            continue;
        }
        const size_t n = line.shape(1);
        const auto radius = [&](size_t i){ return polar ? line(0, i) : std::hypot(line(0, i), line(1, i)); };
        bool closed = true;
        double r_min = radius(0), r_max = radius(0);
        for (size_t i = 0; i < n; i++){
            const double x = line(0, i), y = line(1, i);
            const double escape = std::max({x_lo - x, x - x_hi, y_lo - y, y - y_hi, 0.0});
            out.worst_escape = worst(Magnitude{escape}, out.worst_escape);
            if (i > 0){
                const double dx = x - line(0, i - 1), dy = y - line(1, i - 1);
                out.repeats += (dx == 0 && dy == 0) ? 1 : 0;
                const double gap = polar ? std::hypot(dx, x * dy) : std::hypot(dx, dy);
                out.worst_gap = worst(Magnitude{gap / ds}, out.worst_gap);
            }
            r_min = std::min(r_min, radius(i));
            r_max = std::max(r_max, radius(i));
            // In Cartesian coordinates only circles of radius below 1.5 stay inside the domain
            closed = closed && (polar || radius(i) < 1.45);
        }
        if (closed){
            ++out.closed_circles;
            out.worst_drift = worst(Magnitude{r_max - r_min}, out.worst_drift);
        }
    }
    return out;
}

TEMPLATE_TEST_CASE("streamplot_data covers a field with streamlines along it", "[vector-field][streamplot]", RegularField2, RegularFieldN){
    const auto field = rotation_field<TestType>();
    const double ds = 0.05;
    const auto lines = field.streamplot_data(3.0, ds, 10, 1e-9, 1e-9, 0, 0, 0, rk45);
    const StreamplotSummary summary = summarize(lines, ds, false, -2, 2, -1.5, 1.5);
    CAPTURE(summary.lines, summary.closed_circles);
    CHECK(summary.lines >= 3);
    CHECK(summary.malformed == 0);
    // A line that leaves the domain ends there, rather than piling up points on its edge
    CHECK(summary.repeats == 0);
    CHECK(summary.closed_circles >= 1);
    CHECK_THAT(summary.worst_drift, IsAtMost(1e-6));
    // Points are ds apart along the line, or closer where it stops at the edge
    CHECK_THAT(summary.worst_gap, IsAtMost(1 + 1e-6));
    CHECK_THAT(summary.worst_escape, IsAtMost(ds));

    SECTION("and refuses settings it cannot draw with"){
        CHECK_THROWS_AS(field.streamplot_data(3.0, ds, 1, 1e-9, 1e-9, 0, 0, 0, rk45), std::domain_error);
        CHECK_THROWS_AS(field.streamplot_data(ds, ds, 10, 1e-9, 1e-9, 0, 0, 0, rk45), std::domain_error);
        CHECK_THROWS_AS(field.streamplot_data(3.0, 0.0, 10, 1e-9, 1e-9, 0, 0, 0, rk45), std::domain_error);
        CHECK_THROWS_AS(field.streamplot_data(-1.0, -2.0, 10, 1e-9, 1e-9, 0, 0, 0, rk45), std::domain_error);
    }
}

/// @brief A uniform flow along x.
void flow_along_x(double* out, const double* /*x*/){
    out[0] = 1;
    out[1] = 0;
}

TEMPLATE_TEST_CASE("A streamplot line runs max_length each way from its seed and keeps both ends", "[vector-field][streamplot]", RegularField2, RegularFieldN){
    // A uniform flow along x over [-4, 4]^2, seeded on a 3 x 3 grid of cells 4 wide. A line of
    // max_length 1 each way never leaves the cell it starts in, so nothing stops it short: the one
    // from the centre takes 5 steps of 0.25 each way, 11 points from x = -1.25 to x = 1.25
    const Rectilinear grid{{{-4.0, 0.0, 4.0}, {-4.0, 0.0, 4.0}}};
    const auto values = sample(flow_along_x, grid.points(), 2, 2);
    const TestType field(view_of(values, grid.size(), 2), grid.axes, CoordType::Cartesian, true);
    const auto lines = field.streamplot_data(1.0, 0.25, 3, 1e-10, 1e-10, 0, 0, 0, rk45);

    const auto through_centre = [](const auto& line){
        for (size_t i = 0; i < line.shape(1); i++){
            if (std::abs(line(0, i)) < 1e-12 && std::abs(line(1, i)) < 1e-12){
                return true;
            }
        }
        return false;
    };
    const auto centre = std::find_if(lines.begin(), lines.end(), through_centre);
    REQUIRE(centre != lines.end());
    REQUIRE(centre->shape(1) == 11);
    for (size_t i = 0; i < 11; i++){
        CAPTURE(i);
        CHECK_THAT(abs_error((*centre)(0, i), -1.25 + 0.25 * double(i)), IsAtMost(1e-12));
        CHECK_THAT(abs_error((*centre)(1, i), 0.0), IsAtMost(1e-12));
    }
}


// ============================================================================================
//                                Polar and spherical coordinates
// ============================================================================================

/// @brief A field over (r, phi) in [0.5, 2] x [0, 2 pi], or (r, theta, phi) in [0.5, 2] x
/// [0.3, 2.8] x [0, 2 pi], from its physical components at each point.
template<typename F>
F curvilinear_field(CoordType coords, const Field& physical){
    const size_t nd = coords == CoordType::Polar ? 2 : 3;
    Rectilinear grid{{linspace(0.5, 2, 7)}};
    if (nd == 3){
        grid.axes.push_back(linspace(0.3, 2.8, 11));
    }
    grid.axes.push_back(linspace(0, 2 * pi, 13));
    const auto values = sample(physical, grid.points(), nd, nd);
    return F(view_of(values, grid.size(), nd), grid.axes, coords, true);
}

/// @brief Where a streamline ended, as a point.
template<typename Line>
std::vector<double> end_of(const Line& line, size_t nd){
    if (line->q().empty()){
        return std::vector<double>(nd, std::numeric_limits<double>::quiet_NaN());
    }
    const auto& q = line->q().back();
    std::vector<double> out(nd);
    for (size_t i = 0; i < nd; i++){
        out[i] = q[i];
    }
    return out;
}

TEST_CASE("Fields in polar and spherical coordinates hold physical components", "[vector-field][coordinates]"){
    SECTION("in polar coordinates"){
        // v = r e_r + r e_phi: at unit speed dr/ds = 1/sqrt 2, and r dphi/ds = 1/sqrt 2
        const auto field = curvilinear_field<RegularField2>(CoordType::Polar, [](double* out, const double* x){
            out[0] = x[0];
            out[1] = x[0];
        });
        const double from[] = {0.8, 0.1};
        const auto line = field.streamline(from, 1.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, true);
        const double r = 0.8 + 1 / std::sqrt(2.0);
        const auto end = end_of(line, 2);
        CHECK_THAT(abs_error(end[0], r), IsAtMost(1e-8));
        CHECK_THAT(abs_error(end[1], 0.1 + std::log(r / 0.8)), IsAtMost(1e-8));
    }
    SECTION("in spherical coordinates"){
        const double from[] = {1.5, 1.0, 0.3};
        SECTION("along a parallel"){
            // v = r sin(theta) e_phi, a rotation about the polar axis: dphi/ds = 1/(r sin theta)
            const auto field = curvilinear_field<RegularFieldN>(CoordType::Spherical, [](double* out, const double* x){
                out[0] = 0;
                out[1] = 0;
                out[2] = x[0] * std::sin(x[1]);
            });
            const auto end = end_of(field.streamline(from, 1.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, true), 3);
            CHECK_THAT(abs_error(end[0], 1.5), IsAtMost(1e-8));
            CHECK_THAT(abs_error(end[1], 1.0), IsAtMost(1e-8));
            CHECK_THAT(abs_error(end[2], 0.3 + 1 / (1.5 * std::sin(1.0))), IsAtMost(1e-8));
        }
        SECTION("along a meridian"){
            // v = r e_theta: dtheta/ds = 1/r
            const auto field = curvilinear_field<RegularFieldN>(CoordType::Spherical, [](double* out, const double* x){
                out[0] = 0;
                out[1] = x[0];
                out[2] = 0;
            });
            const auto end = end_of(field.streamline(from, 0.9, 1e-10, 1e-10, 0, 0, 0, 1, rk45, true), 3);
            CHECK_THAT(abs_error(end[0], 1.5), IsAtMost(1e-8));
            CHECK_THAT(abs_error(end[1], 1.0 + 0.9 / 1.5), IsAtMost(1e-8));
            CHECK_THAT(abs_error(end[2], 0.3), IsAtMost(1e-8));
        }
    }
}

TEST_CASE("A streamline at the field's own speed reads its components as physical too", "[vector-field][coordinates]"){
    // Without normalizing, the field's physical components are still turned into the rates of
    // its coordinates, at the field's own speed. v = r e_phi, linear in r and so interpolated
    // exactly, turns at dphi/dt = 1 in the plane, and at dphi/dt = 1/sin(theta) in space, about the
    // polar axis on the cone of its own theta
    SECTION("in polar coordinates"){
        const auto field = curvilinear_field<RegularField2>(CoordType::Polar, [](double* out, const double* x){
            out[0] = 0;
            out[1] = x[0];
        });
        const double from[] = {1.5, 0.2};
        const auto end = end_of(field.streamline(from, 1.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, false), 2);
        CHECK_THAT(abs_error(end[0], 1.5), IsAtMost(1e-8));
        CHECK_THAT(abs_error(end[1], 1.2), IsAtMost(1e-8));
    }
    SECTION("in spherical coordinates"){
        const auto field = curvilinear_field<RegularFieldN>(CoordType::Spherical, [](double* out, const double* x){
            out[0] = 0;
            out[1] = 0;
            out[2] = x[0];
        });
        const double from[] = {1.5, 1.0, 0.3};
        const auto end = end_of(field.streamline(from, 1.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, false), 3);
        CHECK_THAT(abs_error(end[0], 1.5), IsAtMost(1e-8));
        CHECK_THAT(abs_error(end[1], 1.0), IsAtMost(1e-8));
        CHECK_THAT(abs_error(end[2], 0.3 + 1 / std::sin(1.0)), IsAtMost(1e-8));
    }
}

TEST_CASE("streamplot_data in polar coordinates follows the field", "[vector-field][coordinates][streamplot]"){
    const auto field = curvilinear_field<RegularField2>(CoordType::Polar, [](double* out, const double* x){
        out[0] = 0;
        out[1] = x[0];
    });
    const double ds = 0.05;
    const auto lines = field.streamplot_data(3.0, ds, 8, 1e-9, 1e-9, 0, 0, 0, rk45);
    const StreamplotSummary summary = summarize(lines, ds, true, 0.5, 2, 0, 2 * pi);
    CAPTURE(summary.lines);
    CHECK(summary.lines >= 3);
    CHECK(summary.malformed == 0);
    // A line that leaves the domain ends there, rather than piling up points on its edge
    CHECK(summary.repeats == 0);
    // Along circles, ds apart in physical length
    CHECK_THAT(summary.worst_drift, IsAtMost(1e-6));
    CHECK_THAT(summary.worst_gap, IsAtMost(1 + 1e-6));
}

TEST_CASE("A vector field needs a dimension its coordinates have", "[vector-field][coordinates]"){
    const auto build = [](size_t nd, CoordType coords){
        const Rectilinear grid{std::vector<std::vector<double>>(nd, {0.5, 1.0, 1.5})};
        const auto values = sample([nd](double* out, const double*){ std::fill(out, out + nd, 1.0); }, grid.points(), nd, nd);
        return RegularFieldN(view_of(values, grid.size(), nd), grid.axes, coords, true);
    };
    CHECK_THROWS_AS(build(1, CoordType::Cartesian), std::invalid_argument);
    CHECK_THROWS_AS(build(2, CoordType::Spherical), std::invalid_argument);
    CHECK_THROWS_AS(build(4, CoordType::Polar), std::invalid_argument);
    CHECK_THROWS_AS(build(4, CoordType::Spherical), std::invalid_argument);
    CHECK(build(4, CoordType::Cartesian).ndim() == 4);
    CHECK(build(3, CoordType::Spherical).ndim() == 3);
    CHECK(build(2, CoordType::Polar).ndim() == 2);
}


// ============================================================================================
//                         One set of samples, as a grid and as points
// ============================================================================================

TEST_CASE("The same samples give the same streamlines read as a grid or as scattered points", "[vector-field][regular][scattered]"){
    const auto regular = rotation_field<RegularField2>();
    const auto scattered = rotation_field<ScatteredField2>();
    // Also on a triangulation shared with another interpolator
    const Rectilinear grid = custom_grid();
    const auto values = sample(fields::rotation, grid.points(), 2, 2);
    const ScatteredField2 shared(scattered.tri(), view_of(values, grid.size(), 2), true);
    REQUIRE(shared.tri().shared_tri() == scattered.tri().shared_tri());

    for (int dir : {1, -1}){
        for (bool normalized : {false, true}){
            CAPTURE(dir, normalized);
            const auto integrate = [&](const auto& field){
                auto ode = field.get_streamline_ode(start, 1e-10, 1e-10, 0, 0, 0, dir, rk45, normalized);
                ode::OdeResult<double, ode::interp::FieldSD<2>> line;
                ode->integrate_until(&line, dir * 2.5);
                return line;
            };
            const auto on_grid = integrate(regular);
            const auto on_points = integrate(scattered);
            const auto on_shared = integrate(shared);
            for (const auto* line : {&on_grid, &on_points, &on_shared}){
                CHECK(line->t().back() == dir * 2.5);
                CHECK_THAT(distance_to_circle(*line, r0, angular_speed(normalized)), IsAtMost(1e-8));
            }
            const auto& a = on_grid.q().back();
            const auto& b = on_points.q().back();
            CHECK_THAT(abs_error(a[0], b[0]), IsAtMost(1e-9));
            CHECK_THAT(abs_error(a[1], b[1]), IsAtMost(1e-9));
        }
    }
}

} // namespace
} // namespace odetest
