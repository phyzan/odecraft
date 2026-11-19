// Interpolation and vector fields of the compiled interface, which are over doubles only and
// take their samples in the one container their constructors are compiled for, field_values_t,
// over the axes of grid_axes_t. The same samples are read as a grid and as scattered points.


#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

#include "../support/crafted_test.hpp" // IWYU pragma: keep
#include "../support/grids.hpp"

namespace odetest {
namespace {

using namespace ode::crafted;
using std::numbers::pi;

/// @brief The axes of a grid, as the compiled constructors take them.
grid_axes_t axes_of(const Rectilinear& grid){
    grid_axes_t axes;
    for (const auto& axis : grid.axes){
        axes.emplace_back(axis.data(), axis.size());
    }
    return axes;
}

/// @brief The largest difference between an interpolator and the field it samples, over points.
Magnitude interpolation_error(const VirtualNdInterpolator& interp, const Field& f, const std::vector<double>& points, size_t nfields){
    const size_t nd = size_t(interp.ndim());
    std::vector<double> got(nfields), expected(nfields);
    Magnitude err{0};
    for (size_t p = 0; p < points.size() / nd; p++){
        if (!interp.interp(got.data(), &points[p * nd])){
            return {std::numeric_limits<double>::quiet_NaN()};
        }
        f(expected.data(), &points[p * nd]);
        for (size_t k = 0; k < nfields; k++){
            err = worst(abs_error(got[k], expected[k]), err);
        }
    }
    return err;
}

/// @brief The largest distance from a streamline of the rotation from (r, 0) to its circle,
/// run at angular speed w.
Magnitude distance_to_circle(const OdeResult<double, FieldSD>& line, double r, double w){
    if (line.t().empty()){
        return {std::numeric_limits<double>::quiet_NaN()};
    }
    Magnitude err{0};
    for (size_t i = 0; i < line.t().size(); i++){
        err = worst(abs_error(line.q()[i][0], r * std::cos(w * line.t()[i])), err);
        err = worst(abs_error(line.q()[i][1], r * std::sin(w * line.t()[i])), err);
    }
    return err;
}

const Steppers::Id rk45 = Steppers::id<tags::RK45>;
constexpr double start[] = {1.2, 0.0};


TEST_CASE("The compiled interpolation types are those of the header-only interface", "[crafted][nd-interpolation]"){
    STATIC_REQUIRE(std::is_same_v<field_values_t, ndspan::View<const double>>);
    STATIC_REQUIRE(std::is_same_v<grid_axes_t, std::vector<ndspan::Array1D<double, 0>>>);
    STATIC_REQUIRE(std::is_same_v<FieldSD, ode::interp::FieldSD<0>>);
    STATIC_REQUIRE(std::is_same_v<FieldState, ode::interp::FieldState<0>>);
    STATIC_REQUIRE(std::is_same_v<RegularGrid, ode::interp::rgi::RegularGrid<double, 0>>);
    STATIC_REQUIRE(std::is_same_v<RegularGridInterpolator, ode::interp::rgi::RegularGridInterpolator<0, true>>);
    STATIC_REQUIRE(std::is_same_v<RegularVectorField, ode::interp::rgi::RegularVectorField<0, true>>);
    STATIC_REQUIRE(std::is_same_v<DelaunayTri, ode::interp::sci::DelaunayTri<0>>);
    STATIC_REQUIRE(std::is_same_v<TriView, ode::interp::sci::TriView<0>>);
    STATIC_REQUIRE(std::is_same_v<ScatteredNdInterpolator, ode::interp::sci::ScatteredNdInterpolator<0, true>>);
    STATIC_REQUIRE(std::is_same_v<ScatteredVectorField, ode::interp::sci::ScatteredVectorField<0, true>>);
    STATIC_REQUIRE(std::is_same_v<VirtualVectorField, ode::interp::VirtualVectorField<0>>);
    STATIC_REQUIRE(std::is_same_v<CoordType, ode::interp::rgi::CoordType>);
    // Every one of them is reached through its interface
    STATIC_REQUIRE(std::is_base_of_v<VirtualNdInterpolator, RegularGridInterpolator>);
    STATIC_REQUIRE(std::is_base_of_v<VirtualNdInterpolator, ScatteredNdInterpolator>);
    STATIC_REQUIRE(std::is_base_of_v<VirtualVectorField, RegularVectorField>);
    STATIC_REQUIRE(std::is_base_of_v<VirtualVectorField, ScatteredVectorField>);
}

TEST_CASE("A compiled grid interpolator reproduces every multilinear function", "[crafted][nd-interpolation][regular]"){
    const Rectilinear grid = custom_grid();
    const grid_axes_t axes = axes_of(grid);
    const RegularGrid regular_grid(axes);
    CHECK(regular_grid.ndim() == 2);
    CHECK(regular_grid.length(0) == 4.0);
    CHECK(regular_grid.find_left_idx(-1.0, 0) == 1);
    CHECK(regular_grid.find_left_idx(2.0, 0) == 6);
    CHECK(regular_grid.find_left_idx(2.5, 0) == -1);

    const auto values = sample(fields::bilinear, grid.points(), 2, 2);
    const RegularGridInterpolator interp(field_values_t(values.data(), grid.size(), size_t(2)), axes, true);
    CHECK(interp.grid().ndim() == 2);
    CHECK(interp.n_points() == int(grid.size()));
    CHECK_THAT(interpolation_error(interp, fields::bilinear, random_points(200, {-2, -1.5}, {2, 1.5}), 2), IsAtMost(1e-13));

    // With the point axis last
    const auto transposed = transpose(values, grid.size(), 2);
    const RegularGridInterpolator last(field_values_t(transposed.data(), size_t(2), grid.size()), axes, false);
    CHECK_THAT(interpolation_error(last, fields::bilinear, grid.points(), 2), IsAtMost(1e-14));

    const double beyond[] = {0.0, 1.6};
    double out[2];
    CHECK_FALSE(interp.contains(beyond));
    CHECK_FALSE(interp.interp(out, beyond));
}

TEST_CASE("A compiled scattered interpolator triangulates its points or shares a triangulation", "[crafted][nd-interpolation][scattered]"){
    auto points = random_points(60, {-2, -1.5}, {2, 1.5});
    points.insert(points.end(), {-2, -1.5, 2, -1.5, 2, 1.5, -2, 1.5});
    const size_t n = points.size() / 2;
    const auto values = sample(fields::affine, points, 2, 2);
    const auto inside = random_points(200, {-2, -1.5}, {2, 1.5}, 3);

    const ScatteredNdInterpolator interp(points.data(), field_values_t(values.data(), n, size_t(2)), 2, true);
    CHECK(interp.ndim() == 2);
    CHECK(interp.points().shape(0) == n);
    CHECK_THAT(interpolation_error(interp, fields::affine, inside, 2), IsAtMost(1e-13));

    const TriView& tri = interp.tri();
    CHECK(tri->npoints() == int(n));
    CHECK_THAT(abs_error(tri.total_volume(), 12.0), IsAtMost(1e-12));
    const ScatteredNdInterpolator shared(tri, field_values_t(values.data(), n, size_t(2)), true);
    CHECK(shared.tri().shared_tri() == tri.shared_tri());
    CHECK_THAT(interpolation_error(shared, fields::affine, inside, 2), IsAtMost(1e-13));

    // A triangulation of its own
    const DelaunayTri own(points.data(), n, 2);
    CHECK(own.nsimplices() == tri->nsimplices());
    const double x[] = {0.33, -0.71};
    CHECK(own.find_simplex(x, -1) >= 0);
}

TEST_CASE("Compiled vector fields integrate their streamlines", "[crafted][nd-interpolation][vector-field]"){
    const Rectilinear grid = custom_grid();
    const auto points = grid.points();
    const auto values = sample(fields::rotation, points, 2, 2);
    const field_values_t view(values.data(), grid.size(), size_t(2));

    const RegularVectorField regular(view, axes_of(grid), CoordType::Cartesian, true);
    const ScatteredVectorField scattered(points.data(), view, 2, true);
    const ScatteredVectorField shared(scattered.tri(), view, true);

    for (const VirtualVectorField* field : {static_cast<const VirtualVectorField*>(&regular), static_cast<const VirtualVectorField*>(&scattered),
                                            static_cast<const VirtualVectorField*>(&shared)}){
        CHECK(field->ndim() == 2);
        CHECK(field->contains(start));

        const pbox::Box<OdeResult<double, FieldSD>> line = field->streamline(start, 2.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, false);
        CHECK(line->t().back() == 2.0);
        CHECK_THAT(distance_to_circle(*line, 1.2, 1.0), IsAtMost(1e-8));

        const std::vector<double> t_eval = {0.5, 1.0};
        const auto sampled = field->streamline(start, 1.0, 1e-10, 1e-10, 0, 0, 0, 1, rk45, true, t_eval);
        CHECK(sampled->t() == t_eval);
        CHECK_THAT(distance_to_circle(*sampled, 1.2, 1 / 1.2), IsAtMost(1e-8));

        // Backward, through the streamline's own ODE
        pbox::Box<OdeDriver<double, FieldSD>> ode = field->get_streamline_ode(start, 1e-10, 1e-10, 0, 0, 0, -1, rk45, false);
        OdeResult<double, FieldSD> backward;
        REQUIRE(ode->integrate_until(&backward, -2.0));
        CHECK_THAT(distance_to_circle(backward, 1.2, 1.0), IsAtMost(1e-8));
    }

    SECTION("and lay them out over the grid"){
        const double ds = 0.05;
        const auto lines = regular.streamplot_data(3.0, ds, 10, 1e-9, 1e-9, 0, 0, 0, rk45);
        REQUIRE_FALSE(lines.empty());
        for (const auto& line : lines){
            CHECK(line.shape(0) == 2);
            CHECK(line.shape(1) >= 2);
        }
        CHECK_THROWS_AS(regular.streamplot_data(3.0, ds, 1, 1e-9, 1e-9, 0, 0, 0, rk45), std::domain_error);
    }
}

TEST_CASE("A compiled vector field in polar coordinates holds physical components", "[crafted][nd-interpolation][vector-field][coordinates]"){
    // v = r e_phi turns at dphi/ds = 1/r once normalized to unit speed
    const Rectilinear grid{{linspace(0.5, 2, 7), linspace(0, 2 * pi, 13)}};
    const auto values = sample([](double* out, const double* x){ out[0] = 0; out[1] = x[0]; }, grid.points(), 2, 2);
    const RegularVectorField field(field_values_t(values.data(), grid.size(), size_t(2)), axes_of(grid), CoordType::Polar, true);
    const double from[] = {1.5, 0.2};
    const auto line = field.streamline(from, 1.5, 1e-10, 1e-10, 0, 0, 0, 1, rk45, true);
    REQUIRE_FALSE(line->q().empty());
    CHECK_THAT(abs_error(line->q().back()[0], 1.5), IsAtMost(1e-8));
    CHECK_THAT(abs_error(line->q().back()[1], 1.2), IsAtMost(1e-8));
}

} // namespace
} // namespace odetest
