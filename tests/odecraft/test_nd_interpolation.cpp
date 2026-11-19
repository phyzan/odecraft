// Interpolation of sampled fields: rectilinear grids, Delaunay triangulations and the
// interpolators over them, checked against the analytic fields they sample. The same samples are
// also read both as a regular grid and as scattered points. Vector fields and their streamlines
// are in test_vector_fields.cpp.

#include "odecraft_test.hpp"
#include "grids.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace odetest {
namespace {

namespace rgi = ode::interp::rgi;
namespace sci = ode::interp::sci;
using ode::interp::VirtualNdInterpolator;

// An interpolator's dimension is fixed at compile time or decided at runtime (0), and it is
// reached directly or through VirtualNdInterpolator (true)
using Grid2 = rgi::RegularGridInterpolator<2>;
using GridN = rgi::RegularGridInterpolator<0>;
using VirtualGrid2 = rgi::RegularGridInterpolator<2, true>;
using VirtualGridN = rgi::RegularGridInterpolator<0, true>;

using Scattered2 = sci::ScatteredNdInterpolator<2>;
using ScatteredN = sci::ScatteredNdInterpolator<0>;
using VirtualScattered2 = sci::ScatteredNdInterpolator<2, true>;
using VirtualScatteredN = sci::ScatteredNdInterpolator<0, true>;

using Tri1 = sci::DelaunayTri<1>;
using Tri2 = sci::DelaunayTri<2>;
using Tri3 = sci::DelaunayTri<3>;
using TriN = sci::DelaunayTri<0>;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/// @brief Points around the custom grid, outside of it, and one that is not a point at all.
const std::vector<std::vector<double>> outside_custom_grid = {
    {-2.1, 0}, {2.1, 0}, {0, -1.6}, {0, 1.6}, {3, 3}, {-2.5, -2}, {nan, 0},
};

/// @brief The largest difference between what an interpolator gives and the field it samples,
/// over an array of points of shape (n, ndim). NaN if the interpolator declines any of them.
template<typename Interpolator>
Magnitude interpolation_error(const Interpolator& interp, const Field& f, const std::vector<double>& points, size_t ndim, size_t nfields){
    std::vector<double> got(nfields), expected(nfields);
    Magnitude err{0};
    for (size_t p = 0; p < points.size() / ndim; p++){
        const double* x = points.data() + p * ndim;
        if (!interp.interp(got.data(), x)){
            return {nan};
        }
        f(expected.data(), x);
        for (size_t k = 0; k < nfields; k++){
            err = worst(abs_error(got[k], expected[k]), err);
        }
    }
    return err;
}

/// @brief How many of the points an interpolator, or a triangulation, finds inside its domain.
template<typename Domain>
size_t count_contained(const Domain& domain, const std::vector<std::vector<double>>& points){
    return size_t(std::count_if(points.begin(), points.end(), [&](const auto& x){ return domain.contains(x.data()); }));
}

/// @brief Checks the interpolator declines every point outside the custom grid.
template<typename Interpolator>
void check_declines_outside(const Interpolator& interp, size_t nfields){
    std::vector<double> out(nfields);
    CHECK(count_contained(interp, outside_custom_grid) == 0);
    for (const auto& x : outside_custom_grid){
        CAPTURE(x);
        CHECK_FALSE(interp.interp(out.data(), x.data()));
    }
}

/// @brief The interpolators of the custom grid's samples of a field.
template<typename Interpolator>
Interpolator interpolator_of(const Field& f, size_t nfields, bool coord_axis_first = true){
    const Rectilinear grid = custom_grid();
    const auto values = sample(f, grid.points(), 2, nfields);
    const auto layout = coord_axis_first ? values : transpose(values, grid.size(), nfields);
    const auto view = coord_axis_first ? view_of(layout, grid.size(), nfields) : view_of(layout, nfields, grid.size());
    if constexpr (requires { Interpolator(view, grid.axes, coord_axis_first); }){
        return Interpolator(view, grid.axes, coord_axis_first);
    } else {
        return Interpolator(grid.points().data(), view, 2, coord_axis_first);
    }
}


// ============================================================================================
//                                   Rectilinear grids
// ============================================================================================

TEST_CASE("A rectilinear grid locates coordinates along its axes", "[nd-interpolation][grid]"){
    const Rectilinear grid = custom_grid();
    const auto check = [&grid](const auto& g){
        CHECK(g.ndim() == 2);
        CHECK(g.data().size() == 2);
        CHECK(g.shape()[0] == 8);
        CHECK(g.shape()[1] == 6);
        CHECK(g.length(0) == 4.0);
        CHECK(g.length(1) == 3.0);
        for (int axis = 0; axis < 2; axis++){
            CHECK(std::equal(grid.axes[axis].begin(), grid.axes[axis].end(), g.x(axis).data()));
        }

        // Both ends belong to the grid
        CHECK(g.contains(-2.0, 0));
        CHECK(g.contains(2.0, 0));
        CHECK_FALSE(g.contains(2.0001, 0));
        CHECK_FALSE(g.contains(-1.6, 1));
        CHECK_FALSE(g.contains(nan, 0));
        const double corner[] = {2.0, -1.5}, inside[] = {0.1, 0.2}, beyond[] = {0.1, 1.7};
        CHECK(g.contains(corner));
        CHECK(g.contains(inside));
        CHECK_FALSE(g.contains(beyond));

        // A coordinate falls in the cell its node starts, and the last node closes the last cell
        for (int axis = 0; axis < 2; axis++){
            const auto& x = grid.axes[axis];
            const int n = int(x.size());
            for (int k = 0; k + 1 < n; k++){
                CAPTURE(axis, k);
                CHECK(g.find_left_idx(x[k], axis) == k);
                CHECK(g.find_left_idx(0.5 * (x[k] + x[k + 1]), axis) == k);
            }
            CHECK(g.find_left_idx(x.back(), axis) == n - 2);
            CHECK(g.find_left_idx(x.front() - 0.1, axis) == -1);
            CHECK(g.find_left_idx(x.back() + 0.1, axis) == -1);
        }
    };
    SECTION("with its dimension fixed at compile time"){
        check(rgi::RegularGrid<double, 2>(grid.axes));
    }
    SECTION("with its dimension decided at runtime"){
        check(rgi::RegularGrid<double, 0>(grid.axes));
    }
}


// ============================================================================================
//                                Regular grid interpolators
// ============================================================================================

TEMPLATE_TEST_CASE("A grid interpolator reproduces every multilinear function", "[nd-interpolation][regular]", Grid2, GridN, VirtualGrid2, VirtualGridN){
    using Interpolator = TestType;
    const Rectilinear grid = custom_grid();
    std::vector<double> values = sample(fields::bilinear, grid.points(), 2, 2);
    const Interpolator interp(view_of(values, grid.size(), 2), grid.axes, true);
    const auto inside = random_points(200, {-2, -1.5}, {2, 1.5});

    CHECK(interp.ndim() == 2);
    CHECK(interp.grid().ndim() == 2);
    CHECK(interp.n_points() == int(grid.size()));
    CHECK(interp.nvals_per_point() == 2);
    CHECK(interp.output_dims() == 1);
    CHECK(interp.output_shape()[0] == 2);
    CHECK(interp.values().size() == values.size());

    SECTION("at its samples"){
        CHECK_THAT(interpolation_error(interp, fields::bilinear, grid.points(), 2, 2), IsAtMost(1e-14));
    }
    SECTION("between them"){
        CHECK(count_contained(interp, {{-2, -1.5}, {2, 1.5}, {0.33, -0.71}}) == 3);
        CHECK_THAT(interpolation_error(interp, fields::bilinear, inside, 2, 2), IsAtMost(1e-13));
    }
    SECTION("from a copy of its samples, which the caller may discard"){
        std::fill(values.begin(), values.end(), 0.0);
        CHECK_THAT(interpolation_error(interp, fields::bilinear, inside, 2, 2), IsAtMost(1e-13));
    }
    SECTION("but not outside the grid"){
        check_declines_outside(interp, 2);
    }
    SECTION("normalized to a unit vector"){
        const double x[] = {0.33, -0.71};
        double out[2], expected[2];
        REQUIRE(interp.norm_interp(out, x));
        fields::bilinear(expected, x);
        const double norm = std::hypot(expected[0], expected[1]);
        CHECK_THAT(abs_error(out[0], expected[0] / norm), IsAtMost(1e-14));
        CHECK_THAT(abs_error(out[1], expected[1] / norm), IsAtMost(1e-14));
        const double beyond[] = {0, 2};
        CHECK_FALSE(interp.norm_interp(out, beyond));
    }
    if constexpr (std::is_base_of_v<VirtualNdInterpolator, Interpolator>){
        SECTION("through VirtualNdInterpolator"){
            const VirtualNdInterpolator& base = interp;
            CHECK(base.ndim() == 2);
            CHECK(base.nvals_per_point() == 2);
            CHECK(base.output_dims() == 1);
            CHECK(base.output_shape()[0] == 2);
            CHECK_THAT(interpolation_error(base, fields::bilinear, inside, 2, 2), IsAtMost(1e-13));
        }
    }
}

TEMPLATE_TEST_CASE("A grid interpolator reads its values with the point axis first or last", "[nd-interpolation][regular][layout]", Grid2, GridN, VirtualGrid2, VirtualGridN){
    using Interpolator = TestType;
    const Rectilinear grid = custom_grid();
    const size_t n = grid.size();
    const auto inside = random_points(50, {-2, -1.5}, {2, 1.5});
    std::vector<double> out_first(6), out_last(6);

    SECTION("vectors"){
        const auto first = interpolator_of<Interpolator>(fields::bilinear, 2, true);
        const auto last = interpolator_of<Interpolator>(fields::bilinear, 2, false);
        CHECK(last.nvals_per_point() == 2);
        CHECK(last.n_points() == int(n));
        for (size_t p = 0; p < inside.size() / 2; p++){
            REQUIRE(first.interp(out_first.data(), &inside[2 * p]));
            REQUIRE(last.interp(out_last.data(), &inside[2 * p]));
            CHECK(std::equal(out_first.begin(), out_first.begin() + 2, out_last.begin()));
        }
    }
    SECTION("tensors, which keep their shape"){
        // A 2 x 3 tensor at every point, its entries affine combinations of the bilinear fields
        const Field tensor = [](double* out, const double* x){
            double f[2];
            fields::bilinear(f, x);
            for (int k = 0; k < 6; k++){
                out[k] = (k + 1) * f[k % 2] - k;
            }
        };
        const auto values = sample(tensor, grid.points(), 2, 6);
        const auto transposed = transpose(values, n, 6);
        const Interpolator first(view_of(values, n, 2, 3), grid.axes, true);
        const Interpolator last(view_of(transposed, 2, 3, n), grid.axes, false);
        for (const Interpolator* interp : {&first, &last}){
            CHECK(interp->nvals_per_point() == 6);
            CHECK(interp->output_dims() == 2);
            CHECK(interp->output_shape()[0] == 2);
            CHECK(interp->output_shape()[1] == 3);
            CHECK_THAT(interpolation_error(*interp, tensor, inside, 2, 6), IsAtMost(1e-12));
        }
    }
    SECTION("scalars, as a flat array"){
        std::vector<double> values(n);
        const auto points = grid.points();
        for (size_t p = 0; p < n; p++){
            fields::bilinear(out_first.data(), &points[2 * p]);
            values[p] = out_first[0];
        }
        const Interpolator interp(view_of(values, n), grid.axes, true);
        CHECK(interp.nvals_per_point() == 1);
        CHECK(interp.output_dims() == 1);
        CHECK(interp.output_shape()[0] == 1);
        const Field first_field = [](double* out, const double* x){ double f[2]; fields::bilinear(f, x); out[0] = f[0]; };
        CHECK_THAT(interpolation_error(interp, first_field, inside, 2, 1), IsAtMost(1e-13));
    }
}

TEST_CASE("Multilinear interpolation of a smooth function is second-order accurate", "[nd-interpolation][regular][convergence]"){
    const auto probes = random_points(200, {-2, -1.5}, {2, 1.5});
    std::vector<double> errors;
    for (size_t n : {11, 21, 41, 81}){
        const Rectilinear grid{{linspace(-2, 2, n), linspace(-1.5, 1.5, n)}};
        const auto values = sample(fields::smooth, grid.points(), 2, 1);
        const Grid2 interp(view_of(values, grid.size(), 1), grid.axes, true);
        errors.push_back(interpolation_error(interp, fields::smooth, probes, 2, 1).value);
    }
    // Halving the spacing divides the error by about 4
    for (size_t i = 0; i + 1 < errors.size(); i++){
        CAPTURE(i, errors[i], errors[i + 1]);
        CHECK(errors[i] / errors[i + 1] > 3.0);
        CHECK(errors[i] / errors[i + 1] < 5.5);
    }
}

TEST_CASE("Multilinear interpolation works in any number of dimensions", "[nd-interpolation][regular]"){
    SECTION("one"){
        const std::vector<std::vector<double>> axes = {{-1.0, -0.3, 0.4, 2.0}};
        const Field line = [](double* out, const double* x){ out[0] = 2 - 3 * x[0]; };
        const auto values = sample(line, {-1.0, -0.3, 0.4, 2.0}, 1, 1);
        for_each_type<std::tuple<rgi::RegularGridInterpolator<1>, GridN>>([&]<typename Interpolator>(){
            const Interpolator interp(view_of(values, 4), axes, true);
            CHECK(interp.ndim() == 1);
            CHECK_THAT(interpolation_error(interp, line, random_points(50, {-1}, {2}), 1, 1), IsAtMost(1e-14));
            const double beyond[] = {2.5};
            double out[1];
            CHECK_FALSE(interp.interp(out, beyond));
        });
    }
    SECTION("three"){
        const Rectilinear grid{{{-1.0, 0.0, 0.5, 1.0}, {0.0, 1.0, 3.0}, {-2.0, -1.0, 0.0, 1.5, 2.0}}};
        const auto values = sample(fields::trilinear, grid.points(), 3, 2);
        const auto inside = random_points(200, {-1, 0, -2}, {1, 3, 2});
        for_each_type<std::tuple<rgi::RegularGridInterpolator<3>, GridN>>([&]<typename Interpolator>(){
            const Interpolator interp(view_of(values, grid.size(), 2), grid.axes, true);
            CHECK(interp.ndim() == 3);
            CHECK(interp.n_points() == int(grid.size()));
            CHECK_THAT(interpolation_error(interp, fields::trilinear, inside, 3, 2), IsAtMost(1e-13));
            const double beyond[] = {0, 3.5, 0};
            double out[2];
            CHECK_FALSE(interp.contains(beyond));
            CHECK_FALSE(interp.interp(out, beyond));
        });
    }
}


// ============================================================================================
//                                Delaunay triangulations
// ============================================================================================

/// @brief How the faces of a triangulation fit together.
struct Adjacency {
    size_t boundary_faces = 0; // faces with no simplex on their other side
    size_t broken_faces = 0;   // faces whose neighbor does not share them, or does not point back
};

/// @brief Every face is shared by the two simplices on its sides, or lies on the hull. Neighbor i
/// of a simplex is the one across the face opposite its vertex i.
template<typename Tri>
Adjacency adjacency_of(const Tri& tri){
    const int nd = tri.ndim();
    Adjacency out;
    for (int s = 0; s < tri.nsimplices(); s++){
        const int* v = tri.get_simplex(s);
        const int* across = tri.get_neighbors(s);
        for (int i = 0; i <= nd; i++){
            if (across[i] == -1){
                ++out.boundary_faces;
                continue;
            }
            const int* w = tri.get_simplex(across[i]);
            const int* back = tri.get_neighbors(across[i]);
            const auto holds = [&](int vertex){ return std::find(w, w + nd + 1, vertex) != w + nd + 1; };
            bool shared = std::find(back, back + nd + 1, s) != back + nd + 1;
            for (int j = 0; j <= nd; j++){
                shared = shared && (holds(v[j]) == (j != i));
            }
            out.broken_faces += shared ? 0 : 1;
        }
    }
    return out;
}

/// @brief The largest distance from a point to its reconstruction from its barycentric coordinates
/// in the simplex found for it, NaN if it is not found or its coordinates are not within [0, 1].
template<typename Tri>
Magnitude barycentric_error(const Tri& tri, const std::vector<double>& points){
    const int nd = tri.ndim();
    std::vector<double> bary(nd + 1), back(nd);
    Magnitude err{0};
    for (size_t p = 0; p < points.size() / nd; p++){
        const double* x = points.data() + p * nd;
        const int s = tri.find_simplex(x);
        if (s < 0 || !tri.compute_barycentric(bary.data(), s, x)){
            return {nan};
        }
        double sum = 0;
        std::fill(back.begin(), back.end(), 0.0);
        for (int i = 0; i <= nd; i++){
            if (bary[i] < -1e-12 || bary[i] > 1 + 1e-12){
                return {nan};
            }
            sum += bary[i];
            for (int a = 0; a < nd; a++){
                back[a] += bary[i] * tri.get_points()(tri.get_simplex(s)[i], a);
            }
        }
        err = worst(abs_error(sum, 1.0), err);
        for (int a = 0; a < nd; a++){
            err = worst(abs_error(back[a], x[a]), err);
        }
    }
    return err;
}

/// @brief The largest difference between a triangulation's interpolation of the samples of a field
/// and the field itself, over an array of points.
template<typename Tri>
Magnitude tri_interpolation_error(const Tri& tri, const Field& f, const std::vector<double>& samples, const std::vector<double>& points, size_t nfields){
    const size_t nd = tri.ndim();
    std::vector<double> got(nfields), expected(nfields);
    Magnitude err{0};
    for (size_t p = 0; p < points.size() / nd; p++){
        const double* x = points.data() + p * nd;
        if (!tri.interpolate(got.data(), x, samples.data(), int(nfields))){
            return {nan};
        }
        f(expected.data(), x);
        for (size_t k = 0; k < nfields; k++){
            err = worst(abs_error(got[k], expected[k]), err);
        }
    }
    return err;
}

/// @brief Scattered points over [-2, 2] x [-1.5, 1.5], with the corners, so that their hull is that box.
std::vector<double> scattered_points(){
    auto points = random_points(60, {-2, -1.5}, {2, 1.5});
    points.insert(points.end(), {-2, -1.5, 2, -1.5, 2, 1.5, -2, 1.5});
    return points;
}


TEMPLATE_TEST_CASE("A Delaunay triangulation of a rectilinear grid splits every cell along a diagonal", "[nd-interpolation][delaunay]", Tri2, TriN){
    const Rectilinear grid = custom_grid();
    const auto points = grid.points();
    const TestType tri(points.data(), grid.size(), 2);
    const size_t nx = grid.axes[0].size(), ny = grid.axes[1].size();

    CHECK(tri.ndim() == 2);
    CHECK(tri.npoints() == int(grid.size()));
    CHECK(std::equal(points.begin(), points.end(), tri.get_points().data()));
    REQUIRE(tri.nsimplices() == int(2 * (nx - 1) * (ny - 1)));
    CHECK(tri.get_simplices().shape(0) == size_t(tri.nsimplices()));
    CHECK(tri.get_simplices().shape(1) == 3);
    CHECK(tri.get_neighbors().shape(1) == 3);
    CHECK_THAT(abs_error(tri.total_volume(), grid.volume()), IsAtMost(1e-12));

    const Adjacency adjacency = adjacency_of(tri);
    CHECK(adjacency.broken_faces == 0);
    CHECK(adjacency.boundary_faces == 2 * (nx - 1) + 2 * (ny - 1));

    // Each triangle is half of one cell: its corners span one step along each axis
    size_t halves = 0;
    for (int s = 0; s < tri.nsimplices(); s++){
        size_t lo[2] = {nx, ny}, hi[2] = {0, 0};
        for (int i = 0; i < 3; i++){
            const auto idx = grid.index(size_t(tri.get_simplex(s)[i]));
            for (int a = 0; a < 2; a++){
                lo[a] = std::min(lo[a], idx[a]);
                hi[a] = std::max(hi[a], idx[a]);
            }
        }
        const double cell = (grid.axes[0][hi[0]] - grid.axes[0][lo[0]]) * (grid.axes[1][hi[1]] - grid.axes[1][lo[1]]);
        const bool half = hi[0] == lo[0] + 1 && hi[1] == lo[1] + 1 && std::abs(tri.volume(s) - cell / 2) < 1e-12;
        halves += half ? 1 : 0;
    }
    CHECK(halves == size_t(tri.nsimplices()));
}

TEMPLATE_TEST_CASE("Barycentric coordinates place a point within the simplex found for it", "[nd-interpolation][delaunay]", Tri2, TriN){
    const auto points = scattered_points();
    const TestType tri(points.data(), points.size() / 2, 2);
    REQUIRE(tri.nsimplices() > 0);
    CHECK(adjacency_of(tri).broken_faces == 0);
    CHECK_THAT(abs_error(tri.total_volume(), 12.0), IsAtMost(1e-12));

    SECTION("inside the hull"){
        CHECK_THAT(barycentric_error(tri, random_points(300, {-2, -1.5}, {2, 1.5}, 99)), IsAtMost(1e-13));
        // Including the points the triangulation is made of
        CHECK_THAT(barycentric_error(tri, points), IsAtMost(1e-13));
    }
    SECTION("but not outside it"){
        for (const auto& x : outside_custom_grid){
            CAPTURE(x);
            CHECK(tri.find_simplex(x.data()) == -1);
        }
        CHECK(count_contained(tri, outside_custom_grid) == 0);
    }
    SECTION("from any simplex the previous search ended in"){
        // The search walks from where the last one ended: alternate between far apart points
        const auto inside = random_points(100, {-2, -1.5}, {2, 1.5}, 5);
        std::vector<double> zigzag;
        for (size_t p = 0; p < inside.size() / 2; p++){
            zigzag.insert(zigzag.end(), {inside[2 * p], inside[2 * p + 1], -inside[2 * p], -inside[2 * p + 1]});
        }
        CHECK_THAT(barycentric_error(tri, zigzag), IsAtMost(1e-13));
    }
}

TEMPLATE_TEST_CASE("Barycentric interpolation is exact for affine fields", "[nd-interpolation][delaunay]", Tri2, TriN){
    const auto points = scattered_points();
    const size_t n = points.size() / 2;
    const TestType tri(points.data(), n, 2);
    const auto samples = sample(fields::affine, points, 2, 2);
    const auto inside = random_points(300, {-2, -1.5}, {2, 1.5}, 3);

    CHECK_THAT(tri_interpolation_error(tri, fields::affine, samples, inside, 2), IsAtMost(1e-13));

    // One field on its own
    std::vector<double> first(n);
    for (size_t p = 0; p < n; p++){
        first[p] = samples[2 * p];
    }
    const double x[] = {0.33, -0.71};
    double expected[2];
    fields::affine(expected, x);
    CHECK_THAT(abs_error(tri.get_value(x, first.data()), expected[0]), IsAtMost(1e-14));

    // Outside the hull there is nothing to interpolate
    double out[2] = {0, 0};
    const double beyond[] = {2.5, 0};
    CHECK_FALSE(tri.interpolate(out, beyond, samples.data(), 2));
    CHECK(std::isnan(out[0]));
    CHECK(std::isnan(out[1]));
    CHECK(std::isnan(tri.get_value(beyond, first.data())));
}

TEMPLATE_TEST_CASE("A triangulation in one dimension joins its points in order", "[nd-interpolation][delaunay]", Tri1, TriN){
    const std::vector<double> points = {0.5, -1.0, 2.0, 0.0, 1.2};
    const TestType tri(points.data(), points.size(), 1);
    REQUIRE(tri.nsimplices() == 4);
    CHECK(tri.ndim() == 1);

    // In order: -1 (1), 0 (3), 0.5 (0), 1.2 (4), 2 (2)
    const std::vector<std::vector<int>> segments = {{1, 3}, {3, 0}, {0, 4}, {4, 2}};
    for (int s = 0; s < 4; s++){
        CAPTURE(s);
        CHECK(std::vector<int>(tri.get_simplex(s), tri.get_simplex(s) + 2) == segments[s]);
    }
    // Neighbor 0 is across from the left end, to the right
    CHECK(tri.get_neighbors(0)[0] == 1);
    CHECK(tri.get_neighbors(0)[1] == -1);
    CHECK(tri.get_neighbors(3)[0] == -1);
    CHECK(tri.get_neighbors(3)[1] == 2);
    CHECK(adjacency_of(tri).broken_faces == 0);
    CHECK_THAT(abs_error(tri.volume(1), 0.5), IsAtMost(1e-15));
    CHECK_THAT(abs_error(tri.total_volume(), 3.0), IsAtMost(1e-14));

    const Field line = [](double* out, const double* x){ out[0] = 2 - 3 * x[0]; };
    const auto samples = sample(line, points, 1, 1);
    CHECK_THAT(tri_interpolation_error(tri, line, samples, random_points(100, {-1}, {2}), 1), IsAtMost(1e-14));
    const double x = 0.7, beyond = 2.5;
    CHECK(tri.find_simplex(&x) == 2);
    CHECK_FALSE(tri.contains(&beyond));
}

TEMPLATE_TEST_CASE("A triangulation in three dimensions locates and interpolates within its hull", "[nd-interpolation][delaunay]", Tri3, TriN){
    const Rectilinear grid{{{-1.0, 0.0, 1.0}, {-1.0, 0.5, 1.0}, {-1.0, -0.2, 1.0}}};
    auto points = grid.points();
    const auto extra = random_points(20, {-1, -1, -1}, {1, 1, 1}, 11);
    points.insert(points.end(), extra.begin(), extra.end());
    const TestType tri(points.data(), points.size() / 3, 3);

    CHECK(tri.ndim() == 3);
    REQUIRE(tri.nsimplices() > 0);
    CHECK(tri.get_simplices().shape(1) == 4);
    CHECK(tri.get_invT().shape(1) == 3);
    CHECK(tri.get_invT().shape(2) == 3);
    CHECK(adjacency_of(tri).broken_faces == 0);

    const auto inside = random_points(300, {-1, -1, -1}, {1, 1, 1}, 13);
    CHECK_THAT(barycentric_error(tri, inside), IsAtMost(1e-13));
    const Field affine3 = [](double* out, const double* x){ out[0] = 1 + x[0] - 2 * x[1] + 3 * x[2]; };
    CHECK_THAT(tri_interpolation_error(tri, affine3, sample(affine3, points, 3, 1), inside, 1), IsAtMost(1e-13));
    const double beyond[] = {0, 0, 1.1};
    CHECK_FALSE(tri.contains(beyond));
}

TEMPLATE_TEST_CASE("Every simplex of a triangulation has a volume even a flat one", "[nd-interpolation][delaunay]", Tri3, TriN){
    // The corners of a box lie on a sphere, so the Delaunay triangulation of a grid is not unique,
    // and the one Qhull settles on may keep flat tetrahedra: their volume is zero, not undefined
    const Rectilinear grid{{{-1.0, 0.0, 1.0}, {-1.0, 0.5, 1.0}, {-1.0, -0.2, 1.0}}};
    const auto points = grid.points();
    const TestType tri(points.data(), grid.size(), 3);
    REQUIRE(tri.nsimplices() > 0);
    size_t undefined = 0;
    for (int s = 0; s < tri.nsimplices(); s++){
        undefined += std::isfinite(tri.volume(s)) ? 0 : 1;
    }
    CHECK(undefined == 0);
    CHECK_THAT(abs_error(tri.total_volume(), grid.volume()), IsAtMost(1e-12));
}

TEMPLATE_TEST_CASE("Too few points make an empty triangulation", "[nd-interpolation][delaunay]", Tri2, TriN){
    const std::vector<double> points = {0, 0, 1, 1};
    const TestType tri(points.data(), 2, 2);
    CHECK(tri.npoints() == 2);
    CHECK(tri.nsimplices() == 0);
    CHECK(tri.total_volume() == 0);
    const double x[] = {0.5, 0.5};
    CHECK_FALSE(tri.contains(x));
    double out = 0;
    const double samples[] = {1, 2};
    CHECK_FALSE(tri.interpolate(&out, x, samples));
    CHECK(std::isnan(out));
}

TEST_CASE("Points that span no area cannot be triangulated", "[nd-interpolation][delaunay]"){
    const std::vector<double> collinear = {0, 0, 1, 1, 2, 2, 3, 3};
    const SilenceStderr quiet; // Qhull explains at length why it gave up
    CHECK_THROWS_AS(Tri2(collinear.data(), 4, 2), std::runtime_error);
}

TEST_CASE("A triangulation is moved or restored from its arrays but never copied", "[nd-interpolation][delaunay]"){
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<Tri2>);
    STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<Tri2>);
    STATIC_REQUIRE(std::is_nothrow_move_constructible_v<Tri2>);

    const auto points = scattered_points();
    Tri2 original(points.data(), points.size() / 2, 2);
    const auto probes = random_points(100, {-2, -1.5}, {2, 1.5}, 17);
    std::vector<int> found;
    for (size_t p = 0; p < probes.size() / 2; p++){
        found.push_back(original.find_simplex(&probes[2 * p]));
    }
    const auto same_answers = [&](const Tri2& tri){
        std::vector<int> again;
        for (size_t p = 0; p < probes.size() / 2; p++){
            again.push_back(tri.find_simplex(&probes[2 * p]));
        }
        return again == found;
    };

    SECTION("restored from the arrays of another"){
        using ndspan::CView2D;
        using ndspan::CView3D;
        const auto view = []<typename View, typename Array>(std::type_identity<View>, const Array& a){
            return View(a.data(), a.shape(), a.ndim());
        };
        Tri2 restored;
        restored.set_state(
            view(std::type_identity<CView2D<const double, 0, 2>>{}, original.get_points()),
            view(std::type_identity<CView2D<const int, 0, 3>>{}, original.get_simplices()),
            view(std::type_identity<CView2D<const int, 0, 3>>{}, original.get_neighbors()),
            view(std::type_identity<CView2D<const double, 0, 2>>{}, original.get_v0()),
            view(std::type_identity<CView3D<const double, 0, 2, 2>>{}, original.get_invT()));
        CHECK(restored.npoints() == original.npoints());
        CHECK(restored.nsimplices() == original.nsimplices());
        CHECK(restored.total_volume() == original.total_volume());
        CHECK(same_answers(restored));
    }
    SECTION("moved"){
        Tri2 moved(std::move(original));
        CHECK(same_answers(moved));
        Tri2 assigned;
        assigned = std::move(moved);
        CHECK(same_answers(assigned));
    }
}


// ============================================================================================
//                                Scattered interpolators
// ============================================================================================

TEMPLATE_TEST_CASE("A scattered interpolator reproduces every affine function", "[nd-interpolation][scattered]", Scattered2, ScatteredN, VirtualScattered2, VirtualScatteredN){
    using Interpolator = TestType;
    const auto points = scattered_points();
    const size_t n = points.size() / 2;
    std::vector<double> values = sample(fields::affine, points, 2, 2);
    const Interpolator interp(points.data(), view_of(values, n, 2), 2, true);
    const auto inside = random_points(200, {-2, -1.5}, {2, 1.5}, 23);

    CHECK(interp.ndim() == 2);
    CHECK(interp.n_points() == int(n));
    CHECK(interp.nvals_per_point() == 2);
    CHECK(interp.output_dims() == 1);
    CHECK(interp.output_shape()[0] == 2);
    CHECK(std::equal(points.begin(), points.end(), interp.points().data()));
    REQUIRE(interp.tri() != nullptr);
    CHECK(interp.tri()->npoints() == int(n));

    SECTION("at its samples"){
        CHECK_THAT(interpolation_error(interp, fields::affine, points, 2, 2), IsAtMost(1e-13));
    }
    SECTION("between them, from its own copy of them"){
        std::fill(values.begin(), values.end(), 0.0);
        CHECK_THAT(interpolation_error(interp, fields::affine, inside, 2, 2), IsAtMost(1e-13));
    }
    SECTION("but not outside their hull"){
        check_declines_outside(interp, 2);
    }
    SECTION("normalized to a unit vector"){
        const double x[] = {0.33, -0.71};
        double out[2], expected[2];
        REQUIRE(interp.norm_interp(out, x));
        fields::affine(expected, x);
        const double norm = std::hypot(expected[0], expected[1]);
        CHECK_THAT(abs_error(out[0], expected[0] / norm), IsAtMost(1e-14));
        CHECK_THAT(abs_error(out[1], expected[1] / norm), IsAtMost(1e-14));
    }
    if constexpr (std::is_base_of_v<VirtualNdInterpolator, Interpolator>){
        SECTION("through VirtualNdInterpolator"){
            const VirtualNdInterpolator& base = interp;
            CHECK(base.ndim() == 2);
            CHECK(base.nvals_per_point() == 2);
            CHECK_THAT(interpolation_error(base, fields::affine, inside, 2, 2), IsAtMost(1e-13));
        }
    }
}

TEMPLATE_TEST_CASE("Scattered interpolators share a triangulation and read either layout", "[nd-interpolation][scattered][layout]", Scattered2, ScatteredN){
    using Interpolator = TestType;
    const auto points = scattered_points();
    const size_t n = points.size() / 2;
    const auto affine = sample(fields::affine, points, 2, 2);
    const auto bilinear_last = transpose(sample(fields::bilinear, points, 2, 2), n, 2);
    const auto inside = random_points(100, {-2, -1.5}, {2, 1.5}, 29);

    const Interpolator first(points.data(), view_of(affine, n, 2), 2, true);
    const Interpolator second(first.tri(), view_of(bilinear_last, 2, n), false);
    CHECK(second.tri() == first.tri());
    CHECK(first.tri().use_count() >= 2);
    CHECK_THAT(interpolation_error(first, fields::affine, inside, 2, 2), IsAtMost(1e-13));
    // A bilinear field is not interpolated exactly, but what is interpolated is the bilinear one
    CHECK_THAT(interpolation_error(second, fields::bilinear, points, 2, 2), IsAtMost(1e-13));
    CHECK_THAT(interpolation_error(second, fields::bilinear, inside, 2, 2), IsAtMost(0.5));

    const Interpolator same(points.data(), view_of(transpose(affine, n, 2), 2, n), 2, false);
    std::vector<double> a(2), b(2);
    for (size_t p = 0; p < inside.size() / 2; p++){
        REQUIRE(first.interp(a.data(), &inside[2 * p]));
        REQUIRE(same.interp(b.data(), &inside[2 * p]));
        CHECK(a == b);
    }
}


// ============================================================================================
//                         One set of samples, as a grid and as points
// ============================================================================================

TEST_CASE("The same samples read as a regular grid and as scattered points", "[nd-interpolation][regular][scattered]"){
    const Rectilinear grid = custom_grid();
    const auto points = grid.points();
    const size_t n = grid.size();
    const auto inside = random_points(300, {-2, -1.5}, {2, 1.5}, 31);
    const auto read_both = [&](const Field& f, size_t nfields){
        const auto values = sample(f, points, 2, nfields);
        return std::make_pair(std::make_unique<Grid2>(view_of(values, n, nfields), grid.axes, true),
                              std::make_unique<Scattered2>(points.data(), view_of(values, n, nfields), 2, true));
    };

    SECTION("span the same domain"){
        const auto [regular, scattered] = read_both(fields::affine, 2);
        std::vector<std::vector<double>> interior;
        for (size_t p = 0; p < inside.size() / 2; p++){
            interior.push_back({inside[2 * p], inside[2 * p + 1]});
        }
        CHECK(count_contained(*regular, interior) == interior.size());
        CHECK(count_contained(*scattered, interior) == interior.size());
        CHECK(count_contained(*regular, outside_custom_grid) == 0);
        CHECK(count_contained(*scattered, outside_custom_grid) == 0);
        CHECK_THAT(abs_error(scattered->tri()->total_volume(), grid.volume()), IsAtMost(1e-12));
    }
    SECTION("reproduce the samples and every affine function"){
        const auto [regular, scattered] = read_both(fields::affine, 2);
        CHECK_THAT(interpolation_error(*regular, fields::affine, points, 2, 2), IsAtMost(1e-14));
        CHECK_THAT(interpolation_error(*scattered, fields::affine, points, 2, 2), IsAtMost(1e-14));
        CHECK_THAT(interpolation_error(*regular, fields::affine, inside, 2, 2), IsAtMost(1e-13));
        CHECK_THAT(interpolation_error(*scattered, fields::affine, inside, 2, 2), IsAtMost(1e-13));
    }
    SECTION("only the grid reproduces a bilinear function"){
        // At the center of a cell of sides hx, hy, either diagonal misses the xy term of
        // (1 + x)(2 - y) by exactly hx hy / 4
        const auto [regular, scattered] = read_both(fields::bilinear, 2);
        Magnitude regular_error{0}, scattered_deviation{0};
        for (size_t i = 0; i + 1 < grid.axes[0].size(); i++){
            for (size_t j = 0; j + 1 < grid.axes[1].size(); j++){
                CAPTURE(i, j);
                const double hx = grid.axes[0][i + 1] - grid.axes[0][i], hy = grid.axes[1][j + 1] - grid.axes[1][j];
                const double center[] = {grid.axes[0][i] + hx / 2, grid.axes[1][j] + hy / 2};
                double exact[2], got[2];
                fields::bilinear(exact, center);
                REQUIRE(regular->interp(got, center));
                regular_error = worst(abs_error(got[0], exact[0]), regular_error);
                REQUIRE(scattered->interp(got, center));
                scattered_deviation = worst(abs_error(std::abs(got[0] - exact[0]), hx * hy / 4), scattered_deviation);
            }
        }
        CHECK_THAT(regular_error, IsAtMost(1e-14));
        CHECK_THAT(scattered_deviation, IsAtMost(1e-14));
    }
    SECTION("approximate a smooth function to second order"){
        const auto probes = random_points(200, {-2, -1.5}, {2, 1.5}, 37);
        std::vector<double> regular_errors, scattered_errors;
        for (size_t m : {11, 21, 41}){
            const Rectilinear uniform{{linspace(-2, 2, m), linspace(-1.5, 1.5, m)}};
            const auto nodes = uniform.points();
            const auto values = sample(fields::smooth, nodes, 2, 1);
            const Grid2 regular(view_of(values, uniform.size(), 1), uniform.axes, true);
            const Scattered2 scattered(nodes.data(), view_of(values, uniform.size(), 1), 2, true);
            regular_errors.push_back(interpolation_error(regular, fields::smooth, probes, 2, 1).value);
            scattered_errors.push_back(interpolation_error(scattered, fields::smooth, probes, 2, 1).value);
        }
        for (size_t i = 0; i + 1 < regular_errors.size(); i++){
            CAPTURE(i, regular_errors, scattered_errors);
            CHECK(regular_errors[i] / regular_errors[i + 1] > 3.0);
            CHECK(scattered_errors[i] / scattered_errors[i + 1] > 3.0);
        }
    }
}

} // namespace
} // namespace odetest
