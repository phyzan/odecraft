#pragma once

/**
 * @file grids.hpp
 * @brief Sampled fields for the interpolation tests: rectilinear grids, as their axes and as the
 *        points they are made of, scattered points, and analytic fields to sample on either.
 *
 * Shared by both test executables, so it includes nothing of the library but ndspan.
 */

#include <ndspan/ndspan.hpp>

#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace odetest {

/// @brief n evenly spaced points over [a, b], both ends included exactly.
inline std::vector<double> linspace(double a, double b, size_t n){
    std::vector<double> x(n);
    for (size_t i = 0; i < n; i++){
        x[i] = (i + 1 == n) ? b : a + (b - a) * double(i) / double(n - 1);
    }
    return x;
}


/**
 * @brief A rectilinear grid, as its axes and as the points it is made of.
 *
 * Points are numbered the way the regular interpolators read their values: in C order over the
 * axes, the first axis varying slowest.
 */
struct Rectilinear {
    std::vector<std::vector<double>> axes;

    size_t ndim() const { return axes.size(); }

    /// @brief The number of points.
    size_t size() const {
        size_t n = 1;
        for (const auto& axis : axes){
            n *= axis.size();
        }
        return n;
    }

    /// @brief The index of point p along each axis.
    std::vector<size_t> index(size_t p) const {
        std::vector<size_t> idx(ndim());
        for (size_t a = ndim(); a-- > 0;){
            idx[a] = p % axes[a].size();
            p /= axes[a].size();
        }
        return idx;
    }

    /// @brief The coordinates of every point, as an array of shape (size, ndim).
    std::vector<double> points() const {
        std::vector<double> out;
        out.reserve(size() * ndim());
        for (size_t p = 0; p < size(); p++){
            const auto idx = index(p);
            for (size_t a = 0; a < ndim(); a++){
                out.push_back(axes[a][idx[a]]);
            }
        }
        return out;
    }

    /// @brief The volume of the box it spans.
    double volume() const {
        double v = 1;
        for (const auto& axis : axes){
            v *= axis.back() - axis.front();
        }
        return v;
    }
};

/// @brief The grid of the interpolation tests: unevenly spaced over [-2, 2] x [-1.5, 1.5].
inline Rectilinear custom_grid(){
    return {{{-2.0, -1.4, -0.9, -0.2, 0.3, 0.8, 1.5, 2.0}, {-1.5, -0.8, 0.1, 0.6, 1.2, 1.5}}};
}


/// @brief n points drawn uniformly from the box [lo, hi]: the same ones on every run and platform.
inline std::vector<double> random_points(size_t n, const std::vector<double>& lo, const std::vector<double>& hi, std::uint32_t seed = 2024){
    std::mt19937 gen(seed);
    std::vector<double> out(n * lo.size());
    for (size_t i = 0; i < out.size(); i++){
        const size_t axis = i % lo.size();
        const double u = (double(gen()) + 0.5) / 4294967296.0; // in (0, 1), without the platform's distributions
        out[i] = lo[axis] + (hi[axis] - lo[axis]) * u;
    }
    return out;
}


/// @brief A field of some number of values over some number of coordinates: f(out, x).
using Field = std::function<void(double*, const double*)>;

/// @brief f at each point of an array of shape (n_points, ndim), as an array of shape (n_points, nfields).
inline std::vector<double> sample(const Field& f, const std::vector<double>& points, size_t ndim, size_t nfields){
    const size_t n_points = points.size() / ndim;
    std::vector<double> out(n_points * nfields);
    for (size_t p = 0; p < n_points; p++){
        f(out.data() + p * nfields, points.data() + p * ndim);
    }
    return out;
}

/// @brief An array of shape (n_points, nfields) transposed to shape (nfields, n_points).
inline std::vector<double> transpose(const std::vector<double>& values, size_t n_points, size_t nfields){
    std::vector<double> out(values.size());
    for (size_t p = 0; p < n_points; p++){
        for (size_t k = 0; k < nfields; k++){
            out[k * n_points + p] = values[p * nfields + k];
        }
    }
    return out;
}

/// @brief A read-only view of values with the given shape, which is how the interpolators take them.
template<std::integral... Int>
ndspan::View<const double> view_of(const std::vector<double>& values, Int... shape){
    return ndspan::View<const double>(values.data(), shape...);
}


/// @brief Analytic fields over (x, y) or (x, y, z).
namespace fields {

/// @brief 1 + 2x - 3y and -1/2 + x + y/4: interpolated exactly on any mesh.
inline void affine(double* out, const double* x){
    out[0] = 1 + 2 * x[0] - 3 * x[1];
    out[1] = -0.5 + x[0] + 0.25 * x[1];
}

/// @brief (1 + x)(2 - y) and 3 - x + xy/2: interpolated exactly on rectilinear grids only.
inline void bilinear(double* out, const double* x){
    out[0] = (1 + x[0]) * (2 - x[1]);
    out[1] = 3 - x[0] + 0.5 * x[0] * x[1];
}

/// @brief sin(x) cos(y), interpolated to second order.
inline void smooth(double* out, const double* x){
    out[0] = std::sin(x[0]) * std::cos(x[1]);
}

/// @brief 1 + x - 2y + 3z and (1 + x)(2 - y)(1/2 + z), over three coordinates.
inline void trilinear(double* out, const double* x){
    out[0] = 1 + x[0] - 2 * x[1] + 3 * x[2];
    out[1] = (1 + x[0]) * (2 - x[1]) * (0.5 + x[2]);
}

/// @brief (-y, x): a rigid rotation, whose streamlines are circles about the origin, run at
/// angular speed 1. Linear, so interpolated exactly on any mesh.
inline void rotation(double* out, const double* x){
    out[0] = -x[1];
    out[1] = x[0];
}

} // namespace fields

} // namespace odetest
