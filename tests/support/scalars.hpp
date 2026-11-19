#pragma once

/**
 * @file scalars.hpp
 * @brief The scalar types every test runs over, and what the tests need to know about them.
 *
 * Shared by both test executables, so it includes nothing that instantiates a solver.
 */

#include <odecraft/Toolkit/Tools.hpp>
#include <lazex/apps/lazex_mpreal.hpp>

#include <catch2/catch_tostring.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>


/**
 * @brief The scalar types of the tests, for TEMPLATE_TEST_CASE.
 *
 * All five, except in the sources the build compiles once per scalar type (see CMakeLists.txt):
 * those are compiled with ODECRAFT_TEST_SCALAR_INDEX set, and run over that one type only.
 */
#if !defined(ODECRAFT_TEST_SCALAR_INDEX)
#define ODECRAFT_TEST_SCALARS float, double, long double, mpfr::mpreal, lazex::LazyType<mpfr::mpreal>
#elif ODECRAFT_TEST_SCALAR_INDEX == 0
#define ODECRAFT_TEST_SCALARS float
#elif ODECRAFT_TEST_SCALAR_INDEX == 1
#define ODECRAFT_TEST_SCALARS double
#elif ODECRAFT_TEST_SCALAR_INDEX == 2
#define ODECRAFT_TEST_SCALARS long double
#elif ODECRAFT_TEST_SCALAR_INDEX == 3
#define ODECRAFT_TEST_SCALARS mpfr::mpreal
#elif ODECRAFT_TEST_SCALAR_INDEX == 4
#define ODECRAFT_TEST_SCALARS lazex::LazyType<mpfr::mpreal>
#else
#error "ODECRAFT_TEST_SCALAR_INDEX must be one of 0, 1, 2, 3, 4"
#endif


namespace odetest {

// Unqualified math (abs, cos, exp, ...) resolves to std for built-in types, and to mpfr, lazex
// or xdiff through argument-dependent lookup. Without it, a long double would silently go
// through ::cos(double).
ODECRAFT_USING_MATH;

using mpreal = mpfr::mpreal;
using lazy_mpreal = lazex::LazyType<mpfr::mpreal>;

/// @brief The precision of mpfr::mpreal throughout the tests, set before the first test case runs.
inline constexpr mpfr_prec_t mpreal_bits = 128;

template<typename T>
inline constexpr bool is_multiprecision = std::is_base_of_v<mpreal, T>;


/// @brief A magnitude as a double, which is all an error or a tolerance needs.
template<typename T>
double to_double(const T& x){
    if constexpr (std::is_arithmetic_v<T>){
        return static_cast<double>(x);
    } else {
        return x.toDouble();
    }
}

/// @brief Machine epsilon of T, at the current precision for the multiprecision types.
template<typename T>
double epsilon(){
    return to_double(T(std::numeric_limits<T>::epsilon()));
}

/// @brief x in scientific notation, which is how an error or a tolerance reads best.
inline std::string scientific(double x){
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3e", x);
    return buffer;
}


/**
 * @brief The size of an error, as checks compare it with IsAtMost.
 *
 * A type of its own only for how a failed check prints it: Catch2 writes a double in fixed
 * notation, so an error of 1e-21 would read as 0.0.
 */
struct Magnitude {
    double value;
};

/// @brief The larger of two errors, where NaN wins: a NaN anywhere must fail the comparison it feeds.
inline Magnitude worst(Magnitude a, Magnitude b){
    return (std::isnan(a.value) || a.value > b.value) ? a : b;
}

/// @brief Matches a Magnitude no larger than a bound. NaN never matches.
class IsAtMostMatcher final : public Catch::Matchers::MatcherBase<Magnitude> {
public:
    explicit IsAtMostMatcher(double bound) : bound_(bound) {}

    bool match(const Magnitude& m) const override { return m.value <= bound_; }

    std::string describe() const override { return "is at most " + scientific(bound_); }

private:
    double bound_;
};

inline IsAtMostMatcher IsAtMost(double bound){
    return IsAtMostMatcher(bound);
}

/// @brief |actual - expected|, computed in T before it is narrowed to a double.
template<typename T>
Magnitude abs_error(const T& actual, const T& expected){
    const T diff = actual - expected;
    const T err = abs(diff);
    return {to_double(err)};
}

/// @brief |actual - expected| / |expected|, or the absolute error when expected is zero.
template<typename T>
Magnitude rel_error(const T& actual, const T& expected){
    const double scale = to_double(T(abs(expected)));
    const Magnitude err = abs_error(actual, expected);
    return {scale == 0 ? err.value : err.value / scale};
}

/// @brief A scalar written out to every digit it carries.
template<typename T>
std::string to_string(const T& x){
    std::ostringstream out;
    if constexpr (is_multiprecision<T>){
        out << static_cast<const mpreal&>(x).toString();
    } else {
        out.precision(std::numeric_limits<T>::max_digits10);
        out << x;
    }
    return out.str();
}

} // namespace odetest


// Catch2 prints float and double itself, these three it would print at 6 digits or not at all
namespace Catch {

template<>
struct StringMaker<long double> {
    static std::string convert(long double x){ return odetest::to_string(x); }
};

template<>
struct StringMaker<mpfr::mpreal> {
    static std::string convert(const mpfr::mpreal& x){ return odetest::to_string(x); }
};

template<>
struct StringMaker<lazex::LazyType<mpfr::mpreal>> {
    static std::string convert(const lazex::LazyType<mpfr::mpreal>& x){ return odetest::to_string(x); }
};

template<>
struct StringMaker<odetest::Magnitude> {
    static std::string convert(const odetest::Magnitude& m){ return odetest::scientific(m.value); }
};

} // namespace Catch
