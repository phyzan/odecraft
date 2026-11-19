#pragma once

/**
 * @file odecraft_test.hpp
 * @brief The one include of the tests of the header-only interface.
 *
 * Conventions:
 *   - Every test case runs over all five scalar types (ODECRAFT_TEST_SCALARS), unless what it
 *     tests has no scalar type.
 *   - Tests that loop over steppers or state kinds do so at compile time, with one
 *     DYNAMIC_SECTION each, so that a failure names the stepper and the state kind.
 *   - Errors are computed in the scalar type, and compared as Magnitudes, which print in
 *     scientific notation: CHECK_THAT(max_error<SD>(q, exact), IsAtMost(1e-9)).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <odecraft/odecraft.hpp>

#include "scalars.hpp" // IWYU pragma: keep
#include "states.hpp" // IWYU pragma: keep
#include "problems.hpp" // IWYU pragma: keep
#include "steppers.hpp" // IWYU pragma: keep
#include "checks.hpp" // IWYU pragma: keep
