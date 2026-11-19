# odecraft tests

A [Catch2](https://github.com/catchorg/Catch2) v3 suite. Every check compares against a known
answer, so a run passes or fails by itself, and CTest registers each test case separately.

## Building and running

The suite needs the library's dependencies. Catch2 comes from an installed package if CMake finds one, and is otherwise downloaded at configure time (a pinned release, checked by hash).

The tests build as part of the library, from the repository root, in a build directory of their
own. Their executables land in `build-tests/tests/`:

```sh
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release -DODECRAFT_BUILD_TESTS=ON
cmake --build build-tests -j 4
ctest --test-dir build-tests --output-on-failure -j 4
```

You can explicitly specify a compiler at configure time to cover more ground, in a directory of its own (CMake keeps the compiler a directory was first configured with):
```sh
CXX=clang++ cmake -S . -B build-tests-clang -DCMAKE_BUILD_TYPE=Release -DODECRAFT_BUILD_TESTS=ON
```

Build options:

| Option | Effect |
|---|---|
| `ODECRAFT_TESTS_SANITIZE` | Builds with AddressSanitizer and UndefinedBehaviorSanitizer. |
| `ODECRAFT_TESTS_WERROR` | Turns warnings into errors. |

Assertions stay enabled in the header-only tests whatever the build type.

A build makes two executables. CTest labels their tests `odecraft` and `crafted`:

| Executable | What it tests |
|---|---|
| `odecraft_tests` | The header-only interface, namespace `ode`. |
| `odecraft_crafted_tests` | The pre-compiled interface, `ode::crafted`. It includes only `<odecraft/Crafted.hpp>`, so an instantiation the library fails to ship shows up as a link error. |

Run part of the suite by label or name, or through the executable with Catch2 tags:

```sh
ctest --test-dir build-tests -L crafted
ctest --test-dir build-tests -R "Lyapunov"
./build-tests/tests/odecraft_tests "[events]"          # all tests tagged [events]
./build-tests/tests/odecraft_tests "[chaos]" -s        # with every passing check printed too
./build-tests/tests/odecraft_tests --list-tests
```

On GitHub, [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) builds and runs the
suite this way on every push and pull request, with every major version of GCC from 11 and of
Clang from 13.


## Layout

| File | Covers |
|---|---|
| `odecraft/test_steppers.cpp` | Every stepper against exact solutions, forward and backward: orders of convergence, tolerance proportionality, full-precision steps, stiffness |
| `odecraft/test_precision.cpp` | Extended precision: long double, mpreal, LazyType |
| `odecraft/test_state_descriptors.cpp` | State descriptors: runtime- and static-sized states, tensor layouts, Jacobian linear algebra |
| `odecraft/test_solver_lifecycle.cpp` | Construction, `advance*`, observers, checkpoints, `SetICs`, `Reset`, `clone`, interpolators |
| `odecraft/test_solver_failure.cpp` | Stopped solvers: their collapsed states, no interpolation, no advancing |
| `odecraft/test_jacobians.cpp` | Autodiff, exact and finite-difference Jacobians, with explicit or automatic steps |
| `odecraft/test_runtime_events.cpp` | Precise and periodic events: directions, tolerances including 0, several per step, masks, delays |
| `odecraft/test_static_events.cpp` | The same for static objectives |
| `odecraft/test_dense_output.cpp` | `interpolate_until`, across masked jumps and when the solver dies midway |
| `odecraft/test_factories.cpp`, `test_catalogs.cpp` | `getSolver`, the boxing factories, stepper catalogs, the virtual interface |
| `odecraft/test_ode_driver.cpp` | The `ODE` driver of `OdeInt.hpp`: integration routines, `t_eval`, events and their options, dense solutions, `clone`, `reset`, `clear` |
| `odecraft/test_chaos.cpp`, `test_chaos_jacobians.cpp` | Variational solvers: exact Lyapunov exponents, the main and full-system Jacobians, every integration routine, the variational driver |
| `odecraft/test_toolkit.cpp`, `test_univariate_interpolation.cpp` | Numerical tools, and the dense-output interpolators |
| `odecraft/test_nd_interpolation.cpp`, `test_vector_fields.cpp` | Regular and scattered N-dimensional interpolation, Delaunay triangulations, vector fields and streamlines |
| `crafted/*.cpp` | The whole `ode::crafted` interface, over the scalar types it ships |
| `support/` | Shared helpers: scalar types and error matchers, test problems with exact solutions, stepper settings, sampled grids |

## Writing tests

- **Scalar types.** A test that involves a scalar type runs over all five of them, with
  `TEMPLATE_TEST_CASE(..., ODECRAFT_TEST_SCALARS)`: float, double, long double, `mpfr::mpreal`
  and `lazex::LazyType<mpfr::mpreal>`. Crafted tests use `ODECRAFT_CRAFTED_SCALARS`, the four
  the library ships.
- **Where a test goes.** The sources listed in `ODECRAFT_TESTS_PER_SCALAR` are compiled once
  per scalar type, which keeps each compiler process within memory, so they may only hold test
  cases over `ODECRAFT_TEST_SCALARS`. Plain `TEST_CASE`s, and templates over other type lists,
  go in the sources of `ODECRAFT_TESTS_ONCE`. Otherwise they would be registered five times,
  and Catch2 refuses to start.
- **Comparisons.** Compute errors in the scalar type and compare them as a `Magnitude`, which
  prints in scientific notation: `CHECK_THAT(max_error<SD>(q, exact), IsAtMost(1e-9))`. A NaN
  never passes.
- **Bounds** come from the stepper's `Settings` (`support/settings.hpp`) or from the analysis of
  the method. They are not fitted to a run.
- **Loops over steppers or state kinds** run at compile time with one `DYNAMIC_SECTION` each,
  so a failure names its case.
- **Names.** Test names must not contain commas, which Catch2's command line reads as
  separators.
- **Lambdas** that return a scalar declare `-> T`. Otherwise a `LazyType` expression template
  dangles.
- **Crashes.** A check that may crash the process goes in a test case of its own, so that it
  takes nothing else down with it.
