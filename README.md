<p align="center">
<img src="https://img.shields.io/badge/C%2B%2B20-blue?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++20">
<img src="https://img.shields.io/badge/Header_Only-green?style=for-the-badge" alt="Header Only">
</p>

<h1 align="center">OdeCraft</h1>

<p align="center">
  <strong>A Modern C++ Library for Ordinary Differential Equations</strong>
</p>

<p align="center">
  High-performance, templated ODE solvers with flexible event detection mechanisms
</p>

---

# OdeCraft

**OdeCraft** is a modern, object-oriented C++ library for solving **Ordinary Differential Equations (ODEs)**.

The library implements standard integration algorithms, but utilizes a **template-based design** to allow for arbitrary precision, automatic differentiation, and lazy evaluation,
while offering a clean interface for defining and integrating ODE systems.

It also provides a flexible event detection system for runtime or compile-time event handling.

# Getting Started

**OdeCraft** can be cloned using:
```bash
git clone --recursive https://github.com/phyzan/odecraft.git
```
which will also pull all of its submodule dependencies. These can be updated via
```bash
git submodule update --init --recursive
```

## Requirements
- C++20 compatible compiler: GCC or Clang.
- **qhull**, for Delaunay triangulation (scattered-data interpolation).
- **mpfr** and **gmp** (optional). Their `C++` wrapper [mpreal](https://github.com/advanpix/mpreal) is bundled as a submodule.

All three come as prebuilt packages:

| Platform | Command |
|---|---|
| Debian / Ubuntu | `sudo apt install libqhull-dev libmpfr-dev libgmp-dev` |
| Fedora | `sudo dnf install qhull-devel mpfr-devel gmp-devel` |
| Arch | `sudo pacman -S qhull mpfr gmp` |
| macOS (Homebrew) | `brew install qhull mpfr gmp` |
| Windows (MSYS2, UCRT64 shell) | `pacman -S mingw-w64-ucrt-x86_64-qhull mingw-w64-ucrt-x86_64-mpfr mingw-w64-ucrt-x86_64-gmp` |

# Linking headers via CMake

In an external project, use

```cmake
add_subdirectory(path/to/odecraft)
target_link_libraries(your_target PRIVATE odecraft::odecraft)
```
This gives you `<odecraft/...>`, `<xdiff/...>` etc. includes, the required C++20 standard, and [options](#CMake-options) (toggle with e.g. `-D<OPTION>=ON`)

# CMake options

CMake options that toggle preprocessor macros across the library and its bundled dependencies:

| CMake Option | Effect |
|--------------|-------|
| `ODECRAFT_RK4_DENSE` | Enable accurate RK4 dense output for the `RK4` solver, but for large ODE systems this can be expensive. |
| `ODECRAFT_NO_WARN` | Disable ODE solver console warnings. |
| `ODECRAFT_NO_NAN_CHECK` | Disable NaN/inf checks on solver output, for performance. |
| `DEBUG` | Debug build: `-O0 -g3 -ggdb3 -fno-omit-frame-pointer -UNDEBUG` (asserts enabled), instead of the default optimized release build (`-O3 -DNDEBUG`, LTO where supported). Also triggered by `-DCMAKE_BUILD_TYPE=Debug`. |
| `ODECRAFT_BUILD_CRAFTED` | Build the API for the `ode::crafted` namespace. Defaults to `ON` when configuring odecraft directly, `OFF` when pulled in via `add_subdirectory` by another project. |
| `ODECRAFT_RHS_THROWS` | Allow `Rhs` and `Jac` to throw: the exception then propagates out of the stepping call. **Off by default**. See the [`OdeType`](#odetype-template-parameter) section. |
| `ODECRAFT_USE_LAZY_MPREAL` | Compile `ode::crafted`'s `mpreal_t` as `lazex::LazyType<mpfr::mpreal>` instead of plain `mpfr::mpreal`. The lazy wrapper elides temporaries in compound expressions, a large win for MPFR where every temporary is a heap allocation. **On by default**; turn it off for plain `mpfr::mpreal`. Either way the type is spelled `mpreal_t`, so nothing else in your code changes. |

See useful [options](https://github.com/phyzan/lazex#cmake-options) for the `lazex` submodule.

## Calling math functions: `ODECRAFT_USING_MATH`

Put `ODECRAFT_USING_MATH;` in every namespace of yours that calls `abs()`, `sqrt()`, `pow()` or other math functions **unqualified**.

One line per namespace, or once at global scope:

```cpp
#include <odecraft/odecraft.hpp>

ODECRAFT_USING_MATH

using T = double;

struct MyOdeSystem{

    // Lorenz system
    // Written generic with `auto` to allow for automatic differentiation
    T sigma = 10.0;
    T rho = 28.0;
    T beta = 8.0/3.0;

    inline void Rhs(auto& dq_dt, const auto& /*t*/, const auto& q) const {
        //3D lorenz system, args = {sigma, rho, beta}
        dq_dt[0] = sigma*(q[1] - q[0]);
        dq_dt[1] = q[0]*(rho - q[2]) - q[1];
        dq_dt[2] = q[0]*q[1] - beta*q[2];
    }
};


int main(){

    using State = ode::common::State<T, 3>; // Simplest state-vector for 3D systems



    State y0 = {1.0, 1.0, 1.0};
    State rtol = {1e-9, 1e-9, 1e-9};
    State atol = {1e-12, 1e-12, 1e-12};

    T min_step{0}; // no minimum
    T max_step{0}; // no maximum
    T first_step{0}; // first step
    int dir = 1; // integration direction

    {
        ode::BoxedEvent<T, State> event = ode::make_precise_event<T, State>(
            "MyEvent",
            [&](T t, const State& q){
                return q[1]*sin(t) - 2; // event when y*sin(t) = 2
            },
            T{1e-10} // detection precision
        );

        ode::EventList<T, State> event_list = ode::make_event_list<T, State>(std::move(event));

        auto solver = ode::common::make_rich_vsolver(
            ode::Stepper::RK45,
            MyOdeSystem{},
            T{0.0},
            y0,
            rtol, atol,
            min_step, max_step, first_step,
            dir,
            std::move(event_list));

        solver->do_advance_to_event(10000, {"MyEvent"});

        solver->show_state(std::cout, 12);

        // fill `v` with the current state vector
        State v;
        solver->get_current_vector(v);

        std::cout << std::setprecision(12) << "Event value (should be close to 2): "
        << (v[1] * sin(solver->get_time())) << std::endl;
    }

    /*
    Integrate the augmented system of equations which doubles the system size,
    to approximate the Lyapunov exponent (~0.906)
    */

    {
        using FullState = ode::chaos::common::VarState<T, 3>;

        FullState full_q0{.main = y0, .variation = {1, 0, 0}};
        FullState full_rtol{.main = rtol, .variation = rtol};
        FullState full_atol{.main = atol, .variation = atol};

        auto solver = ode::chaos::common::make_variational_solver<ode::UtilPolicy::Virtual, T>(
            ode::Stepper::RK45,
            MyOdeSystem{},
            T{0.0},
            full_q0,
            T{0.1},
            full_rtol, full_atol,
            min_step, max_step, first_step,
            dir
        );

        solver->do_advance_until(10000);

        std::cout << "\n------ Integrating augmented system ------ \n\n";
        std::cout << "Lyapunov exponent: " << solver->get_lyapunov_exponent() << "\n";
        std::cout << "Steps required: " << solver->get_step_count() << std::endl;
    }

    return 0;
}```
