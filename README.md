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

## Requirements
- C++20 compatible compiler

For the optional arbitrary precision support, install the prebuilt **mpfr** and **gmp** libraries:
```bash
sudo apt install libmpfr-dev libgmp-dev
```

**OdeCraft** can be cloned using:
```bash
git clone --recursive https://github.com/phyzan/odecraft.git
```
which will also pull all of its submodule dependencies. If you have already cloned the repository without the `--recursive` flag, you can initialize the submodules with:
```bash
git submodule update --init --recursive
```

Submodules can be updated via
```bash
git submodule update --recursive
```

## Building **OdeCraft**

Optionally, you can build the project from source using CMake, which compiles specific template instantiations of the `ode` namespace, and provides all compiled interface in the `ode::crafted` namespace. This can substantially reduce compile times for external projects, but since it does that by using type-erasing wrappers like `std::function`, this comes at a small runtime performance cost.

For maximum performance, use lambdas or functors where possible, without building the project.

### Building the library

Configure the project with some [options](#CMake-options) and build it:
```bash
cmake -S . -B build \
  -DXDIFF_LAZY_NESTED_DUAL=ON \
  -DLAZY_MPFR_RND=MPFR_RNDN \
  -DXDIFF_FAST=ON \
  -DXDIFF_LEIBNIZ_OPT=OFF \
  -DXDIFF_SCALAR_OPTIMIZATIONS=ON \
  -DODECRAFT_RK4_DENSE=OFF \
  -DODECRAFT_NO_WARN=ON \
  -DODECRAFT_NO_NAN_CHECK=ON \
  -DODECRAFT_USE_FLAT_AUTODIFF=OFF \
  -DODECRAFT_USE_LAZY_MPREAL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/libodecraft_crafted.a`. Similarly, the tests and tutorials can be configured from their own directories:

```bash
cd tests     && cmake -S . -B build && cmake --build build -j
cd tutorials && cmake -S . -B build && cmake --build build -j
```

Each pulls **OdeCraft** in as a subproject, so there is nothing to install or configure first. Add
`-DDEBUG=ON` to either, for an `-O0` build with asserts (and AddressSanitizer for the tests).

### Linking the compiled interface

In an external project, request the build and link `odecraft::crafted` on top of the headers:

```cmake
set(ODECRAFT_BUILD_CRAFTED ON CACHE BOOL "" FORCE)
add_subdirectory(path/to/odecraft)

target_link_libraries(your_target PRIVATE odecraft::crafted)
```

`odecraft::crafted` links `odecraft::odecraft` publicly, so `<odecraft/Compiled/odecraft.hpp>` is
all you need to include.

Guard the link with `if(TARGET odecraft::crafted)` if the build is conditional in your project.

Note that `ODECRAFT_BUILD_CRAFTED` defaults to `OFF` for consumers, so a project that only wants
the header-only `ode` interface pays nothing for a library it never links.

## Linking headers via CMake

In an external project, use

```cmake
add_subdirectory(path/to/odecraft)
target_link_libraries(your_target PRIVATE odecraft::odecraft)
```
This gives you `<odecraft/...>`, `<xdiff/...>` etc. includes, the required C++20 standard, and [options](#CMake-options) (toggle with e.g. `-D<OPTION>=ON`)

## CMake options

CMake options that toggle preprocessor macros across the library and its bundled dependencies:

| CMake Option | Effect |
|--------------|-------|
| `ODECRAFT_RK4_DENSE` | Enable accurate RK4 dense output for the `RK4` solver, but for large ODE systems this can be expensive. |
| `ODECRAFT_NO_WARN` | Disable ODE solver console warnings. |
| `ODECRAFT_NO_NAN_CHECK` | Disable NaN/inf checks on solver output, for performance. |
| `ODECRAFT_USE_FLAT_AUTODIFF` | Store automatic-differentiation duals in `xdiff`'s flat layout (a single contiguous array) for systems whose size `N` is known at compile time. Dynamically sized systems (`N == 0`) stay on the nested layout either way, since the flat one needs its variable count at compile time. Off by default, so every system uses the nested layout. |
| `DEBUG` | Debug build: `-O0 -g3 -ggdb3 -fno-omit-frame-pointer -UNDEBUG` (asserts enabled), instead of the default optimized release build (`-O3 -DNDEBUG`, LTO where supported). Also triggered by `-DCMAKE_BUILD_TYPE=Debug`. |
| `ODECRAFT_BUILD_CRAFTED` | Build the API for the `ode::crafted` namespace. Defaults to `ON` when configuring odecraft directly, `OFF` when pulled in via `add_subdirectory` by another project. |
| `ODECRAFT_USE_LAZY_MPREAL` | Compile `ode::crafted`'s `mpreal_t` as `lazy::LazyType<mpfr::mpreal>` instead of plain `mpfr::mpreal`. The lazy wrapper elides temporaries in compound expressions, a large win for MPFR where every temporary is a heap allocation. **On by default**; turn it off for plain `mpfr::mpreal`. Either way the type is spelled `mpreal_t`, so nothing else in your code changes. |

See useful [options](https://github.com/phyzan/xdiff#macros) for the `xdiff` submodule.


---

# Features:

- **Header-only**: No compilation needed, just include and use
- **Event system**: Detect and respond to user-defined conditions during integration
- **Dense output**: Smooth interpolation between computed steps
- **Memory efficient**: Solvers preallocate memory for zero heap (de)allocations between steps
- **Flexible solver policies**: Choose between static, rich, virtual, and rich-virtual solvers for performance vs. flexibility trade-offs
- **Extensible**: Easily add new solvers or event types
- **Template-based**: Allows for any numeric type including arbitrary precision (MPFR), supports automatic differentiation via [XDiff](https://github.com/phyzan/xdiff), lazy evaluation via [lazy](https://github.com/phyzan/lazy), and more
- **Dynamical systems analysis**: Built-in support for variational equations and Lyapunov exponent calculations

##

The `ode::crafted` namespace provides headers linked to the compiled project. The template parameter `T` found in all the classes and functions in that namespace has been compiled for `float`, `double` and `long double`.

Everything in it is the `ode` interface with every template parameter but `T` pinned down: the
system size is dynamic (`N = 0`), every callable is type-erased, and every solver is
`SolverPolicy::RichVirtual`. There is no policy to choose — a rich solver already *is* an
`OdeSolver`, so if you do not need events, pass none and hold the result by `OdeSolver<T>`.

Include `<odecraft/Compiled/odecraft.hpp>` for all of it, or a single header for one piece.

**Vocabulary** — `Compiled/Toolkit.hpp`

| Type | Signature |
|------|-----------|
| `rhs_t<T>` | `void(T* out, const T& t, const T* q)` |
| `objfun_t<T>` | `T(const T& t, const T* q)` |
| `observer_t<T>` | `bool(const T& t, const T* q, const T* t_ptr)` |
| `interp_t<T>` | `void(T* out, const T& t)` |
| `ode_t<T>` | `OdeData<rhs_t<T>, rhs_t<T>>` — the one system type the interface accepts |
| `mpreal_t` | `lazy::LazyType<mpfr::mpreal>`, or plain `mpfr::mpreal` with `ODECRAFT_USE_LAZY_MPREAL=OFF` — the fourth compiled scalar |

`ode_t<T>`'s Jacobian is optional: leave it null and the implicit steppers fall back to finite
differences. Supply one where you can — `BDF` and the variational solvers benefit most, and with a
type-erased Rhs there is no autodiff to fall back on.

**Arbitrary precision.** `mpreal_t` is compiled alongside `float`, `double` and `long double`.

| Function | Purpose |
|----------|---------|
| `set_mpreal_prec(prec)` | Set the working precision in bits. Dispatches to `lazy::set_default_mpreal_prec` or `mpfr::mpreal::set_default_prec` depending on the backend. |
| `get_default_prec()` | The precision that new `mpreal_t` values are created with. Named after, and forwarding to, `mpfr::mpreal::get_default_prec`. |

Call `set_mpreal_prec` **before** constructing anything on `mpreal_t`: MPFR fixes an object's
precision at construction, so raising it afterwards leaves existing values behind. Compile the tutorials and run the `MPRealCrafted` [example](tutorials/MPRealCrafted.cpp) to see how to use `mpreal_t` in a solver, using the compiled interface.

**Warning**: `set_mpreal_prec` only affects the current thread (as the `mpfr` library does).


**Solvers and stepping** — `Compiled/SolverBase.hpp`, `Compiled/Steppers.hpp`

| Name | Purpose |
|------|---------|
| `OdeSolver<T>`, `OdeRichSolver<T>` | Abstract interfaces; the latter adds event detection |
| `BoxedRichSolver<T>`, `BoxedInterp<T>` | Owning handles |
| `make_rich_vsolver<T>(method, ode, t0, q0, rtol, atol, ...)` | Build a solver of any method |
| `Euler<T>`, `RK4<T>`, `RK23<T>`, `RK45<T>`, `DOP853<T>`, `BDF<T>` | The concrete steppers, if you want one by name |

**Integration front end** — `Compiled/ODE.hpp`, `Compiled/OdeHistory.hpp`

| Name | Purpose |
|------|---------|
| `ODE<T>` | Owns a solver and records its trajectory |
| `OdeResult<T>` | Sampled points, event hits and status of a finished run |
| `OdeSolution<T>` | An `OdeResult` that also answers `operator()(t)` anywhere in range |

**Events** — `Compiled/Events.hpp`

| Name | Purpose |
|------|---------|
| `make_precise_event<T>(name, obj_fun, tol, dir, mask, delay_mask)` | Fires where `obj_fun` crosses zero |
| `make_periodic_event<T>(name, period, mask, delay_mask)` | Fires every `period` time units |
| `PreciseEvent<T>`, `PeriodicEvent<T>`, `EventList<T>`, `EventOptions` | The underlying types |

**Dense output** — `Compiled/Interpolators.hpp`: `Interpolator<T>`, `LocalInterpolator<T>`,
`LinkedInterpolator<T>`, `InterpObj<T>`.

**Chaos** — `Compiled/Chaos.hpp`

| Name | Purpose |
|------|---------|
| `make_variational_solver<T>(method, ode, t0, q0, delta_q0, period, ...)` | Solver for the augmented system |
| `ChaoticSolver<T>` | Abstract variational solver; reports Lyapunov data |
| `VariationalODE<T>` | Driver that also records renormalisation times and Lyapunov values |

---

# Integrators

The library focuses on integrating systems of ODEs through objects that once instantiated, preallocate memory for any process that might be encountered,
and can be advanced in time while only updating their state in-place, without storing any integration history (if not requested).
This is particularly useful for long-running simulations, where memory usage and performance are critical.

Essentially, a solver object *iterates* over the solution of a system with a predefined accuracy, providing explicit
control over the integration process, and allowing for event detection and interpolation between steps (dense output).

This is achieved by defining a common interface for all solvers through the `BaseSolver` class, which is then specialized for any integration algorithm:

```cpp
template<typename Derived, typename T, size_t N, ode::SolverPolicy SP, ode::hasRhsFunc<T> OdeType>
class BaseSolver;
```

where:
- `Derived` is the derived solver class (CRTP)
- `T` is the numeric type (e.g., `double`, `float`, `mpfr::mpreal`)
- `N` is the number of equations in the system (`N > 0` for size known at compile-time, `N=0` for dynamic size using heap allocation)
- `SP` is the solver policy (see below)
- `OdeType` is the type of the ODE function (must satisfy `hasRhsFunc<T>` concept)

## SolverPolicy (SP) template parameter

The `SolverPolicy` template parameter controls inheritance and feature availability:

| Policy | Virtual | Events | Use Case |
|--------|---------|--------|---------------|
| `Static` | No | No | Maximum performance, compile-time type |
| `RichStatic` | No | Yes | Events needed, type known at compile-time |
| `Virtual` | Yes | No | Runtime solver selection, no events |
| `RichVirtual` | Yes | Yes | Full flexibility at runtime |


## OdeType template parameter

The `OdeType` template parameter must satisfy the `hasRhsFunc<T>` concept, which requires the ODE type to expose an `Rhs` member computing dq/dt:

```cpp
void Rhs(T* out, const T& t, const T* q);
```

- `out` — output array, receives dq/dt
- `t` — independent variable (time)
- `q` — current state array

An analytic Jacobian can optionally be provided via the `N x N` Jacobian matrix function
```cpp
void Jac(T* jac_mat, const T& t, const T* q);
```
which satisfies the `hasJacFunc<T>` concept, and is filled in column-major order as
```cpp
jac_mat[i + j*system_size] = df_i/dx_j
```

## Automatic differentiation

If no analytic `Jac` is given but `Rhs` is written generically enough (templated), ideally as
```cpp
void Rhs(auto* out, const auto& t, auto q);
```
to also run over `xdiff::Dual` numbers (checked via `supportsDualRhs`), the library seeds the state with dual numbers and differentiates `Rhs` itself to obtain an exact Jacobian at no extra coding cost — no finite-difference approximation needed.

Jacobian source is chosen automatically in the following order of preference: exact analytic `Jac` > autodiff via `xdiff` > finite-difference approximation.

This means that if both an `Rhs` that supports `Dual`s and an analytic `Jac` are provided, the solver will prefer the analytic `Jac` over automatic differentiation. However it is important to ensure that the analytic `Jac` is correctly implemented and in column-major order, as an incorrect Jacobian can lead to inaccurate results or solver instability. Define the Rhs generically with `auto` parameters as shown earlier before providing an analytic `Jac`, and if the latter is correctly implemented and faster, provide it to the solver.

For clarity, the appropriate generic overload for `Rhs` that supports automatic differentiation is:
```cpp
template<typename T, size_t N, size_t Order>
void Rhs(DualType<T, N, Order>* out, const T& t, SeedVec<T, N, Order> q);
```
where
- `out` — output array of dual numbers, receives dq/dt
- `t` — independent variable (time)
- `q` — a seed vector of dual numbers representing the current state.
  This is a special class that wraps over a `const T* rhs`, and `q[i]` returns an
  `xdiff::Seed<T, N, Order>` object, that contains the value and its derivative is `1` for the `i`-th variable and `0` for all other components.

However this signature does not need to be used, the generic `Rhs` with `auto` parameters as shown earlier is sufficient for automatic differentiation, and also works as the standard right-hand side function for the solvers.

See the `Autodiff` [example](tutorials/Autodiff.cpp) for a practical demonstration of automatic differentiation in action, and how well it compares with using an exact Jacobian or finite-difference approximation, performance-wise.

---

The `BaseSolver` class provides a common interface for all solvers, with the following main methods:

```cpp
// Accessors
void                Rhs(T* out, const T& t, const T* q) const; // Compute the right-hand side of the ODE system
void                Jac(T* out, const T& t, const T* q) const; // Compute the Jacobian of the ODE system (optional)
const T&            t() const; // Get the current time
View1D<T, N>        vector() const; // Get the current state vector
State<T>            ics() const; // Get the initial conditions
bool                is_running() const; // Check if the solver can still advance (e.g. has no NaNs/Infs)
bool                is_dead() const; // Returns !is_running()
bool                diverges() const; // Check if the solver has diverged (NaN/Inf detected)
void                interp(T* out, const T& t) const; // Interpolate the solution at a given time within the old and new step interval
const std::string&  status() const; // Get the solver's status message

// Modifiers
bool                advance(); // Advance the solver by one step (automatic step size control)
bool                advance_until(const T& time); // Advance the solver until a specified time is reached
bool                advance_until(const T& time, Callable&& observer); // Advance the solver until a specified time is reached, calling an observer function at each step
bool                set_ics(T t0, const T* y0, T stepsize, int direction); // Set new initial conditions and reset the solver in-place (no memory reallocation happens)
void                Reset(); // Reset the solver to its initial state
BoxedInterp<T, N>   interpolate_until(const T& time, Callable&& observer); // Advance the solver until a specified time is reached, returning an interpolator over the integration interval
```

### Available methods

Currently, the following solver classes are provided, overriding the proper `BaseSolver` methods for their respective algorithms:

| Solver | Type | Order | Description |
|--------|------|-------|-------------|
| `Euler` | Explicit | 1 | Basic Euler method |
| `RK23` | Explicit | 2/3 | Runge-Kutta 2(3) with adaptive stepping |
| `RK45` | Explicit | 4/5 | Dormand-Prince method (recommended for most problems) |
| `DOP853` | Explicit | 8 | High-order method with excellent dense output |
| `BDF` | Implicit | 1-5 | Backward Differentiation Formula for stiff problems |
| `RK4` | Explicit | 4 | Classic Runge-Kutta method with fixed step size |

# Event Detection

One main component of the library is the event detection system, which allows users to define conditions that trigger during integration. This feature was mainly developed for accurately detecting crossings in a *Poincaré surface of section* in dynamical systems, but it can be used for any situation where you need to detect when a certain condition is met during the integration of a system of ODE's.

- **Compile-time events**: For events that can be hardcoded in a project, it is preferred to use the compile-time event system, which avoids the overhead of virtual function calls, and allows for inlining and more compiler optimizations. This is achieved via the `StaticEventStepper` class. See the relevant [example](tutorials/CompileTimeEvents.cpp) for different ways to declare relevant solvers.

- **Runtime events**: For events whose number or type is not known at compile-time, the polymorphic `Event<T>` class
is provided, which requires that the solver is declared with `ode::SolverPolicy::RichVirtual` or `RichStatic`.
See the relevant [example](tutorials/RuntimeEvents.cpp) for examples of how to use the runtime event system.

# Arbitrary Precision Support

All classes are templated, and the `T` template parameter can be any numeric type, including arbitrary precision:
```cpp
#include <odecraft/odecraft.hpp>
#include <mpreal.h>

using namespace ode;

int main(){

    // Set precision to 100 bits for all subsequent mpreal objects
    mpfr::mpreal::set_default_prec(100);

    using T = mpfr::mpreal; // `T` alias for simplicity
    std::array<T, 2> q0 = {10, 0}; // Initial conditions
    
    // Let's create a solver for the simple harmonic oscillator using the RK45 method
    pbox::Box<OdeSolver<T, 2>> solver = make_vsolver(
        Stepper::RK45,
        OdeData{
            .Rhs=[](auto* dq_dt, const auto& t, const auto* q){
                dq_dt[0] = q[1];
                dq_dt[1] = -q[0];
            },
        },
        T{0}, // t0
        View1D<T, 2>{q0.data()},
        T{1e-10}, // relative tolerance
        T{1e-10} // absolute tolerance
    );

    solver->do_advance_until(1000);
    const T& x = solver->get_vector()[0];
    const T& v = solver->get_vector()[1];
    std::cout << "Final state: " << x << ", " << v << std::endl;
    return 0;
}
```

However, `mpfr::mpreal` performs heap allocation when instantiated, and every intermediate algebraic expression
creates a temporary `mpreal` object. This can be avoided by using the `lazy` library, which allows for lazy evaluation of expressions and avoids unnecessary temporaries. See the [lazy](https://github.com/phyzan/lazy) submodule for more details. In practice, it can be used exactly like `mpreal` in most cases, by simply replacing `mpfr::mpreal` with `lazy::LazyType<mpfr::mpreal>` in the code above, using
```cpp
#include <lazy/apps/mpfrLazy.hpp>
```
and calling
```cpp
lazy::set_default_mpreal_prec(prec);
```
instead of
```cpp
mpfr::mpreal::set_default_prec(prec);
```

For instance, this [example](tutorials/CompileTimeEvents.cpp) demonstrates the performance difference between `mpreal` and `lazy::LazyType<mpfr::mpreal>` for a simple harmonic oscillator.

Note that as the number of requested bits of precision increases, the performance difference diminishes,
and the overhead of algebraic evaluations dominates.


# Architecture

The library uses a **two-tier architecture** combining static and dynamic polymorphism via CRTP (Curiously Recurring Template Pattern):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VIRTUAL INTERFACE LAYER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ┌──────────────────┐         ┌─────────────────────┐                 │
│        │  OdeSolver<T,N>  │────────▶│ OdeRichSolver<T,N>  │                 │
│        └──────────────────┘         └─────────────────────┘                 │
│         (base interface)             (+ runtime events)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         STATIC IMPLEMENTATION LAYER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ┌────────────────────┐       ┌───────────────────────┐               │
│        │ BaseSolver<T,N,SP> │──────▶│  RichSolver<T,N,SP>   │               │
│        └────────────────────┘       └───────────────────────┘               │
│              (CRTP base) │             (+ runtime events)                   │
│                   │                             │                           │
│                   │                             │                           │
│                   ───────────────────────────────                           │
│                                  │                                          │
│        ┌─────────────────────────┼────────────────────────┐                 │
│        │            │            │            │           │                 │
│     ┌───▼───┐  ┌────▼───┐   ┌────▼───┐   ┌────▼───┐   ┌───▼───┐             │
│     │ Euler │  │  RK23  │   │  RK45  │   │ DOP853 │   │  BDF  │             │
│     └───────┘  └────────┘   └────────┘   └────────┘   └───────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The `SolverPolicy` template parameter controls inheritance and feature availability:

```
┌─────────────┬───────────────────────────────────────────────────────────────┐
│   POLICY    │                      INHERITANCE CHAIN                        │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│   Static    │   RK45 ───▶ BaseSolver                                        │
│             │   (maximum performance, no virtuals, no events)               │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│ RichStatic  │   RK45 ───▶ RichSolver ───▶ BaseSolver                        │
│             │   (events, no virtuals)                                       │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│  Virtual    │   RK45 ───▶ BaseSolver ───▶ OdeSolver                         │
│             │   (runtime polymorphism, no events)                           │
│             │                                                               │
├─────────────┼───────────────────────────────────────────────────────────────┤
│             │                                                               │
│ RichVirtual │   RK45 ───▶ RichSolver ───▶ BaseSolver ───▶ OdeRichSolver     │
│             │   (full features: virtuals + events)                          │
│             │                                                               │
└─────────────┴───────────────────────────────────────────────────────────────┘
```


### Design Patterns

| Pattern | Usage |
|---------|-------|
| **CRTP** | `BaseSolver<Derived, ...>` enables static dispatch without virtual overhead |
| **Policy Pattern** | `SolverPolicy` enum for compile-time feature selection |
| **Factory Pattern** | `getSolver()` and `make_solver()` for solver instantiation |

---

## Directory Structure

```
odecraft/
├── include/
│   └── odecraft/                    # All headers, header-only
│       ├── Core/                    # Foundation & base classes
│       │   ├── VirtualBase.hpp      # Virtual interfaces & solver policies
│       │   ├── VirtualTraits.hpp    # Traits for virtual solvers
│       │   ├── BaseSolver.hpp       # CRTP base solver
│       │   ├── RichBase.hpp         # Event-aware solver extension
│       │   ├── Events.hpp           # Event detection system
│       │   ├── FinDiff.hpp          # Finite difference utilities
│       │   ├── StaticEventStepper.hpp  # Objective-based solver interface
│       │   ├── SolverFactory.hpp    # Factory for solver instantiation
│       │   └── *_impl.hpp           # Implementation files
│       │
│       ├── Steppers/                # Concrete solver implementations
│       │   ├── Steppers.hpp         # Common solver includes
│       │   ├── Euler.hpp            # Simple Euler method (1st order)
│       │   ├── RungeKutta.hpp       # Generic Runge-Kutta framework
│       │   ├── DOPRI.hpp            # Runge-Kutta RK23, RK45 (adaptive)
│       │   ├── DOP853.hpp           # High-order explicit RK (8th order)
│       │   ├── BDF.hpp              # Implicit solver for stiff systems
│       │   └── *_impl.hpp           # Implementation files
│       │
│       ├── Interpolation/           # Dense output & interpolation
│       │   ├── NdInterpolator.hpp   # N-dimensional interpolator base
│       │   ├── VectorFields.hpp     # Sampled vector field interpolation
│       │   ├── Regular/             # Regular grid interpolation
│       │   │   ├── Grids.hpp        # Grid data structures
│       │   │   └── RegularGridInterpolator.hpp
│       │   ├── Scattered/           # Scattered data interpolation
│       │   │   ├── Delaunay.hpp     # Delaunay triangulation
│       │   │   └── ScatteredNdInterpolator.hpp
│       │   ├── Univariate/          # 1D interpolation
│       │   │   └── StateInterp.hpp  # State interpolation for solvers
│       │   └── *_impl.hpp           # Implementation files
│       │
│       ├── Chaos/                   # Dynamical systems analysis
│       │   ├── VariationalSolvers.hpp    # Lyapunov exponent computation
│       │   └── VariationalSolvers_impl.hpp
│       │
│       ├── DenseOde/                # High-level ODE wrapper
│       │   ├── OdeInt.hpp           # Dense-output integration front end
│       │   └── OdeInt_impl.hpp      # Implementation
│       │
│       ├── OdeResult/               # Integration result storage
│       │   ├── OdeResult.hpp        # Result container
│       │   └── OdeResult_impl.hpp   # Implementation
│       │
│       ├── Tools.hpp                # Utilities, shared concepts & OdeData
│       └── odecraft.hpp             # Main include (all headers)
│
├── external/                        # Git submodules (bundled header-only dependencies)
│   ├── xdiff/                       # Automatic differentiation library (bundles its own `lazy` + `mpreal` submodules)
│   ├── ndspan/                      # Multi-dimensional array views and utilities
│   ├── polybox/                     # Wrapper for dynamically allocated types
│   └── qhull/                       # Convex hull library (for Delaunay triangulation)
│
├── tests/                           # C++ test suite, compiled into one odecraft_tests executable
│   ├── include/                     # One <name>.hpp per test file, declaring void test_<name>()
│   └── src/                         # One <name>.cpp per test file, implementing it; main.cpp calls them all
│
├── tutorials/                       # Standalone example programs referenced from the README
│   ├── ArbitraryPrecision.cpp       # Arbitrary-precision (mpreal / lazy) usage
│   ├── CompileTimeEvents.cpp        # Compile-time event system usage
│   └── RuntimeEvents.cpp            # Runtime (polymorphic) event system usage
│
├── .clang-tidy                      # clang-tidy check configuration
├── CMakeLists.txt                   # odecraft::odecraft interface target + odecraft_tests build
├── LICENSE
└── README.md
```

---

## Performance Tips

- **Prefer SolverPolicy::Static**  when no runtime-event detection or type-erasure is required 
- **Set appropriate tolerances** - tighter tolerances mean smaller steps
- **Use `BDF`** for stiff problems


---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with modern C++ for scientists and engineers</sub>
</p>
