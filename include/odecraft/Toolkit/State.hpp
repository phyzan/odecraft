#pragma once



/*


namespace ode{

=======================================
Interface:

Solver<Real, Scalar, SD, JD = void>

where:

Real = real valued scalar type of the integration parameter
Scalar = scalar type of the state vector entried, can be anything that
    has standard arithmetic operations (+, -, /, *, pow) ans abs(Scalar) -> Real.
    This can be complex<Real>, eigen array, etc.
SD = State Descriptor
JD = Jacobian Descriptor
=======================================

struct StateDescriptor{

    // Decoupled from the Rhs function, since the same SD
    // can be for many Rhs's - dynamical systems

    // --------------------------------------------
    template<typename T>
    using Scalar = std::complex<T>;  // default void

    template<typename S>
    using State = std::array<S, 5> // default void

    template<typename T>
    static T norm(const State<Scalar<T>>& state);
    // --------------------------------------------

    OR

    // --------------------------------------------
    using State = std::array<std::complex<double>, 5>;

    using Scalar = std::complex<double>;

    static double norm(const State& state);
    // --------------------------------------------

    AND

    for_each

    In both cases,

    Scalar will be the State elements that SD::zip(f, states...)
    will be passing in f.
};


// optional
If void, JacMat = Array2D<Scalar> and default Jac function.
Otherwise, MUST provide a Jac function and LU / tools as functions,
which will be specified.

struct JacobianDescriptor{

    --------------------------------------
    template<typename S>
    using JacMat = Array2D<S>;

    OR

    using JacMat = Array2D<MyScalar>;
    --------------------------------------

    AND

    toolkit for matrix-related calculations.


};

} // namespace ode



TODO

General:

Be careful using the {} constructor, it does not always zero-initialize.
Fix in xdiff too. Maybe call for_each to zero-out its elements

For runtime size of ode systems:

1) Assume that the state's size remains constant.
2) Automatic sizing of temporary states, by copying from the
initial state. Must happen in xdiff too, in the seeding where Duals are created
on the spot.
3) zip (rename for_each) must run over const-only states too. must make safe for runtime size, with the method proposed. For LazType, maybe update the workers when an operation with a different size is invoked. lazex should provide an API for that. 

*/







/*


namespace ode{

=======================================
Interface:

Solver<T, SD, JD = void>

where:

T = real valued scalar type of the integration parameter
SD = State Descriptor
JD = Jacobian Descriptor
=======================================

struct StateDescriptor{

    template<typename T>
    using scalar = std::complex<T>;

    template<typename S>
    using state_type = std::array<S, 5>;

    template<typename S>
    using jacobian_matrix = ...;

    template<typename T>
    static T norm_sqr(const scalar<T>& s);

    template<typename F, typename First, typename... Rest>
    static void for_each(F&& f, First&& proto, Rest&&... rest);

    template<typename To, typename From>
    static void resize_like(state_type<To>& dst, const state_type<From>& ref);
};


// optional
If void, JacMat = Array2D<Scalar> and default Jac function.
Otherwise, MUST provide a Jac function and LU / tools as functions,
which will be specified.

struct JacobianDescriptor{
    
    toolkit for matrix-related calculations.
    Does not need to be templated, as calls will be
    made using JD::some_func(my_buffers...) and for templated
    functions, template resolution works, for non-templated it works as long as the types match.

    Which is why the provided functions must at least accept
    types that correspond to the Real type `T` provided in the Solver's
    template parameters.
};

} // namespace ode



TODO

General:

Be careful using the {} constructor, it does not always zero-initialize.
Fix in xdiff too. Maybe call for_each to zero-out its elements

For runtime size of ode systems:

1) Assume that the state's size remains constant.
2) Automatic sizing of temporary states, by copying from the
initial state. Must happen in xdiff too, in the seeding where Duals are created
on the spot.
3) zip (rename for_each) must run over const-only states too. must make safe for runtime size, with the method proposed. For LazType, maybe update the workers when an operation with a different size is invoked. lazex should provide an API for that. 

*/
