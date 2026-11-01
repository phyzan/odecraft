#pragma once



/*


namespace ode{

=======================================
Interface:

Solver<T, SD>

where:

T = real valued scalar type of the integration parameter
SD = State Descriptor
SD::JD = Jacobian Descriptor
=======================================

ode::DenseDescriptor<N> (DenseDescriptor.hpp) implements this interface for real states held in a
std::array<T, N>, or in a std::vector<T> with N = 0, with a dense Jacobian.

struct StateDescriptor{

    template<typename T>
    using value_type = std::complex<T>;

    template<typename S>
    using state_type = std::array<S, 5>;

    template<typename T>
    static T norm_sqr(const scalar<T>& s);

    template<typename To, typename From>
    static void resize_like(state_type<To>& dst, const state_type<From>& ref);

    template<typename F, typename First, typename... Rest>
    static void for_each(F&& f, First&& proto, Rest&&... rest);

    template<typename F, typename V>
    static bool all_true(F&& f, const state_type<V>& state){
        // assuming f(const V&) -> true
        // e.g. if state is an array
        for (size_t i = 0; i < state.size(); ++i){
            if (!f(state[i])){
                return false;
            }
        }
        return true;
    }

    // Prints the state, in show_state
    template<typename V>
    static std::ostream& console(std::ostream& out, const state_type<V>& state);

    // or could be `using JD = ...` if JD exists elsewhere, but is implemented for this SD's state_type
    struct JD{



        The ode solvers reach the Jacobian only through SD::JD::resize_like and the functions below,
        and never index it. For usual matrix-like jacobian: J(i, j) = df_i/dq_j. The jacobian_matrix need not hold n*n elements, so the ode operations must be jacobian-agnostic.
        e.g. a pattern may store only its nonzero ones.

        template<typename S>
        using jacobian_matrix = ...;

        template<typename S, typename From>
        static void resize_like(jacobian_matrix<S>& J, const state_type<From>& ref){
            // like `resize_like`: gives J the shape of the Jacobian at a state shaped like ref
            // (a no-op for a static size). The solvers create every Jacobian through it.
        }

        template<typename F, typename V>
        static bool all_true(F&& f, const jacobian_matrix<V>& jacmat); // similar to SD::all_true


        struct linalg{

            Toolkit for matrix-related calculations.
            Anything inside this struct does NOT need to be templated, as calls will be
            made using JD::linalg::some_func(my_buffers...), and not somefunc::template <T>...
            
            For templated functions, template resolution works, for non-templated it works as long as the types match, which is why the library guarantees that the jacobian_matrix's template
            will only be substituted with SD::value_type<T>, where T is the main real-type template
            scalar value (e.g. double, mpreal or whatever) that is passed to the solver.


            Which is why the provided functions must at least accept
            types that correspond to the Real type `T` provided in the Solver's
            template parameters. Templated on T, as below, they take the elements
            as value_type<T>, and abs() of an element gives a T.

            // ---- Filling a Jacobian ----

            // By columns (finite differences). out = std::tie(J_0, J_1, ...), all of one shape. For
            // each column j, g(states[j]...) returns that column of every J_k as a tuple, e.g.
            // std::tie(col_0, col_1, ...), and f(J_k(i, j), col_k[i]) writes each entry J_k holds.
            // Use g's result before the next call.
            template<typename Prim, typename Sec, typename... J, typename... Q>
            static void for_each_column(Prim&& f, Sec&& g, std::tuple<J&...> out, Q&&... states){
                // Q... are state_type<something>, such that g(states[j]...) -> tuple of state_type<something_else>
            }

            // By rows (autodiff), the same transposed: g(states[i]...) returns row i of every J_k.
            template<typename Prim, typename Sec, typename... J, typename... Q>
            static void for_each_row(Prim&& f, Sec&& g, std::tuple<J&...> out, Q&&... states){
                transpose iteration from `for_each_column`
            }



            // ---- Products ----

            // out = J * v. out must not be v.
            // The variational equations, d(delta_q)/dt = J * delta_q.
            template<typename T>
            static void dot(state_type<value_type<T>>& out, const jacobian_matrix<value_type<T>>& J, const state_type<value_type<T>>& v);

            // out += a*J*v. out must not be v.
            template<typename T>
            static void dot_add(state_type<value_type<T>>& out, const T& a, const jacobian_matrix<value_type<T>>& J, const state_type<value_type<T>>& v);

            // out = I - c*J, the iteration matrix of the implicit methods (BDF: c = h/alpha). out may be J.
            template<typename T>
            static void identity_minus(jacobian_matrix<value_type<T>>& out, const T& c, const jacobian_matrix<value_type<T>>& J);

            // out = c*src. out may be src.
            template<typename T>
            static void scale(jacobian_matrix<value_type<T>>& out, const T& c, const jacobian_matrix<value_type<T>>& src);


            // ---- Linear systems ----

            // The row pivots of an LU factorization
            using pivots_type = ... // e.g. std::vector<size_t>;

            // Factorizes A in place, P*A = L*U with partial pivoting: A is overwritten with L below the
            // diagonal (its unit diagonal implied) and U on and above it, and piv takes the size of A.
            // Returns false if A is (nearly) singular.
            // A jacobian_matrix of a pattern whose factors fill in (e.g. a band, with pivoting) must
            // leave room for the fill-in, as LAPACK's band storage does.
            template<typename T>
            static bool lu_factor(jacobian_matrix<value_type<T>>& A, pivots_type& piv);

            // x = A^-1 * b, from the A and piv that lu_factor wrote. x may be b.
            template<typename T>
            static void lu_solve(state_type<value_type<T>>& x, const jacobian_matrix<value_type<T>>& LU, const pivots_type& piv, const state_type<value_type<T>>& b);
        }

    };
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
