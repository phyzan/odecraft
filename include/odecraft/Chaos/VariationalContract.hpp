#pragma once

#include "../Toolkit/Toolkit.hpp"
#include "odecraft/Toolkit/Tools.hpp"
#include <tuple>
#include <type_traits>

namespace ode::chaos{

namespace detail{

// Whether a tuple holds only lvalue references, so that what they refer to outlives it
template<typename Tuple>
inline constexpr bool holds_references = false;

template<typename... E>
inline constexpr bool holds_references<std::tuple<E...>> = (std::is_lvalue_reference_v<E> && ...);

} // namespace ode::chaos::detail

template<typename State>
struct VarState{
    State main;
    State variation;
};

template<typename SD, typename OdeType>
struct VarSD{

    template<typename T>
    using value_type = value_t<T, SD>; // same value_type

    template<typename S>
    using state_type = VarState<typename SD::template state_type<S>>;

    template<typename T>
    static T norm_sqr(const value_type<T>& value){
        return SD::norm_sqr(value);
    }

    template<typename To, typename From>
    static void resize_like(state_type<To>& dst, const state_type<From>& ref){
        SD::resize_like(dst.main, ref.main);
        SD::resize_like(dst.variation, ref.variation);
    }

    template<typename F, typename First, typename... Rest>
    static void for_each(F&& f, First&& proto, Rest&&... rest){
        // Forwarding the state, then taking its member, keeps its constness and value category
        SD::for_each(f, std::forward<First>(proto).main, std::forward<Rest>(rest).main...);
        SD::for_each(f, std::forward<First>(proto).variation, std::forward<Rest>(rest).variation...);
    }

    template<typename F, typename V>
    static bool all_true(F&& f, const state_type<V>& state){
        // if false for state.main, then it wont even check for state.variation
        return SD::all_true(f, state.main) && SD::all_true(f, state.variation);
    }

    template<typename V>
    static std::ostream& console(std::ostream& out, const state_type<V>& state){
        SD::console(out, state.main) << "\n";
        return SD::console(out, state.variation);
    }


    struct JD{

        /*
        [J,             0
         dJ/dq*delta_q  J]
        */

        // mat[0] = jacobian of the main system, df/dq, for upper left and lower right of the true jacobian
        // mat[1] = d(J)/dq * delta_q for lower left
        template<typename S>
        using jacobian_matrix = std::array<typename SD::JD::template jacobian_matrix<S>, 2>;

        template<typename S, typename From>
        static void resize_like(jacobian_matrix<S>& J, const state_type<From>& ref){
            SD::JD::resize_like(J[0], ref.main);
            SD::JD::resize_like(J[1], ref.variation);
        }

        template<typename F, typename V>
        static bool all_true(F&& f, const jacobian_matrix<V>& jacmat){
            return SD::JD::all_true(f, jacmat[0]) && SD::JD::all_true(f, jacmat[1]);
        }

        struct linalg{

            // Only the columns by the elements of main: each is that column of both blocks,
            // [J; dJ/dq*delta_q]. The columns by variation are [0; J], already in block 0.
            template<typename Prim, typename Sec, typename... J, typename... Q>
            static void for_each_column(Prim&& f, Sec&& g, std::tuple<J&...> out, Q&&... states){
                SD::JD::linalg::for_each_column(std::forward<Prim>(f), [&g](auto&&... elems){
                    decltype(auto) cols = g(std::forward<decltype(elems)>(elems)...);
                    static_assert(detail::holds_references<std::remove_cvref_t<decltype(cols)>>, "g must return references, e.g. std::tie");
                    return std::apply([](auto&... col){
                        return std::tuple_cat(std::tie(col.main, col.variation)...);
                    }, cols);
                }, blocks(out), states.main...);
            }

            // Only the rows of the outputs of variation: each gradient is that row of both blocks,
            // [dJ/dq*delta_q, J]. The rows of main are [J, 0], nothing new.
            template<typename Prim, typename Sec, typename... J, typename... Q>
            static void for_each_row(Prim&& f, Sec&& g, std::tuple<J&...> out, Q&&... states){
                SD::JD::linalg::for_each_row(std::forward<Prim>(f), [&g](auto&&... elems){
                    decltype(auto) rows = g(std::forward<decltype(elems)>(elems)...);
                    static_assert(detail::holds_references<std::remove_cvref_t<decltype(rows)>>, "g must return references, e.g. std::tie");
                    return std::apply([](auto&... row){
                        return std::tuple_cat(std::tie(row.variation, row.main)...);
                    }, rows);
                }, blocks(out), states.variation...);
            }

            // out = J*v: out.main = J*v.main, out.variation = dJ/dq*delta_q*v.main + J*v.variation.
            // out must not be v.
            template<typename T>
            static void dot(state_type<value_type<T>>& out, const jacobian_matrix<value_type<T>>& J, const state_type<value_type<T>>& v){
                SD::JD::linalg::dot(out.main, J[0], v.main);
                SD::JD::linalg::dot(out.variation, J[1], v.main);
                SD::JD::linalg::dot_add(out.variation, T(1), J[0], v.variation);
            }

            // out = I - c*J: the identity is on the diagonal blocks only, so block 0 is I - c*J
            // and block 1 is -c*dJ/dq*delta_q. out may be J.
            template<typename T>
            static void identity_minus(jacobian_matrix<value_type<T>>& out, const T& c, const jacobian_matrix<value_type<T>>& J){
                SD::JD::linalg::identity_minus(out[0], c, J[0]);
                SD::JD::linalg::scale(out[1], -c, J[1]);
            }

            // out = c*src, block by block. out may be src.
            template<typename T>
            static void scale(jacobian_matrix<value_type<T>>& out, const T& c, const jacobian_matrix<value_type<T>>& src){
                SD::JD::linalg::scale(out[0], c, src[0]);
                SD::JD::linalg::scale(out[1], c, src[1]);
            }

            // The pivots of block 0, the only block factored
            using pivots_type = SD::JD::linalg::pivots_type;

            // The iteration matrix is block lower triangular, [[A, 0], [C, A]]: only A, block 0, is
            // factored, and the matrix is singular exactly when A is
            template<typename T>
            static bool lu_factor(jacobian_matrix<value_type<T>>& A, pivots_type& piv){
                return SD::JD::linalg::lu_factor(A[0], piv);
            }

            // Block forward substitution: A x.main = b.main, then A x.variation = b.variation - C x.main.
            // x may be b.
            template<typename T>
            static void lu_solve(state_type<value_type<T>>& x, const jacobian_matrix<value_type<T>>& LU, const pivots_type& piv, const state_type<value_type<T>>& b){
                SD::JD::linalg::lu_solve(x.main, LU[0], piv, b.main);
                if (&x != &b){
                    copy_state<SD>(x.variation, b.variation);
                }
                SD::JD::linalg::dot_add(x.variation, T(-1), LU[1], x.main);
                SD::JD::linalg::lu_solve(x.variation, LU[0], piv, x.variation);
            }

        private:

            // Both blocks of every matrix: std::tie(J_0[0], J_0[1], J_1[0], J_1[1], ...)
            template<typename... J>
            static auto blocks(std::tuple<J&...> out){
                return std::apply([](auto&... Jk){
                    return std::tuple_cat(std::tie(Jk[0], Jk[1])...);
                }, out);
            }

        };

    };

};

} // namespace ode::chaos