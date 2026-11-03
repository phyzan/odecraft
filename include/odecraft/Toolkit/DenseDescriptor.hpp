#pragma once

#include "Tools.hpp"
#include <cstddef>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>


namespace ode{

namespace detail{

template<size_t N>
using dense_state_base = std::conditional_t<(N > 0), xdiff::StaticState<N>, xdiff::DynamicState>;

} // namespace ode::detail


/// @brief The State Descriptor of a real state held in a std::array<T, N>, or in a std::vector<T>
/// with N = 0 for a runtime size, with a dense column-major Jacobian, FArray2D<T, N, N>.
/// See State.hpp for the interface.
template<size_t N>
struct DenseDescriptor : public detail::dense_state_base<N>{

    using Base = detail::dense_state_base<N>;

    template<typename T>
    using value_type = T;

    template<typename S>
    using state_type = Base::template state_type<S>;

    template<typename T>
    static T norm_sqr(const value_type<T>& s){
        return s*s;
    }

    template<typename F, typename V>
    static bool all_true(F&& f, const state_type<V>& state){
        for (const V& s : state){
            if (!f(s)){
                return false;
            }
        }
        return true;
    }

    template<typename V>
    static std::ostream& console(std::ostream& out, const state_type<V>& state){
        for (size_t i = 0; i < state.size(); ++i){
            out << (i > 0 ? " " : "") << state[i];
        }
        return out;
    }

    struct JD{

        template<typename S>
        using jacobian_matrix = FArray2D<S, N, N>;

        template<typename S, typename From>
        static void resize_like(jacobian_matrix<S>& J, const state_type<From>& ref){
            if constexpr (N == 0){
                J.resize(ref.size(), ref.size());
            }
        }

        template<typename F, typename V>
        static bool all_true(F&& f, const jacobian_matrix<V>& J){
            const V* a = J.data();
            for (size_t k = 0; k < J.size(); ++k){
                if (!f(a[k])){
                    return false;
                }
            }
            return true;
        }

        // The loops run down the columns, which are contiguous
        struct linalg{

            using pivots_type = state_type<size_t>;

            // out = std::tie(J_0, J_1, ...): g returns column j of every J_k as a tuple
            template<typename Prim, typename Sec, typename... J, typename... Q>
            static void for_each_column(Prim&& f, Sec&& g, std::tuple<J&...> out, Q&&... states){
                // Q... are state_type<something>, such that g(states[j]...) -> tuple of state_type<something_else>
                const size_t n = rows(std::get<0>(out));
                for (size_t j = 0; j < n; ++j){
                    decltype(auto) cols_j = g(states[j]...);
                    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(cols_j)>> == sizeof...(J), "g must return one column per matrix");
                    [&]<size_t... k>(std::index_sequence<k...>){
                        (fill_column(f, std::get<k>(out), j, std::get<k>(cols_j)), ...);
                    }(std::index_sequence_for<J...>{});
                }
            }

            template<typename Prim, typename Sec, typename... J, typename... Q>
            static void for_each_row(Prim&& f, Sec&& g, std::tuple<J&...> out, Q&&... states){
                const size_t n = rows(std::get<0>(out));
                for (size_t i = 0; i < n; ++i){
                    decltype(auto) rows_i = g(states[i]...);
                    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(rows_i)>> == sizeof...(J), "g must return one row per matrix");
                    [&]<size_t... k>(std::index_sequence<k...>){
                        (fill_row(f, std::get<k>(out), i, std::get<k>(rows_i)), ...);
                    }(std::index_sequence_for<J...>{});
                }
            }

            // out = J * v, a sum of the columns of J. out must not be v.
            template<typename T>
            static void dot(state_type<value_type<T>>& out, const jacobian_matrix<value_type<T>>& J, const state_type<value_type<T>>& v){
                using S = value_type<T>;
                const size_t n = rows(J);
                const S* a = J.data();
                for (size_t i = 0; i < n; ++i){
                    out[i] = 0;
                }
                for (size_t j = 0; j < n; ++j){
                    const S* J_j = a + j*n;
                    const cache_t<S> v_j = v[j];
                    for (size_t i = 0; i < n; ++i){
                        out[i] += J_j[i] * v_j;
                    }
                }
            }

            // out += a*J*v, the columns of J scaled by a*v. out must not be v.
            template<typename T>
            static void dot_add(state_type<value_type<T>>& out, const T& a, const jacobian_matrix<value_type<T>>& J, const state_type<value_type<T>>& v){
                using S = value_type<T>;
                const size_t n = rows(J);
                const S* M = J.data();
                for (size_t j = 0; j < n; ++j){
                    const S* J_j = M + j*n;
                    const cache_t<S> av_j = a*v[j];
                    for (size_t i = 0; i < n; ++i){
                        out[i] += J_j[i] * av_j;
                    }
                }
            }

            // out = I - c*J. out may be J.
            template<typename T>
            static void identity_minus(jacobian_matrix<value_type<T>>& out, const T& c, const jacobian_matrix<value_type<T>>& J){
                using S = value_type<T>;
                const size_t n = rows(J);
                for (size_t j = 0; j < n; ++j){
                    for (size_t i = 0; i < n; ++i){
                        out(i, j) = (i == j) ? S(1) - c*J(i, j) : -c*J(i, j);
                    }
                }
            }

            // out = c*src. out may be src.
            template<typename T>
            static void scale(jacobian_matrix<value_type<T>>& out, const T& c, const jacobian_matrix<value_type<T>>& src){
                const size_t n = rows(src);
                for (size_t j = 0; j < n; ++j){
                    for (size_t i = 0; i < n; ++i){
                        out(i, j) = c*src(i, j);
                    }
                }
            }

            // LU factorization with partial pivoting, P*A = L*U, in place: A is overwritten with L below
            // the diagonal (its unit diagonal implied) and U on and above it. Step k of the elimination
            // swaps row k with row piv[k] >= k. Returns false if A is (nearly) singular.
            template<typename T>
            static bool lu_factor(jacobian_matrix<value_type<T>>& A, pivots_type& piv){
                using S = value_type<T>;
                const size_t n = rows(A);
                if constexpr (N == 0){
                    piv.resize(n);
                }
                S* a = A.data();

                // Elimination below the diagonal, one column k at a time
                for (size_t k = 0; k < n; ++k){
                    S* A_k = a + k*n;

                    // The pivot row is the one with the largest entry of column k, on or below the diagonal
                    size_t p = k;
                    T max_val = abs(A_k[k]);
                    for (size_t i = k+1; i < n; ++i){
                        T val = abs(A_k[i]);
                        if (val > max_val){
                            max_val = val;
                            p = i;
                        }
                    }
                    piv[k] = p;

                    // Check singular. NaN passes, and surfaces in the solutions instead.
                    if (max_val < 1e-15){
                        return false;
                    }

                    if (p != k){
                        for (size_t j = 0; j < n; ++j){
                            std::swap(a[k + j*n], a[p + j*n]);
                        }
                    }

                    // Column k of L, then the update of the columns after it
                    for (size_t i = k+1; i < n; ++i){
                        A_k[i] /= A_k[k];
                    }
                    for (size_t j = k+1; j < n; ++j){
                        S* A_j = a + j*n;
                        const cache_t<S> u_kj = A_j[k];
                        for (size_t i = k+1; i < n; ++i){
                            A_j[i] -= A_k[i] * u_kj;
                        }
                    }
                }
                return true;
            }

            // x = A^-1 * b, from the LU and piv that lu_factor wrote. x may be b.
            template<typename T>
            static void lu_solve(state_type<value_type<T>>& x, const jacobian_matrix<value_type<T>>& LU, const pivots_type& piv, const state_type<value_type<T>>& b){
                using S = value_type<T>;
                const size_t n = rows(LU);
                const S* a = LU.data();

                if (&x != &b){
                    x = b;
                }

                // P*b, with the swaps in the order of the elimination
                for (size_t k = 0; k < n; ++k){
                    if (piv[k] != k){
                        std::swap(x[k], x[piv[k]]);
                    }
                }

                // Forward substitution, L * y = P*b, subtracting one column of L at a time
                for (size_t k = 0; k < n; ++k){
                    const S* L_k = a + k*n;
                    const cache_t<S> y_k = x[k];
                    for (size_t i = k+1; i < n; ++i){
                        x[i] -= L_k[i] * y_k;
                    }
                }

                // Backward substitution, U * x = y, from the last column of U
                for (size_t k = n; k-- > 0;){
                    const S* U_k = a + k*n;
                    x[k] /= U_k[k];
                    const cache_t<S> x_k = x[k];
                    for (size_t i = 0; i < k; ++i){
                        x[i] -= U_k[i] * x_k;
                    }
                }
            }

        private:

            // f(J(i, j), col[i]) down column j
            template<typename Prim, typename S, typename V>
            static void fill_column(Prim& f, jacobian_matrix<S>& J, size_t j, const V& col){
                const size_t n = rows(J);
                for (size_t i = 0; i < n; ++i){
                    f(J(i, j), col[i]);
                }
            }

            // f(J(i, j), row[j]) along row i
            template<typename Prim, typename S, typename V>
            static void fill_row(Prim& f, jacobian_matrix<S>& J, size_t i, const V& row){
                const size_t n = rows(J);
                for (size_t j = 0; j < n; ++j){
                    f(J(i, j), row[j]);
                }
            }

            // The number of rows (and columns), a compile time constant for a static size
            template<typename T>
            static size_t rows(const jacobian_matrix<value_type<T>>& A){
                if constexpr (N > 0){
                    return N;
                } else {
                    return A.shape(0);
                }
            }

        };

    };

};

} // namespace ode
