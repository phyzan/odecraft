#pragma once


#include "Tools.hpp"
#include <cstddef>
#include <utility>


namespace ode{


template<typename SD, typename T>
T rms_norm(const state_t<T, SD>& state){
    T norm2{0};
    size_t n = 0;
    SD::for_each([&norm2, &n](const value_t<T, SD>& s){
        norm2 += SD::norm_sqr(s);
        ++n;
    }, state);
    return sqrt(norm2/n);
}

template<typename SD, typename T>
T rms_norm(const state_t<T, SD>& state, const real_state_t<T, SD>& scale){
    T norm2{0};
    size_t n = 0;
    SD::for_each([&norm2, &n](const value_t<T, SD>& s_i, const T& scale_i){
        norm2 += SD::norm_sqr(s_i) / (scale_i*scale_i);
        ++n;
    }, state, scale);
    return sqrt(norm2/n);
}

template<typename T>
void inv_mat_row_major(T* out, const T* mat, size_t N, T* work, T* lu, size_t* pivot) {
    // out: size N*N, row-major order
    // mat: size N*N, row-major order
    // work: size N
    // lu: size N*N
    // pivot: size N
    if (N == 0){
        return;
    }
    
    for (size_t i = 0; i < N * N; ++i){
        lu[i] = mat[i];
    }

    for (size_t i = 0; i < N; ++i){
        pivot[i] = i;
    }

    for (size_t i = 0; i < N; ++i) {
        size_t max_row = i;
        T max_val = abs(lu[i*N + i]);
        for (size_t j = i + 1; j < N; ++j) {
            T val = abs(lu[j*N + i]);
            if (val > max_val) { max_val = val; max_row = j; }
        }
        // assert(max_val != 0 && "Matrix is singular");

        if (max_row != i) {
            for (size_t k = 0; k < N; ++k){
                std::swap(lu[i*N + k], lu[max_row*N + k]);
            }
            std::swap(pivot[i], pivot[max_row]);
        }

        for (size_t j = i + 1; j < N; ++j) {
            lu[j*N + i] /= lu[i*N + i];
            for (size_t k = i + 1; k < N; ++k){
                lu[j*N + k] -= lu[j*N + i] * lu[i*N + k];
            }
        }
    }

    for (size_t col = 0; col < N; ++col) {
        for (size_t i = 0; i < N; ++i){
            work[i] = (pivot[i] == col) ? 1 : 0;
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < i; ++j){
                work[i] -= lu[i*N + j] * work[j];
            }
        }

        for (size_t i = N; i-- > 0;) {
            for (size_t j = i + 1; j < N; ++j){
                work[i] -= lu[i*N + j] * work[j];
            }
            work[i] /= lu[i*N + i];
        }

        for (size_t i = 0; i < N; ++i){
            out[i*N + col] = work[i];
        }

    }
}


template<typename T>
T detLU_row_major(T* mat, size_t N) {
    if (N == 0) {
        return 1;
    }

    T det = 1;
    int sign = 1;

    for (size_t i = 0; i < N; ++i) {
        // Partial pivoting
        size_t pivot = i;
        T max_val = abs(mat[i * N + i]);
        for (size_t j = i + 1; j < N; ++j) {
            T val = abs(mat[j * N + i]);
            if (val > max_val) {
                pivot = j;
                max_val = val;
            }
        }

        if (max_val == T(0)) {
            return T(0);
        }

        if (pivot != i) {
            // Swap rows
            for (size_t k = 0; k < N; ++k) {
                T tmp = mat[i * N + k];
                mat[i * N + k] = mat[pivot * N + k];
                mat[pivot * N + k] = tmp;
            }
            sign = -sign;
        }

        det *= mat[i * N + i];

        // Eliminate below pivot
        for (size_t j = i + 1; j < N; ++j) {
            T factor = mat[j * N + i] / mat[i * N + i];
            for (size_t k = i; k < N; ++k) {
                mat[j * N + k] = mat[j * N + k] - factor * mat[i * N + k];
            }
        }
    }

    return det*sign;
}


// out = A * B over the leading n x n blocks of column-major matrices, the rest of out is left as
// it is. out must be neither A nor B.
template<typename T, size_t N>
void matmul(FArray2D<T, N, N>& out, const FArray2D<T, N, N>& A, const FArray2D<T, N, N>& B, size_t n){
    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < n; ++i){
            out(i, j) = 0;
        }
        for (size_t k = 0; k < n; ++k){
            const cache_t<T> b_kj = B(k, j);
            for (size_t i = 0; i < n; ++i){
                out(i, j) += A(i, k) * b_kj;
            }
        }
    }
}

} // namespace ode
