#ifndef ODECRAFT_VARIATIONAL_SOLVERS_IMPL_HPP
#define ODECRAFT_VARIATIONAL_SOLVERS_IMPL_HPP

#include <odecraft/Chaos/VariationalSolvers.hpp>
#include <odecraft/Tools.hpp>

namespace ode::chaos{

template<typename T, size_t N, hasRhsFunc<T> OdeType>
VariationalOdeSys<T, N, OdeType>::VariationalOdeSys(OdeType ode, size_t ode_nsys, T atol) : ode_(std::move(ode)), scratch(ode_nsys), nsys_(ode_nsys), atol_(std::move(atol)) {
    if constexpr (N > 0){
        assert(N==ode_nsys && "Incorrect number of equations in VariationalOdeSys");
    }

    if constexpr (JP_MAIN == JacPolicy::Nullable){
        if (ode_.Jac == nullptr && atol_ == 0){
            throw std::runtime_error("Variational ODE systems that do not provide a Jacobian function for the main ODE system require a non-zero tolerance to compute the stepsize for finite differences");
        }
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys scratch
// ----------------------------------------------------------------------------

template<typename T, size_t N, hasRhsFunc<T> OdeType>
VariationalOdeSys<T, N, OdeType>::ScratchDynamic::ScratchDynamic(size_t nsys_main):
    nsys(nsys_main),
    findiffs_(3*nsys),
    jacmat_(nsys*nsys),
    duals_(nsys),
    dduals_(nsys){
    assert((N == 0 || N == nsys) && "Invalid nsys argument in VariationalOdeSys::ScratchDynamic");
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<T, 3*N>& VariationalOdeSys<T, N, OdeType>::ScratchDynamic::findiffs() const {return findiffs_;}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<T, N*N>& VariationalOdeSys<T, N, OdeType>::ScratchDynamic::jacmat() const {return jacmat_;}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<DualType<T, N, 1>, N>& VariationalOdeSys<T, N, OdeType>::ScratchDynamic::duals() const {return duals_;}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<DualType<T, N, 2>, N>& VariationalOdeSys<T, N, OdeType>::ScratchDynamic::dduals() const {return dduals_;}


template<typename T, size_t N, hasRhsFunc<T> OdeType>
VariationalOdeSys<T, N, OdeType>::ScratchStatic::ScratchStatic(size_t nsys_main){
    assert((N == 0 || N == nsys_main) && "Invalid nsys argument in VariationalOdeSys::ScratchStatic");
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<T, 3*N> VariationalOdeSys<T, N, OdeType>::ScratchStatic::findiffs() const {return Array1D<T, 3*N>();}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<T, N*N> VariationalOdeSys<T, N, OdeType>::ScratchStatic::jacmat() const {return Array1D<T, N*N>();}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<DualType<T, N, 1>, N> VariationalOdeSys<T, N, OdeType>::ScratchStatic::duals() const {return Array1D<DualType<T, N, 1>, N>();}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
Array1D<DualType<T, N, 2>, N> VariationalOdeSys<T, N, OdeType>::ScratchStatic::dduals() const {return Array1D<DualType<T, N, 2>, N>();}


// ----------------------------------------------------------------------------
// VariationalOdeSys accessors
// ----------------------------------------------------------------------------

template<typename T, size_t N, hasRhsFunc<T> OdeType>
const OdeType& VariationalOdeSys<T, N, OdeType>::ode() const{
    return ode_;
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
constexpr size_t VariationalOdeSys<T, N, OdeType>::nsys_main() const{
    if constexpr (N > 0){
        return N;
    } else {
        return nsys_;
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys Rhs
// ----------------------------------------------------------------------------

template<typename T, size_t N, hasRhsFunc<T> OdeType>
template<size_t Order>
requires (detail::FullRhsSupportsDuals<T, N, OdeType, Order> && N > 0)
void VariationalOdeSys<T, N, OdeType>::Rhs(DualType<T, 2*N, Order>* out, const T& t, SeedVec<T, 2*N, Order> q) const{
    /*
    TODO
    Is the first branch indeed preferable to the second one?
    It requires both an Rhs and a Jac call of Duals<Order>, filling
    N + N*N elements
    However the second branch only calls Rhs a single time,
    but using Duals<Order+1>
    Maybe the branches should be reversed ?

    TODO
    We could also provide an overloaded Rhs for dynamic size N,
    (and for specific Order, maybe just =1)
    but for now the provided Jac(...) is enough. Also its very unlikely
    that dynamic sized ODE systems will be used for variational equations,
    let alone call a templated Rhs for autodiff. Compile-size N unlocks all features.
    */

    /*
    Only the dual *width* is that of the augmented system (2*N), because the caller
    differentiates with respect to all 2*N augmented variables. The loop bounds and
    offsets stay at N: `out` and `q` hold 2*N entries in total, the original state in
    [0, N) and the deviation vector in [N, 2*N).
    */
    using AugDual = DualType<T, 2*N, Order>;
    if constexpr (MainRhsSupportsAugDuals<Order> && MainJacSupportsAugDuals<Order>){
        std::array<AugDual, N*N> scratch_jacmat; // size n*n, what ode_.Jac fills
        ode_.Rhs(out, t, q); // Fills the first half
        ode_.Jac(scratch_jacmat.data(), t, q);
        AugDual* delta_qdot = out + N;
        std::fill(delta_qdot, delta_qdot+N, T{0});
        for (size_t j=0; j<N; j++){
            for (size_t i=0; i<N; i++){
                delta_qdot[i] += scratch_jacmat[j*N + i] * q[N + j];
            }
        }
    } else {
        using AugDualHi = DualType<T, 2*N, Order+1>;
        std::array<AugDualHi, N> scratch_duals; // n for the rhs
        auto* rhs = scratch_duals.data();
        ode_.Rhs(rhs, t, q.template with_order<Order+1>());
        std::fill(out+N, out+2*N, T{0});
        for (size_t j=0; j<N; j++){
            out[j] = rhs[j].trimmed(); // Must truncate diff information. Besides it was always used for below.
            for (size_t i=0; i<N; i++){
                // Using q (and not trimming y)
                // because as mentioned earlier, it is assumed that q only contains a value and a gradient
                // with the gradient being exactly one along its index.
                out[i+N] += rhs[i].trimmed_diff_wrt(j) * q[N + j];
            }
        }
    }
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::Rhs(T* out, const T& t, const T* q) const{
    const size_t n = this->nsys_main();
    const T* delta_q = q + n;
    if constexpr (MainRhsSupportsDuals<1>){
        DualType<T, N, 1>::with_default_nvars(n,
            [&](){
                decltype(auto) out_duals = scratch.duals(); // n size
                ode_.Rhs(out_duals.data(), t, SeedVec<T, N, 1>{q, n, 1});
                std::fill(out+n, out+2*n, 0);
                for (size_t j=0; j<n; j++){
                    out[j] = out_duals[j].value();
                    for (size_t i=0; i<n; i++){
                        out[i+n] += out_duals[i].get_diff_wrt(j) * delta_q[j];
                    }
                }
            }
        );
    } else {

        ode_.Rhs(out, t, q); // Main system filled

        // First let's fill a temporary jacobian matrix
        decltype(auto) jm_worker = scratch.jacmat(); // n*n size
        T* mat = jm_worker.data();

        if constexpr (JP_MAIN == JacPolicy::Exact || JP_MAIN == JacPolicy::Nullable){
            if constexpr (JP_MAIN == JacPolicy::Exact){
                ode_.Jac(mat, t, q);
            } else if (ode_.Jac != nullptr) {
                ode_.Jac(mat, t, q);
            } else { // Finite differences fallback
                this->jacmat_findiffs(mat, t, q);
            }
        } else { // Finite differences fallback
            this->jacmat_findiffs(mat, t, q);
        }

        // The Jacobian matrix is now filled,
        // proceeding to fill the variational part
        T* delta_qdot = out + n;
        std::fill(delta_qdot, delta_qdot+n, 0);
        for (size_t j=0; j<n; j++){
            for (size_t i=0; i<n; i++){
                delta_qdot[i] += mat[j*n + i] * delta_q[j];
            }
        }
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys Jac
// ----------------------------------------------------------------------------

template<typename T, size_t N, hasRhsFunc<T> OdeType>
template<size_t Order>
requires (detail::FullJacSupportsDuals<T, N, OdeType, Order> && N > 0)
void VariationalOdeSys<T, N, OdeType>::Jac(DualType<T, 2*N, Order>* out, const T& t, SeedVec<T, 2*N, Order> q) const{
    /*
    As in the templated Rhs above, only the dual width is 2*N. The output is the
    (2*N x 2*N) Jacobian of the augmented system in F-storage, built from the
    (N x N) Jacobian of the main system and its derivatives:

        [        J                0 ]
        [ d(J)/dq * delta_q       J ]
    */
    using AugDual = DualType<T, 2*N, Order>;
    if constexpr (MainJacSupportsAugDuals<Order+1>){
        using AugDualHi = DualType<T, 2*N, Order+1>;
        std::array<AugDualHi, N*N> scratch_jacmat; // size n*n
        ode_.Jac(scratch_jacmat.data(), t, q.template with_order<Order+1>());
        MutView<AugDualHi, ndspan::Layout::F, N, N> m_in{scratch_jacmat.data()};
        MutView<AugDual, ndspan::Layout::F, 2*N, 2*N> m_out{out};
        for (size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                m_out(i, j) = m_out(i+N, j+N) = m_in(i, j).trimmed(); // upper left and lower right block
                m_out(i, j+N) = T{0}; // upper right block

                // lower left block: sum_k d(J_ik)/d(q_j) * delta_q_k.
                // The contracted index k is the *column* of J; the differentiation
                // is with respect to j, not the other way around.
                m_out(i+N, j) = T{0};
                for (size_t k=0; k<N; k++){
                    m_out(i+N, j) += m_in(i, k).trimmed_diff_wrt(j) * q[N+k];
                }
            }
        }
    } else {
        using DDual = DualType<T, 2*N, Order+2>;
        std::array<DDual, N> scratch_duals; // n for the rhs

        DDual* rhs = scratch_duals.data();
        ode_.Rhs(rhs, t, q.template with_order<Order+2>());

        MutView<AugDual, ndspan::Layout::F, 2*N, 2*N> m{out};
        for (size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                m(i, j) = m(i+N, j+N) = rhs[i].trimmed_diff_wrt(j).trimmed();
                m(i, j+N) = T{0};
                // d2(f_i)/(dq_k dq_j) == d(J_ik)/d(q_j) by symmetry of the Hessian
                m(i+N, j) = T{0};
                for (size_t k=0; k<N; k++){
                    m(i+N, j) += rhs[i].trimmed_diff_wrt(k, j) * q[N+k];
                }
            }
        }
    }
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::Jac(T* out, const T& t, const T* q) const{
    const size_t n = this->nsys_main();
    if constexpr (MainRhsSupportsDuals<2>){
        using DDual = DualType<T, N, 2>;
        DDual::with_default_nvars(n,
            [&](){
                decltype(auto) out_dduals = scratch.dduals(); // n size

                ode_.Rhs(out_dduals.data(), t, SeedVec<T, N, 2>{q, n, 2});

                MutView<T, ndspan::Layout::F, 2*N, 2*N> m(out, 2*n, 2*n);
                for (size_t i=0; i<n; i++){
                    for (size_t j=0; j<n; j++){
                        m(i, j) = m(i+n, j+n) = out_dduals[i].get_diff_wrt(j);
                        m(i, j+n) = 0;
                        T sum = 0;
                        for (size_t k=0; k<n; k++){
                            sum += out_dduals[i].get_diff_wrt(k, j) * q[n+k];
                        }
                        m(i+n, j) = sum;
                    }
                }
            }
        );
    } else {

        decltype(auto) jm_worker = scratch.jacmat(); // n*n size
        T* mat = jm_worker.data();
        if constexpr (JP_MAIN == JacPolicy::Exact || JP_MAIN == JacPolicy::Nullable){
            if constexpr (JP_MAIN == JacPolicy::Exact){
                ode_.Jac(mat, t, q);
            } else if (ode_.Jac != nullptr) {
                ode_.Jac(mat, t, q);
            } else { // Finite differences fallback
                this->jacmat_findiffs(mat, t, q);
            }
        } else { // Finite differences fallback
            this->jacmat_findiffs(mat, t, q);
        }

        // Fill the 2 main diagonal blocks
        for (size_t j=0; j<n; j++){
            for (size_t i=0; i<n; i++){
                out[2*j*n + i] = mat[j*n + i];
                out[2*n*n + 2*j*n + n + i] = mat[j*n + i];
                // fill the upper right block with zeros
                out[2*n*n + 2*j*n + i] = 0;
            }
        }

        // Now the main lower left block
        this->delta_J(mat, t, q, nullptr);
        for (size_t j=0; j<n; j++){
            for (size_t i=0; i<n; i++){
                out[n + 2*j*n + i] = mat[j*n + i];
            }
        }
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys finite differences
// ----------------------------------------------------------------------------

template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::jacmat_findiffs(T* mat, const T& t, const T* q) const{
    decltype(auto) worker = scratch.findiffs(); // 3*n size
    jac_approx<T>(
        [this](T* out_, const T& t_, const T* q_){
            this->ode_.Rhs(out_, t_, q_);
        },
        mat, worker.data(), t, q, nullptr, atol_, nsys_main()
    );
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::delta_J(T* mat, const T& t, const T* q, const T* dt) const{
    /*
    Requires 4 rhs evaluations per derivative, but only for the bottom left half
    So should be more efficient than using central finite differences on the
    full (augmented) system.
    */

    const T EPS = std::numeric_limits<T>::epsilon();
    // Second derivatives amplify roundoff by 1/h^2, so the sqrt(EPS) step used for
    // first derivatives (see jac_approx) would leave no signal at all here:
    // the roundoff error would be ~ EPS/h^2 = O(1). The optimal step for a
    // second difference balances truncation O(h^2) against roundoff O(EPS/h^2),
    // giving h ~ EPS^(1/4) and an error of ~ sqrt(EPS).
    const T EPS_4TH = sqrt(sqrt(EPS));

    const size_t n = this->nsys_main();
    decltype(auto) worker = scratch.findiffs(); // 3*n size

    T* x = worker.data();
    T* y1 = worker.data() + n;
    T* y2 = worker.data() + 2*n;

    std::fill(mat, mat + n*n, 0); // initialize the entire block to zero, since we will be adding to it in the loops below
    std::copy(q, q + n, x);
    const T* const delta_q = q + n;
    for (size_t j=0; j<n; j++){ // d/dq_j
        T* col = mat + j*n; // j-th column of the bottom left block

        // Now iterating over the index k,
        // which is contracted with delta_q[k] in the final result
        // for each (i, j) element of the bottom left block
        for (size_t k=0; k<n; k++){ // d/dq_k

            const T abs_qj = abs<T>(q[j]);
            const T abs_qk = abs<T>(q[k]);
            T h_j, h_k, h_sq;
            if (dt != nullptr){
                h_j = dt[j];
                h_k = dt[k];
                h_sq = h_j*h_k;
            } else {
                h_j = EPS_4TH * max_ref(atol_, abs_qj);
                h_k = EPS_4TH * max_ref(atol_, abs_qk);
                h_sq = h_j*h_k;
            }

            /*
            f[j-1, k+1]     f[j+1, k+1]
                ------------------
                |        |       |
                |        |       |
                |-----f[j,k]-----|
                |        |       |
                |        |       |
                ------------------
            f[j-1, k-1]     f[j+1, k-1]
            */

            if (j == k){
                // The 4-point stencil below degenerates when both perturbations
                // act on the same axis: it collapses to f[j+1] - f[j-1], which is a
                // first derivative, not a second one. The pure second derivative
                // needs the central 3-point stencil instead.
                x[j] = q[j] + h_j;
                ode_.Rhs(y1, t, x);

                x[j] = q[j] - h_j;
                ode_.Rhs(y2, t, x);
                for (size_t i=0; i<n; i++){
                    y1[i] += y2[i];
                }

                x[j] = q[j]; // unperturbed evaluation
                ode_.Rhs(y2, t, x);
                for (size_t i=0; i<n; i++){
                    // final value = delta_q[j] * d^2f^i/dx_j^2
                    col[i] += delta_q[j] * (y1[i] - 2*y2[i]) / h_sq;
                }
                continue; // x[j] is already restored
            }

            // Bottom left evaluation
            x[j] = q[j] - h_j;
            x[k] = q[k] - h_k;
            ode_.Rhs(y1, t, x);

            // Upper right evaluation
            x[j] = q[j] + h_j;
            x[k] = q[k] + h_k;
            ode_.Rhs(y2, t, x);

            // Combine them into one of the two temporaries
            for (size_t i=0; i<n; i++){
                y1[i] += y2[i];
            }

            // Upper left evaluation
            x[j] = q[j] - h_j;
            ode_.Rhs(y2, t, x);
            for (size_t i=0; i<n; i++){
                y1[i] -= y2[i];
            }

            // Lower right evaluation
            x[j] = q[j] + h_j;
            x[k] = q[k] - h_k;
            ode_.Rhs(y2, t, x);
            for (size_t i=0; i<n; i++){
                // final value = delta_q[k] * df^i/(dx_j dx_k)
                col[i] += delta_q[k] * (y1[i] - y2[i]) / (4*h_sq);
            }
            x[k] = q[k]; // reset x[k] to its original value
        }
        x[j] = q[j]; // reset x[j] to its original value
    }
}




template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
VariationalSolver<S, T, N, SP, OdeType, Derived>::VariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, Args&&... extra) : Base(VariationalOdeSys<T, N, OdeType>(ode, q0.size(),atol), t0,
    !q0.data() || !delta_q0.data() ?
    View1D<T, 2*N>{nullptr, 2*q0.size()} :
    View1D<T, 2*N>{
        join_arrays(q0, delta_q0).data(),
        2*q0.size()
    }, rtol, atol, min_step, max_step, stepsize, dir, std::forward<Args>(extra)...), worker(4*q0.size()), tmp_state_(2*q0.size()), period_(period), t_next_(t0+period*dir), t_last_(t0) {

    if (period <= 0){
        throw std::runtime_error("The renormalization period must be positive");
    }

    if constexpr (is_rich<SP>){
        //make sure there are no masked events, as they would interfere with the renormalization times.
        for (size_t i=0; i<this->event_col().size(); i++){
            if (this->event_col().event(i).is_masked()){
                throw std::runtime_error("VariationalSolver does not support masked events, as they would interfere with the renormalization times.");
            }
        }
    }
    
    const T* ics_vector = this->ics().vector();
    std::copy(ics_vector, ics_vector + 2*q0.size(), tmp_state_.data());

}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::elapsed_time() const{
    return this->t() - this->ics_ptr()[0];
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::stretching_number() const{
    const size_t nsys = this->nsys()/2;
    return log(norm(this->true_state_ptr()+2+nsys, nsys));
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::kick() const{
    return stretching_number()/(this->t() - t_last_);
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::period() const{
    return period_;
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::log_ksi() const{
    return logksi_ + this->stretching_number();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::lyapunov_exponent() const{
    return np == 0 ? T{0} : T(log_ksi()/elapsed_time());
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    const T* ics_vector = this->ics().vector();
    std::copy(ics_vector, ics_vector + this->nsys(), tmp_state_.data());
    t_last_ = this->ics_ptr()[0];
    t_next_ = t_last_ + period_*this->direction();
    np = 0;
    flagged = false;
    logksi_ =  0;
    logksi_last_ = 0;
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::RhsMain(T* out, const T& t, const T* q) const{
    this->ode().ode().Rhs(out, t, q); //fills the first half (nsys) entries
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::JacMain(T* out, const T& t, const T* q) const{
    // Mirrors the dispatch in VariationalOdeSys: an explicitly provided Jacobian is used
    // whenever it exists, and a Nullable one that is null at runtime falls back to
    // finite differences rather than being called.
    constexpr JacPolicy JP_MAIN = VariationalOdeSys<T, N, OdeType>::JP_MAIN;

    const auto& main_ode = this->ode().ode();

    if constexpr (JP_MAIN == JacPolicy::Exact){
        main_ode.Jac(out, t, q);
        return;
    } else if constexpr (JP_MAIN == JacPolicy::Nullable){
        if (main_ode.Jac != nullptr){
            main_ode.Jac(out, t, q);
            return;
        }
    }

    jac_approx<T>(
        [this](T* out_, const T& t_, const T* q_){
            this->RhsMain(out_, t_, q_);
        }, out, worker.data(), t, q, nullptr, this->atol(), this->nsys()/2
    );
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::ReAdjust(const T* /*new_vector*/){
    assert(false && "ReAdjust is not supported in VariationalSolver because it would interfere with the renormalization process.");
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
bool VariationalSolver<S, T, N, SP, OdeType, Derived>::Adv_Impl(Args&&... args) {
    if (flagged){
        Base::ReAdjust(tmp_state_.data());
        flagged = false;
    }

    const int d = this->direction();
    const bool success = Base::Adv_Impl(t_next_, std::forward<Args>(args)...);
    if (success && (this->t() == t_next_)){
        const size_t nsys = this->nsys()/2;
        t_last_ = t_next_;
        t_next_ = this->ics_ptr()[0] + (++np + 1UL)*period_*d;
        std::copy(THIS->true_state_ptr()+2, THIS->true_state_ptr()+2 + 2*nsys, tmp_state_.data());
        logksi_last_ = logksi_;
        logksi_ += log(norm(tmp_state_.data()+nsys, nsys));
        detail::normalized(tmp_state_.data(), tmp_state_.data(), nsys);
        flagged = true;
        return true;
    } else if (success){
        return true;
    } else {
        return false;
    }
}


// ----------------------------------------------------------------------------
// VIRTUAL INTERFACE ALIASES
// Overrides of the ChaoticSolver pure virtuals, forwarding to the non-virtual
// accessors above so that internal callers never pay for a virtual dispatch.
// ----------------------------------------------------------------------------

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::get_rhs_main(T* out, const T& t, const T* q) const{
    RhsMain(out, t, q);
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::get_jac_main(T* out, const T& t, const T* q) const{
    JacMain(out, t, q);
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::get_elapsed_time() const{
    return elapsed_time();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::get_kick() const{
    return kick();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::get_period() const{
    return period();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::get_log_ksi() const{
    return log_ksi();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::get_lyapunov_exponent() const{
    return lyapunov_exponent();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::get_stretching_number() const{
    return stretching_number();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Array1D<T, 2*N> VariationalSolver<S, T, N, SP, OdeType, Derived>::join_arrays(View1D<T, N> q0, View1D<T, N> delta_q0){
    assert(q0.size() == delta_q0.size() && "q0 and delta_q0 must have the same size");
    Array1D<T, 2*N> tmp(2*q0.size());
    std::copy(q0.data(), q0.data() + q0.size(), tmp.data());
    std::copy(delta_q0.data(), delta_q0.data() + delta_q0.size(), tmp.data()+q0.size());
    detail::normalized(tmp.data(), tmp.data(), q0.size());
    return tmp;
}


namespace detail{

template<typename T>
void normalized(T* out, const T* src, size_t nsys){
    T N = norm(src+nsys, nsys);
    for (size_t i=0; i<nsys; i++){
        out[i] = src[i];
        out[i+nsys] /= N;
    }
}

} // namespace ode::detail


template<typename T, size_t N>
template<hasRhsFunc<T> OdeType>
VariationalODE<T, N>::VariationalODE(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> events, Stepper method) : Base(2*q0.size()){
    assert(q0.size() == delta_q0.size() && "q0 and delta_q0 must have the same size in VariationalODE");
    // Must create solver BEFORE register_state(), since it accesses solver_
    this->solver_ = make_variational_solver<UtilPolicy::RichVirtual>(method, ode, t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, dir, std::move(events));

    const EventCollection<T>& event_coll = this->solver()->get_event_col();

    this->cached_idx_.resize(event_coll.size(), 0);
    Base::register_state();
    for (size_t i=0; i<event_coll.size(); i++){
        this->event_data_.allocate_event(event_coll.event(i).name());
    }
}

template<typename T, size_t N>
std::unique_ptr<ODE<T, N>> VariationalODE<T, N>::clone() const{
    return std::make_unique<VariationalODE<T, N>>(*this);
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::renorm_times() const{
    return renorm_times_;
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::lyap_values() const{
    return lyap_values_;
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::kick_values() const{
    return kick_values_;
}

template<typename T, size_t N>
void VariationalODE<T, N>::clear(){
    Base::clear();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, size_t N>
void VariationalODE<T, N>::reset(){
    Base::reset();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, size_t N>
const ChaoticSolver<T, N, UtilPolicy::RichVirtual>* VariationalODE<T, N>::solver() const {
    return static_cast<const ChaoticSolver<T, N, UtilPolicy::RichVirtual>*>(Base::solver());
}

template<typename T, size_t N>
void VariationalODE<T, N>::register_state(){
    Base::register_state();
    renorm_times_.push_back(this->solver()->get_time());
    lyap_values_.push_back(this->solver()->get_lyapunov_exponent());
    kick_values_.push_back(this->solver()->get_kick());
}


template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, 2*N, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, Args&&... args){

    constexpr SolverPolicy SP = UP == UtilPolicy::Virtual ? SolverPolicy::Virtual : SolverPolicy::RichVirtual;

    return choose_integrator_case<pbox::Box<ChaoticSolver<T, 2*N, UP>>>(method,
        [&]<Stepper S>(){
            using Solver = typename ::ode::detail::SolverTypeGetter<S, T, N, SP, OdeType>::type;
            return pbox::make_box<VariationalSolver<
                S,
                typename Solver::value_type,
                Solver::NSYS,
                SP,
                OdeType>>(std::move(ode), t0, q0, delta_q0, period, std::forward<Args>(args)...);
        }
    );
}


template<SolverPolicy SP, Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int direction){
    return VariationalSolver<S, T, N, SP, OdeType, void>(std::move(ode), t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, direction);
}


template<SolverPolicy SP, Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, EventList<T> events){
    return VariationalSolver<S, T, N, SP, OdeType, void>(std::move(ode), t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, direction, std::move(events));
}

} // namespace ode

#endif // ODECRAFT_VARIATIONAL_SOLVERS_IMPL_HPP
