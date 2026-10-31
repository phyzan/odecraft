#pragma once

#include "odecraft/Toolkit/Tools.hpp"
#include <odecraft/Steppers/DOPRI_impl.hpp>
#include <odecraft/Steppers/DOP853.hpp>

namespace ode{


// ============================================================================
// DOP_COEFS: raw Butcher-tableau data
// ============================================================================

template<typename T>
constexpr DOP_COEFS<T>::DOP_A DOP_COEFS<T>::make_A(){
    DOP_A res(N_STAGES_EXT, N_STAGES_EXT);
    res.fill(0);

    // Stage 1
    res(1, 0) = 5.26001519587677318785587544488e-2;

    // Stage 2
    res(2, 0) = 1.97250569845378994544595329183e-2;
    res(2, 1) = 5.91751709536136983633785987549e-2;

    // Stage 3
    res(3, 0) = 2.95875854768068491816892993775e-2;
    res(3, 2) = 8.87627564304205475450678981324e-2;

    // Stage 4
    res(4, 0) = 2.41365134159266685502369798665e-1;
    res(4, 2) = -8.84549479328286085344864962717e-1;
    res(4, 3) = 9.24834003261792003115737966543e-1;

    // Stage 5
    res(5, 0) = T(1)/T(27);
    res(5, 3) = 1.70828608729473871279604482173e-1;
    res(5, 4) = 1.25467687566822425016691814123e-1;

    // Stage 6
    res(6, 0) = T(19)/T(512);
    res(6, 3) = 1.70252211019544039314978060272e-1;
    res(6, 4) = 6.02165389804559606850219397283e-2;
    res(6, 5) = T(-9)/T(512);

    // Stage 7
    res(7, 0) = 3.70920001185047927108779319836e-2;
    res(7, 3) = 1.70383925712239993810214054705e-1;
    res(7, 4) = 1.07262030446373284651809199168e-1;
    res(7, 5) = -1.53194377486244017527936158236e-2;
    res(7, 6) = 8.27378916381402288758473766002e-3;

    // Stage 8
    res(8, 0) = 6.24110958716075717114429577812e-1;
    res(8, 3) = -3.36089262944694129406857109825;
    res(8, 4) = -8.68219346841726006818189891453e-1;
    res(8, 5) = 2.75920996994467083049415600797e1;
    res(8, 6) = 2.01540675504778934086186788979e1;
    res(8, 7) = -4.34898841810699588477366255144e1;

    // Stage 9
    res(9, 0) = 4.77662536438264365890433908527e-1;
    res(9, 3) = -2.48811461997166764192642586468;
    res(9, 4) = -5.90290826836842996371446475743e-1;
    res(9, 5) = 2.12300514481811942347288949897e1;
    res(9, 6) = 1.52792336328824235832596922938e1;
    res(9, 7) = -3.32882109689848629194453265587e1;
    res(9, 8) = -2.03312017085086261358222928593e-2;

    // Stage 10
    res(10, 0) = -9.3714243008598732571704021658e-1;
    res(10, 3) = 5.18637242884406370830023853209;
    res(10, 4) = 1.09143734899672957818500254654;
    res(10, 5) = -8.14978701074692612513997267357;
    res(10, 6) = -1.85200656599969598641566180701e1;
    res(10, 7) = 2.27394870993505042818970056734e1;
    res(10, 8) = 2.49360555267965238987089396762;
    res(10, 9) = -3.0467644718982195003823669022;

    // Stage 11
    res(11, 0) = 2.27331014751653820792359768449;
    res(11, 3) = -1.05344954667372501984066689879e1;
    res(11, 4) = -2.00087205822486249909675718444;
    res(11, 5) = -1.79589318631187989172765950534e1;
    res(11, 6) = 2.79488845294199600508499808837e1;
    res(11, 7) = -2.85899827713502369474065508674;
    res(11, 8) = -8.87285693353062954433549289258;
    res(11, 9) = 1.23605671757943030647266201528e1;
    res(11, 10) = 6.43392746015763530355970484046e-1;

    // Stage 12 (this is B coefficients, but part of extended A matrix)
    res(12, 0) = 5.42937341165687622380535766363e-2;
    res(12, 5) = 4.45031289275240888144113950566;
    res(12, 6) = 1.89151789931450038304281599044;
    res(12, 7) = -5.8012039600105847814672114227;
    res(12, 8) = 3.1116436695781989440891606237e-1;
    res(12, 9) = -1.52160949662516078556178806805e-1;
    res(12, 10) = 2.01365400804030348374776537501e-1;
    res(12, 11) = 4.47106157277725905176885569043e-2;

    // Stage 13
    res(13, 0) = 5.61675022830479523392909219681e-2;
    res(13, 6) = 2.53500210216624811088794765333e-1;
    res(13, 7) = -2.46239037470802489917441475441e-1;
    res(13, 8) = -1.24191423263816360469010140626e-1;
    res(13, 9) = 1.5329179827876569731206322685e-1;
    res(13, 10) = 8.20105229563468988491666602057e-3;
    res(13, 11) = 7.56789766054569976138603589584e-3;
    res(13, 12) = -8.298e-3;

    // Stage 14
    res(14, 0) = 3.18346481635021405060768473261e-2;
    res(14, 5) = 2.83009096723667755288322961402e-2;
    res(14, 6) = 5.35419883074385676223797384372e-2;
    res(14, 7) = -5.49237485713909884646569340306e-2;
    res(14, 10) = -1.08347328697249322858509316994e-4;
    res(14, 11) = 3.82571090835658412954920192323e-4;
    res(14, 12) = -3.40465008687404560802977114492e-4;
    res(14, 13) = 1.41312443674632500278074618366e-1;

    // Stage 15
    res(15, 0) = -4.28896301583791923408573538692e-1;
    res(15, 5) = -4.69762141536116384314449447206;
    res(15, 6) = 7.68342119606259904184240953878;
    res(15, 7) = 4.06898981839711007970213554331;
    res(15, 8) = 3.56727187455281109270669543021e-1;
    res(15, 12) = -1.39902416515901462129418009734e-3;
    res(15, 13) = 2.9475147891527723389556272149;
    res(15, 14) = -9.15095847217987001081870187138;

    return res;
}

template<typename T>
constexpr DOP_COEFS<T>::DOP_B DOP_COEFS<T>::make_B(){
    // B coefficients are row 12 of the A matrix (the 8th order solution weights)
    DOP_A a = make_A();
    DOP_B res(N_STAGES);
    for(size_t i = 0; i < N_STAGES; ++i){
        res(i) = a(12, i);
    }
    return res;
}

template<typename T>
constexpr DOP_COEFS<T>::DOP_C DOP_COEFS<T>::make_C(){
    DOP_C res(N_STAGES_EXT);
    res(0) = 0;
    res(1) = 0.526001519587677318785587544488e-01;
    res(2) = 0.789002279381515978178381316732e-01;
    res(3) = 0.118350341907227396726757197510;
    res(4) = 0.281649658092772603273242802490;
    res(5) = T(1)/T(3);
    res(6) = T(1)/T(4);
    res(7) = T(4)/T(13);
    res(8) = T(127)/T(195);
    res(9) = T(3)/T(5);
    res(10) = T(6)/T(7);
    res(11) = 1.0;
    res(12) = 1.0;
    res(13) = 0.1;
    res(14) = 0.2;
    res(15) = T(7)/T(9);
    return res;
}

template<typename T>
constexpr DOP_COEFS<T>::DOP_E DOP_COEFS<T>::make_E3(){
    DOP_E res(N_STAGES + 1);
    // E3 = B - B3 (3rd order embedded method)
    // Most values are zero, only differences are specified
    res(0) = 5.42937341165687622380535766363e-2 - 0.244094488188976377952755905512;
    res(1) = 0;
    res(2) = 0;
    res(3) = 0;
    res(4) = 0;
    res(5) = 4.45031289275240888144113950566;
    res(6) = 1.89151789931450038304281599044;
    res(7) = -5.8012039600105847814672114227;
    res(8) = 3.1116436695781989440891606237e-1 - 0.733846688281611857341361741547;
    res(9) = -1.52160949662516078556178806805e-1;
    res(10) = 2.01365400804030348374776537501e-1;
    res(11) = 4.47106157277725905176885569043e-2 - 0.220588235294117647058823529412e-1;
    res(12) = 0;
    return res;
}

template<typename T>
constexpr DOP_COEFS<T>::DOP_E DOP_COEFS<T>::make_E5(){
    DOP_E res(N_STAGES + 1);
    res.fill(0);
    // E5 = B - B5 (5th order embedded method)
    res(0) = 0.1312004499419488073250102996e-1;
    res(5) = -0.1225156446376204440720569753e+1;
    res(6) = -0.4957589496572501915214079952;
    res(7) = 0.1664377182454986536961530415e+1;
    res(8) = -0.3503288487499736816886487290;
    res(9) = 0.3341791187130174790297318841;
    res(10) = 0.8192320648511571246570742613e-1;
    res(11) = -0.2235530786388629525884427845e-1;
    return res;
}

template<typename T>
constexpr DOP_COEFS<T>::DOP_D DOP_COEFS<T>::make_D(){
    DOP_D res(INTERP_ORDER - 3, N_STAGES_EXT);
    res.fill(0);

    // D[0, :] - first interpolation coefficients
    res(0, 0) = -0.84289382761090128651353491142e+1;
    res(0, 5) = 0.56671495351937776962531783590;
    res(0, 6) = -0.30689499459498916912797304727e+1;
    res(0, 7) = 0.23846676565120698287728149680e+1;
    res(0, 8) = 0.21170345824450282767155149946e+1;
    res(0, 9) = -0.87139158377797299206789907490;
    res(0, 10) = 0.22404374302607882758541771650e+1;
    res(0, 11) = 0.63157877876946881815570249290;
    res(0, 12) = -0.88990336451333310820698117400e-1;
    res(0, 13) = 0.18148505520854727256656404962e+2;
    res(0, 14) = -0.91946323924783554000451984436e+1;
    res(0, 15) = -0.44360363875948939664310572000e+1;

    // D[1, :] - second interpolation coefficients
    res(1, 0) = 0.10427508642579134603413151009e+2;
    res(1, 5) = 0.24228349177525818288430175319e+3;
    res(1, 6) = 0.16520045171727028198505394887e+3;
    res(1, 7) = -0.37454675472269020279518312152e+3;
    res(1, 8) = -0.22113666853125306036270938578e+2;
    res(1, 9) = 0.77334326684722638389603898808e+1;
    res(1, 10) = -0.30674084731089398182061213626e+2;
    res(1, 11) = -0.93321305264302278729567221706e+1;
    res(1, 12) = 0.15697238121770843886131091075e+2;
    res(1, 13) = -0.31139403219565177677282850411e+2;
    res(1, 14) = -0.93529243588444783865713862664e+1;
    res(1, 15) = 0.35816841486394083752465898540e+2;

    // D[2, :] - third interpolation coefficients
    res(2, 0) = 0.19985053242002433820987653617e+2;
    res(2, 5) = -0.38703730874935176555105901742e+3;
    res(2, 6) = -0.18917813819516756882830838328e+3;
    res(2, 7) = 0.52780815920542364900561016686e+3;
    res(2, 8) = -0.11573902539959630126141871134e+2;
    res(2, 9) = 0.68812326946963000169666922661e+1;
    res(2, 10) = -0.10006050966910838403183860980e+1;
    res(2, 11) = 0.77771377980534432092869265740;
    res(2, 12) = -0.27782057523535084065932004339e+1;
    res(2, 13) = -0.60196695231264120758267380846e+2;
    res(2, 14) = 0.84320405506677161018159903784e+2;
    res(2, 15) = 0.11992291136182789328035130030e+2;

    // D[3, :] - fourth interpolation coefficients
    res(3, 0) = -0.25693933462703749003312586129e+2;
    res(3, 5) = -0.15418974869023643374053993627e+3;
    res(3, 6) = -0.23152937917604549567536039109e+3;
    res(3, 7) = 0.35763911791061412378285349910e+3;
    res(3, 8) = 0.93405324183624310003907691704e+2;
    res(3, 9) = -0.37458323136451633156875139351e+2;
    res(3, 10) = 0.10409964950896230045147246184e+3;
    res(3, 11) = 0.29840293426660503123344363579e+2;
    res(3, 12) = -0.43533456590011143754432175058e+2;
    res(3, 13) = 0.96324553959188282948394950600e+2;
    res(3, 14) = -0.39177261675615439165231486172e+2;
    res(3, 15) = -0.14972683625798562581422125276e+3;

    return res;
}

namespace detail{

template<typename T, typename SD>
void coef_mat_interp_dop853(state_t<T, SD>& result, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1,
                            const std::array<state_t<T, SD>, DOP_COEFS<T>::INTERP_ORDER>& F){
    // The Horner-like scheme from scipy, alternating theta and (1 - theta):
    // y = y1 + x*(F0 + (1-x)*(F1 + x*(F2 + (1-x)*(F3 + x*(F4 + (1-x)*(F5 + x*F6))))))
    assert( (t != t1 && t != t2) && "coef_mat_interp_dop853 must only be called on the open interval");
    using S = value_t<T, SD>;

    const T h = t2 - t1;
    const T x = (t - t1) / h;  // theta
    const T one_minus_x = 1 - x;

    SD::for_each([&](S& r_i, const S& y1_i, const S& F0, const S& F1, const S& F2, const S& F3, const S& F4, const S& F5, const S& F6){
        S r = F6 * x;
        r = (r + F5) * one_minus_x;
        r = (r + F4) * x;
        r = (r + F3) * one_minus_x;
        r = (r + F2) * x;
        r = (r + F1) * one_minus_x;
        r = (r + F0) * x;
        r_i = r + y1_i;
    }, result, y1, F[0], F[1], F[2], F[3], F[4], F[5], F[6]);
}

template<typename T, typename SD, typename Etab>
T dop853_error_norm(const std::array<state_t<T, SD>, DOP_COEFS<T>::N_STAGES_EXT>& K, const Etab& E3, const Etab& E5,
                    const state_t<T, SD>& q, const state_t<T, SD>& q_new, cache_t<T> h,
                    const real_state_t<T, SD>& rtol, const real_state_t<T, SD>& atol){
    // Combination of 3rd and 5th order embedded error estimates (Hairer, Norsett & Wanner):
    // error_norm = |h| * ||err5||_2 / sqrt(||err5||_2^2 + 0.01 * ||err3||_2^2) / sqrt(n)
    using S = value_t<T, SD>;
    T err5_norm_2 = 0;
    T err3_norm_2 = 0;
    size_t n = 0;

    // E3 and E5 vanish but for the stages 0 and 5..11
    [&]<size_t... k>(std::index_sequence<k...>){
        SD::for_each([&](const S& q_j, const S& q_new_j, const T& rtol_j, const T& atol_j, const auto&... K_kj){
            const T scale = atol_j + rtol_j * ndspan::max<T>(abs(q_j), abs(q_new_j));
            const S err5 = (S(0) + ... + (K_kj * E5[k])) / scale;
            const S err3 = (S(0) + ... + (K_kj * E3[k])) / scale;
            err5_norm_2 += SD::norm_sqr(err5);
            err3_norm_2 += SD::norm_sqr(err3);
            ++n;
        }, q, q_new, rtol, atol, K[k]...);
    }(stages<0, 5, 6, 7, 8, 9, 10, 11>);

    if (err5_norm_2 == 0 && err3_norm_2 == 0){
        return 0;
    }

    const auto denom = err5_norm_2 + err3_norm_2 / 100;
    return abs(h) * err5_norm_2 / sqrt(denom * n);
}

template<typename T, typename SD, typename Atab, typename Ctab, typename Etab, typename RhsFn>
XDIFF_FORCEINLINE T dop853_step_impl(T& t_new, state_t<T, SD>& q_new, cache_t<T> t, cache_t<T> h, const state_t<T, SD>& q,
                   std::array<state_t<T, SD>, DOP_COEFS<T>::N_STAGES_EXT>& K, state_t<T, SD>& r,
                   const real_state_t<T, SD>& rtol, const real_state_t<T, SD>& atol,
                   const Atab& A, const Ctab& C, const Etab& E3, const Etab& E5, RhsFn&& rhs){
    // Stage i: r = q + h * sum_j A(i, j) K[j] over the j with a nonzero A(i, j), then K[i] = f(t + C[i] h, r)
    rk_stage<SD>(r, q, h, K, A, 1,  stages<0>);                                 rhs(K[1],  t + C[1]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 2,  stages<0, 1>);                              rhs(K[2],  t + C[2]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 3,  stages<0, 2>);                              rhs(K[3],  t + C[3]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 4,  stages<0, 2, 3>);                           rhs(K[4],  t + C[4]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 5,  stages<0, 3, 4>);                           rhs(K[5],  t + C[5]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 6,  stages<0, 3, 4, 5>);                        rhs(K[6],  t + C[6]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 7,  stages<0, 3, 4, 5, 6>);                     rhs(K[7],  t + C[7]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 8,  stages<0, 3, 4, 5, 6, 7>);                  rhs(K[8],  t + C[8]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 9,  stages<0, 3, 4, 5, 6, 7, 8>);               rhs(K[9],  t + C[9]*h,  r);
    rk_stage<SD>(r, q, h, K, A, 10, stages<0, 3, 4, 5, 6, 7, 8, 9>);            rhs(K[10], t + C[10]*h, r);
    rk_stage<SD>(r, q, h, K, A, 11, stages<0, 3, 4, 5, 6, 7, 8, 9, 10>);        rhs(K[11], t + h,       r);

    // Solution update: row 12 of A holds the weights b (b1 = b2 = b3 = b4 = 0)
    rk_stage<SD>(q_new, q, h, K, A, 12, stages<0, 5, 6, 7, 8, 9, 10, 11>);

    // FSAL: K[12] = f(t+h, q_new), also used for error estimation and the next step
    rhs(K[12], t + h, q_new);
    t_new = t + h;

    return dop853_error_norm<T, SD>(K, E3, E5, q, q_new, h, rtol, atol);
}

} // namespace ode::detail


// ============================================================================
// DOP853
// ============================================================================

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
DOP853<T, SD, JD, OdeType, SP, Derived>::DOP853(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int direction, EventList<T, State> events)
    : Base(std::move(ode), t0, q0, std::move(rtol), std::move(atol), min_step, max_step, stepsize, direction, std::move(events)),
      r_scratch_(q0) {
    // the arrays of states take the shape of q0 too
    K_.fill(q0);
    coef_mat_.fill(q0);
    this->rhs(K_[N_STAGES], t0, q0);
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
Stepper DOP853<T, SD, JD, OdeType, SP, Derived>::method() const{
    return Stepper::DOP853;
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void DOP853<T, SD, JD, OdeType, SP, Derived>::Reset(){
    Base::Reset();
    this->rhs(K_[N_STAGES], this->t_new(), this->vector_new());
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void DOP853<T, SD, JD, OdeType, SP, Derived>::ReAdjust(const State& new_vector){
    Base::ReAdjust(new_vector);
    // the first stage of the next step (FSAL)
    this->rhs(K_[N_STAGES], this->t(), new_vector);
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void DOP853<T, SD, JD, OdeType, SP, Derived>::cache_coef_matrix() const{
    using S = value_t<T, SD>;
    if (coef_mat_is_cached_){
        return;
    }

    // The step size that was actually used, not stepsize(): that returns the newest step's habs,
    // which rk_adapt_step has already grown/shrunk into the *next* step's proposed size, and it
    // also differs from the accepted h whenever an attempt was rejected.
    const T h = h_last_;
    const T& t_old = this->t_old();
    const State& y_old = this->vector_old();
    const State& y_new = this->vector_new();
    decltype(auto) r = r_scratch_.get();

    // Three extra stages, evaluated only for dense-output interpolation.
    detail::rk_stage<SD>(r, y_old, h, K_, A, 13, detail::stages<0, 6, 7, 8, 9, 10, 11, 12>);
    this->rhs(K_[13], t_old + C[13]*h, r);
    detail::rk_stage<SD>(r, y_old, h, K_, A, 14, detail::stages<0, 5, 6, 7, 10, 11, 12, 13>);
    this->rhs(K_[14], t_old + C[14]*h, r);
    detail::rk_stage<SD>(r, y_old, h, K_, A, 15, detail::stages<0, 5, 6, 7, 8, 12, 13, 14>);
    this->rhs(K_[15], t_old + C[15]*h, r);

    // F[0..2], from the end points of the step and the derivatives there: K[0] and K[12]
    SD::for_each([&h](S& F0, S& F1, S& F2, const S& y_old_i, const S& y_new_i, const S& f_old_i, const S& f_new_i){
        F0 = y_new_i - y_old_i;                                     // delta_y
        F1 = h*f_old_i - (y_new_i - y_old_i);                       // h*f_old - delta_y
        F2 = T(2)*(y_new_i - y_old_i) - h*(f_new_i + f_old_i);
    }, coef_mat_[0], coef_mat_[1], coef_mat_[2], y_old, y_new, K_[0], K_[N_STAGES]);

    // F[3:] = h * D @ K, where D vanishes but for the stages 0 and 5..15
    for (size_t i = 0; i < INTERP_ORDER - 3; i++){
        detail::rk_dot<SD>(coef_mat_[3 + i], h, K_, D, i, detail::stages<0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>);
    }
    coef_mat_is_cached_ = true;
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
T DOP853<T, SD, JD, OdeType, SP, Derived>::step_impl(T& t_new, State& q_new, const T& t, const T& h, const State& q){
    decltype(auto) rhs_scratch = r_scratch_.get();
    h_last_ = h; // remembered for the dense-output coefficients
    return detail::dop853_step_impl<T, SD>(
        t_new,
        q_new,
        t,
        h,
        q,
        K_,
        rhs_scratch,
        this->rtol(),
        this->atol(),
        A, C, E3, E5,
        [this](State& out, const T& time, const State& y){
            this->rhs(out, time, y);
        }
    );
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
StepResult DOP853<T, SD, JD, OdeType, SP, Derived>::adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q){
    coef_mat_is_cached_ = false; // K_ is about to change
    // FSAL: the previous step's last stage becomes this step's first
    copy_state<SD>(K_[0], K_[N_STAGES]);
    return detail::rk_adapt_step<T, SD>(
        t_new, habs_new, q_new, t, habs, q,
        this->min_step(),
        this->max_step(),
        this->tiny_factor(),
        this->safety_factor(),
        this->max_factor(),
        this->min_factor(),
        ERR_EXP,
        INC_EXP,
        MIN_ERR,
        this->direction(),
        [this](T& t_out, State& q_out, const T& t_in, const T& h, const State& q_in){
            return this->step_impl(t_out, q_out, t_in, h, q_in);
        }
    );
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
auto DOP853<T, SD, JD, OdeType, SP, Derived>::local_interp_impl() const{
    this->cache_coef_matrix();
    return make_light_functor([](State& out, const T& t, const auto& t1, const auto& t2, const auto& y1, const auto& cm){
        return detail::coef_mat_interp_dop853<T, SD>(out, t, t1, t2, y1, cm);
    }, this->t_old(), this->t_new(), this->vector_old(), this->coef_mat_);
}

} // namespace ode
