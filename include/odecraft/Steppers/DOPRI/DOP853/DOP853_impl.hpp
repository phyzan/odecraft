#pragma once

#include "odecraft/Toolkit/Tools.hpp"
#include "../DOPRI_impl.hpp"
#include "DOP853.hpp"

namespace ode{


// ============================================================================
// DOP_COEFS: the Butcher tableau, exactly
// ============================================================================
//
// Hairer's dop853.f gives these constants as decimals of 26 to 30 digits, which would cap a
// multiprecision T near 1e-28. Here every one is exact: rational, or (p + q sqrt 6)/n. They were
// worked out in exact arithmetic over Q(sqrt 6) from the conditions the method is built on, as
// below. The inputs are Hairer's free choices: the nodes, which entries vanish, and eight
// constants marked "chosen". The decimals give those eight as rationals with denominators of at
// most 11 digits, recovered by continued fractions. Every derived constant agrees with the
// decimals to all of their digits, and the derived constants that the decimals also give as such
// rationals are those rationals exactly, which confirms the inputs.
//
// Notation. Stages are numbered from 0, as in the code, one below Hairer's numbering: stage i
// evaluates K[i] at t + c(i) h from row i of A. The stage defect
//     d_i(k) = sum_j a(i,j) c(j)^(k-1) - c(i)^k / k
// vanishes for k = 1..q when stage i integrates every polynomial of degree below q exactly. The
// order conditions are Butcher's, one per rooted tree t: there are 1, 1, 2, 4, 9, 20, 48 and 115
// trees with 1 to 8 vertices. The stage weights of t are Phi_i(t) = 1 if t is a single vertex,
// and otherwise the product over the subtrees s at its root of sum_j a(i,j) Phi_j(s). Weights w
// have order p when sum_i w(i) Phi_i(t) = 1/gamma(t), gamma(t) being the density of t, for every
// tree of at most p vertices.
//
// 1. Nodes (chosen). c(3) = (6 - sqrt 6)/30, c(4) = (6 + sqrt 6)/30, c(2) = 2 c(3)/3,
//    c(1) = 2 c(2)/3, and c(5..15) = 1/3, 1/4, 4/13, 127/195, 3/5, 6/7, 1, 1, 1/10, 1/5, 7/9.
//
// 2. b, the 8th order solution. It is nonzero on stages 0 and 5..11 only, and there it is the
//    quadrature of degree 7 on their nodes: sum_i b(i) c(i)^(k-1) = 1/k for k = 1..8.
//
// 3. Rows 1..11. Their 50 nonzero entries are subject to 55 linear conditions:
//    - a row with n entries integrates exactly up to degree min(n, 5) - 1: d_i(k) = 0 for
//      k = 1..min(n, 5);
//    - stages 3 and 4, which have no weight, satisfy D(3): sum_i b(i) c(i)^(k-1) a(i,j) = 0
//      for k = 1, 2, 3;
//    - stages 5..10 satisfy D(1): sum_i b(i) a(i,j) = b(j) (1 - c(j));
//    - three order conditions that are linear in the entries: sum_i b(i) d_i(6) = 0 (order 7),
//      and sum_i b(i) c(i) d_i(6) = sum_i b(i) d_i(7) = 0 (order 8).
//    These have rank 49, which leaves a(11,7) free. An order 8 condition that is linear in it
//    gives a(11,7) = -2509602342/877790785. All 200 conditions up to order 8 then hold (checked
//    at 80 digits), and order 9 does not. In rows 5..11, the entries on stages 3 and 4
//    come out as conjugate pairs (p +- q sqrt 6)/n, and the others are rational.
//
// 4. E3 = b - bhh. The 3rd order solution bhh is the quadrature of degree 2 on the nodes of
//    stages 0, 8 and 11: bhh = 31/127, 12675/17272 and 3/136.
//
// 5. E5 = b - bhat. The 5th order solution bhat has the stages of b, and is a quadrature of
//    degree 4 on them, which leaves three of its eight weights free. Those are chosen:
//    bhat(5) = 8670301/1527680, bhat(6) = 22407563/9386244 and bhat(11) = 137909/2056320, which
//    is 3 b(11)/2. Since b, bhh and bhat each sum to 1, E3 and E5 sum to exactly 0, so a step on
//    which f is constant estimates no error. The decimals of E5 sum to -4.6e-28 instead.
//
// 6. Dense output. Stage 12 evaluates f at the new solution, so its row is b, and stages 13..15
//    are evaluated for the dense output alone. At x = (t - t_old)/h,
//        u(x) = y_old + x F0 + x(1-x) F1 + x^2(1-x) F2
//                     + x^2(1-x)^2 (F3 + x F4 + x(1-x) F5 + x^2(1-x) F6),
//    where F0..F2 come from y_old, y_new, f_old and f_new, and F3..F6 = h D K over stages 0 and
//    5..15 (see coef_mat_interp_dop853). With beta_j(x) the weight of K[j] in u(x), u has order 7
//    when sum_j beta_j(x) Phi_j(t) = x^|t| / gamma(t) for every x and each of the 85 trees of at
//    most 7 vertices.
//    - Rows 13 and 15 satisfy C(6): d_i(k) = 0 for k = 1..6. That fixes six of their eight
//      entries, and the two others are chosen: a(13,9) = 57799439/377055000,
//      a(13,12) = -4149/500000, a(15,13) = 157882067000/53564469831 and
//      a(15,14) = -290468882375/31741908048.
//    - Row 14 satisfies C(6) as well, with a(14,13) = 1912306948/13532473845 chosen. Stages 11
//      and 12 share the node 1, so C(6) only fixes a(14,11) + a(14,12).
//    - For each tree, x^|t| / gamma(t) minus the F0..F2 part is x^2(1-x)^2 times a cubic, by the
//      order of b. The F3..F6 part must equal it, which is one linear condition on each row of D.
//      Each row of D thus has 85 conditions on its 12 entries, of rank 12. They are consistent
//      only for a(14,12) = -11518334563/33831184612500, and then D is unique, and rational.
//    The dense output then satisfies every condition up to order 7 (checked at 60 digits, as
//    coef_mat_interp_dop853 evaluates it), and not those of order 8.
//
// Each table is worked out in dop853_work_t<T>, then rounded to T once. For the built-in floating
// types that is long double: where it is wider than double, as on x86, their constants come out
// correctly rounded, as the decimals did.

namespace detail{

/// @brief Where DOP853's constants are worked out before they are rounded to T: long double for
/// the built-in floating types, and T itself for any other.
template<typename T>
using dop853_work_t = std::conditional_t<std::is_floating_point_v<T>, long double, T>;

/// @brief n/d in T. Every n and d below fits in 63 bits, so for a T with at least 64 bits of
/// precision this is correctly rounded.
template<typename T>
T fraction(long long n, long long d){
    return T(n) / T(d);
}

/// @brief The table `work` of DOP_COEFS<dop853_work_t<T>>, with each entry rounded to T.
template<typename T, typename Table, typename Work>
Table rounded_table(const Work& work){
    if constexpr (std::is_same_v<Table, Work>){
        return work;
    } else {
        Table res;
        for (size_t k = 0; k < work.size(); ++k){
            res.data()[k] = T(work.data()[k]);
        }
        return res;
    }
}

} // namespace ode::detail

template<typename T>
constexpr typename DOP_COEFS<T>::DOP_A DOP_COEFS<T>::make_A(){
    using W = detail::dop853_work_t<T>;
    const auto q = detail::fraction<W>;
    const W r6 = sqrt(W(6));
    const auto c = DOP_COEFS<W>::make_C();

    typename DOP_COEFS<W>::DOP_A res(N_STAGES_EXT, N_STAGES_EXT);
    res.fill(0);

    // Stages 1..11: step 3 above
    // Stage 1
    res(1, 0) = c(1);

    // Stage 2
    res(2, 0) = c(2)/W(4);
    res(2, 1) = W(3)*c(2)/W(4);

    // Stage 3
    res(3, 0) = c(3)/W(4);
    res(3, 2) = W(3)*c(3)/W(4);

    // Stage 4
    res(4, 0) = (W(462) + W(107)*r6)/W(3000);
    res(4, 2) = (W(-402) - W(197)*r6)/W(1000);
    res(4, 3) = (W(168) + W(73)*r6)/W(375);

    // Stage 5
    res(5, 0) = q(1, 27);
    res(5, 3) = (W(16) + r6)/W(108);
    res(5, 4) = (W(16) - r6)/W(108);

    // Stage 6
    res(6, 0) = q(19, 512);
    res(6, 3) = (W(118) + W(23)*r6)/W(1024);
    res(6, 4) = (W(118) - W(23)*r6)/W(1024);
    res(6, 5) = q(-9, 512);

    // Stage 7
    res(7, 0) = q(13772, 371293);
    res(7, 3) = (W(51544) + W(4784)*r6)/W(371293);
    res(7, 4) = (W(51544) - W(4784)*r6)/W(371293);
    res(7, 5) = q(-5688, 371293);
    res(7, 6) = q(3072, 371293);

    // Stage 8
    res(8, 0) = q(58656157643, 93983540625);
    res(8, 3) = (W(-1324889724104) - W(318801444819)*r6)/W(626556937500);
    res(8, 4) = (W(-1324889724104) + W(318801444819)*r6)/W(626556937500);
    res(8, 5) = q(96044563816, 3480871875);
    res(8, 6) = q(5682451879168, 281950621875);
    res(8, 7) = q(-165125654, 3796875);

    // Stage 9
    res(9, 0) = q(8909899, 18653125);
    res(9, 3) = (W(-4521408) - W(1137963)*r6)/W(2937500);
    res(9, 4) = (W(-4521408) + W(1137963)*r6)/W(2937500);
    res(9, 5) = q(96663078, 4553125);
    res(9, 6) = q(2107245056, 137915625);
    res(9, 7) = q(-4913652016, 147609375);
    res(9, 8) = q(-78894270, 3880452869);

    // Stage 10
    res(10, 0) = q(-20401265806, 21769653311);
    res(10, 3) = (W(354216) + W(94326)*r6)/W(112847);
    res(10, 4) = (W(354216) - W(94326)*r6)/W(112847);
    res(10, 5) = q(-43306765128, 5313852383);
    res(10, 6) = q(-20866708358144, 1126708119789);
    res(10, 7) = q(14886003438020, 654632330667);
    res(10, 8) = q(35290686222309375, 14152473387134411);
    res(10, 9) = q(-1477884375, 485066827);

    // Stage 11: a(11,7) from an order 8 condition
    res(11, 0) = q(39815761, 17514443);
    res(11, 3) = (W(-3457480) - W(960905)*r6)/W(551636);
    res(11, 4) = (W(-3457480) + W(960905)*r6)/W(551636);
    res(11, 5) = q(-844554132, 47026969);
    res(11, 6) = q(8444996352, 302158619);
    res(11, 7) = q(-2509602342, 877790785);
    res(11, 8) = q(-28388795297996250, 3199510091356783);
    res(11, 9) = q(226716250, 18341897);
    res(11, 10) = q(1371316744, 2131383595);

    // Stage 12 evaluates f at the new solution, whose row is b (FSAL)
    const auto b = DOP_COEFS<W>::make_B();
    for (size_t j = 0; j < N_STAGES; ++j){
        res(12, j) = b(j);
    }

    // Stages 13..15, for dense output only: step 6 above
    // Stage 13: C(6), with a(13,9) and a(13,12) chosen
    res(13, 0) = q(13481885573, 240030000000);
    res(13, 6) = q(139418837528, 549975234375);
    res(13, 7) = q(-11108320068443, 45111937500000);
    res(13, 8) = q(-1769651421925959, 14249385146080000);
    res(13, 9) = q(57799439, 377055000);
    res(13, 10) = q(793322643029, 96734250000000);
    res(13, 11) = q(1458939311, 192780000000);
    res(13, 12) = q(-4149, 500000);

    // Stage 14: C(6), with a(14,13) chosen, and a(14,12) fixed by the order 7 conditions
    res(14, 0) = q(1595561272731, 50120273500000);
    res(14, 5) = q(975183916491, 34457688031250);
    res(14, 6) = q(38492013932672, 718912673015625);
    res(14, 7) = q(-1114881286517557, 20298710767500000);
    res(14, 10) = q(-2538710946863, 23431227861250000);
    res(14, 11) = q(8824659001, 23066716781250);
    res(14, 12) = q(-11518334563, 33831184612500);
    res(14, 13) = q(1912306948, 13532473845);

    // Stage 15: C(6), with a(15,13) and a(15,14) chosen
    res(15, 0) = q(-13613986967, 31741908048);
    res(15, 5) = q(-4755612631, 1012344804);
    res(15, 6) = q(42939257944576, 5588559685701);
    res(15, 7) = q(77881972900277, 19140370552944);
    res(15, 8) = q(22719829234375, 63689648654052);
    res(15, 12) = q(-1199007803, 857031517296);
    res(15, 13) = q(157882067000, 53564469831);
    res(15, 14) = q(-290468882375, 31741908048);

    return detail::rounded_table<T, DOP_A>(res);
}

template<typename T>
constexpr typename DOP_COEFS<T>::DOP_B DOP_COEFS<T>::make_B(){
    // The weights of the 8th order solution: step 2 above
    using W = detail::dop853_work_t<T>;
    const auto q = detail::fraction<W>;
    typename DOP_COEFS<W>::DOP_B res(N_STAGES);
    res.fill(0);
    res(0) = q(104257, 1920240);
    res(5) = q(3399327, 763840);
    res(6) = q(66578432, 35198415);
    res(7) = q(-1674902723, 288716400);
    res(8) = q(54980371265625, 176692375811392);
    res(9) = q(-734375, 4826304);
    res(10) = q(171414593, 851261400);
    res(11) = q(137909, 3084480);
    return detail::rounded_table<T, DOP_B>(res);
}

template<typename T>
constexpr typename DOP_COEFS<T>::DOP_C DOP_COEFS<T>::make_C(){
    // The nodes: step 1 above
    using W = detail::dop853_work_t<T>;
    const W r6 = sqrt(W(6));
    typename DOP_COEFS<W>::DOP_C res(N_STAGES_EXT);
    res(0) = 0;
    res(3) = (W(6) - r6)/W(30);
    res(4) = (W(6) + r6)/W(30);
    res(2) = W(2)*res(3)/W(3);
    res(1) = W(2)*res(2)/W(3);
    res(5) = W(1)/W(3);
    res(6) = W(1)/W(4);
    res(7) = W(4)/W(13);
    res(8) = W(127)/W(195);
    res(9) = W(3)/W(5);
    res(10) = W(6)/W(7);
    res(11) = 1;
    res(12) = 1;
    res(13) = W(1)/W(10);
    res(14) = W(1)/W(5);
    res(15) = W(7)/W(9);
    return detail::rounded_table<T, DOP_C>(res);
}

template<typename T>
constexpr typename DOP_COEFS<T>::DOP_E DOP_COEFS<T>::make_E3(){
    // E3 = b - bhh, bhh being the embedded 3rd order solution: step 4 above
    using W = detail::dop853_work_t<T>;
    const auto q = detail::fraction<W>;
    const auto b = DOP_COEFS<W>::make_B();
    typename DOP_COEFS<W>::DOP_E res(N_STAGES + 1);
    res.fill(0);
    for (size_t i = 0; i < N_STAGES; ++i){
        res(i) = b(i);
    }
    res(0) -= q(31, 127);
    res(8) -= q(12675, 17272);
    res(11) -= q(3, 136);
    return detail::rounded_table<T, DOP_E>(res);
}

template<typename T>
constexpr typename DOP_COEFS<T>::DOP_E DOP_COEFS<T>::make_E5(){
    // E5 = b - bhat, bhat being the embedded 5th order solution: step 5 above
    using W = detail::dop853_work_t<T>;
    const auto q = detail::fraction<W>;
    const auto b = DOP_COEFS<W>::make_B();
    typename DOP_COEFS<W>::DOP_B bhat(N_STAGES);
    bhat.fill(0);
    bhat(0) = q(72864797, 1769693184);
    bhat(5) = q(8670301, 1527680);
    bhat(6) = q(22407563, 9386244);
    bhat(7) = q(-1839305140361, 246371328000);
    bhat(8) = q(3740185851728885, 5654156025964544);
    bhat(9) = q(-676000805, 1389975552);
    bhat(10) = q(2342627240947, 19613062656000);
    bhat(11) = q(137909, 2056320);
    typename DOP_COEFS<W>::DOP_E res(N_STAGES + 1);
    res.fill(0);
    for (size_t i = 0; i < N_STAGES; ++i){
        res(i) = b(i) - bhat(i);
    }
    return detail::rounded_table<T, DOP_E>(res);
}

template<typename T>
constexpr typename DOP_COEFS<T>::DOP_D DOP_COEFS<T>::make_D(){
    // The dense output's F3..F6 = h D K: step 6 above, and coef_mat_interp_dop853
    using W = detail::dop853_work_t<T>;
    const auto q = detail::fraction<W>;
    typename DOP_COEFS<W>::DOP_D res(INTERP_ORDER - 3, N_STAGES_EXT);
    res.fill(0);

    // F3: the factor of x^2 (1-x)^2
    res(0, 0) = q(-17751989329, 2106076560);
    res(0, 5) = q(4272954039, 7539864640);
    res(0, 6) = q(-118476319744, 38604839385);
    res(0, 7) = q(755123450731, 316657731600);
    res(0, 8) = q(3692384461234828125, 1744130441634250432);
    res(0, 9) = q(-4612609375, 5293382976);
    res(0, 10) = q(2091772278379, 933644586600);
    res(0, 11) = q(2136624137, 3382989120);
    res(0, 12) = q(-126493, 1421424);
    res(0, 13) = q(98350000, 5419179);
    res(0, 14) = q(-18878125, 2053168);
    res(0, 15) = q(-1944542619, 438351368);

    // F4: the factor of x^3 (1-x)^2
    res(1, 0) = q(32941697297, 3159114840);
    res(1, 5) = q(456696183123, 1884966160);
    res(1, 6) = q(19132610714624, 115814518155);
    res(1, 7) = q(-177904688592943, 474986597400);
    res(1, 8) = q(-4821139941836765625, 218016305204281304);
    res(1, 9) = q(30702015625, 3970037232);
    res(1, 10) = q(-42958039737137, 1400466879900);
    res(1, 11) = q(-5919468007, 634310460);
    res(1, 12) = q(2479159, 157936);
    res(1, 13) = q(-18750000, 602131);
    res(1, 14) = q(-19203125, 2053168);
    res(1, 15) = q(15700361463, 438351368);

    // F5: the factor of x^3 (1-x)^3
    res(2, 0) = q(12627015655, 631822968);
    res(2, 5) = q(-72955222965, 188496616);
    res(2, 6) = q(-13145744952320, 69488710893);
    res(2, 7) = q(30084216194513, 56998391688);
    res(2, 8) = q(-296858761006640625, 25648977082856624);
    res(2, 9) = q(569140625, 82709109);
    res(2, 10) = q(-18684190637, 18672891732);
    res(2, 11) = q(69644045, 89549712);
    res(2, 12) = q(-11847025, 4264272);
    res(2, 13) = q(-978650000, 16257537);
    res(2, 14) = q(519371875, 6159504);
    res(2, 15) = q(5256837225, 438351368);

    // F6: the factor of x^4 (1-x)^3
    res(3, 0) = q(-450944925, 17550638);
    res(3, 5) = q(-14532122925, 94248308);
    res(3, 6) = q(-595876966400, 2573655959);
    res(3, 7) = q(188748653015, 527762886);
    res(3, 8) = q(2545485458115234375, 27252038150535163);
    res(3, 9) = q(-1376953125, 36759604);
    res(3, 10) = q(53995596795, 518691437);
    res(3, 11) = q(210311225, 7047894);
    res(3, 12) = q(-1718875, 39484);
    res(3, 13) = q(58000000, 602131);
    res(3, 14) = q(-1546875, 39484);
    res(3, 15) = q(-1262172375, 8429834);

    return detail::rounded_table<T, DOP_D>(res);
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
            const auto abs_qj = abs(q_j);
            const auto abs_qj_new = abs(q_new_j);
            const T scale = atol_j + rtol_j * std::max(abs_qj, abs_qj_new);
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

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
DOP853<T, SD, OdeType, SP, Derived>::DOP853(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int direction, EventList<T, State> events)
    : Base(std::move(ode), t0, q0, std::move(rtol), std::move(atol), min_step, max_step, stepsize, direction, std::move(events)),
      r_scratch_(q0) {
    // the arrays of states take the shape of q0 too
    K_.fill(q0);
    coef_mat_.fill(q0);
    this->rhs(K_[N_STAGES], t0, q0);
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void DOP853<T, SD, OdeType, SP, Derived>::Reset(){
    Base::Reset();
    this->rhs(K_[N_STAGES], this->t_new(), this->vector_new());
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void DOP853<T, SD, OdeType, SP, Derived>::ReAdjust(const State& new_vector){
    Base::ReAdjust(new_vector);
    // the first stage of the next step (FSAL)
    this->rhs(K_[N_STAGES], this->t(), new_vector);
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void DOP853<T, SD, OdeType, SP, Derived>::cache_coef_matrix() const{
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

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T DOP853<T, SD, OdeType, SP, Derived>::step_impl(T& t_new, State& q_new, const T& t, const T& h, const State& q){
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

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
StepResult DOP853<T, SD, OdeType, SP, Derived>::adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q){
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

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
auto DOP853<T, SD, OdeType, SP, Derived>::local_interp_impl() const{
    this->cache_coef_matrix();
    return make_light_functor([](State& out, const T& t, const auto& t1, const auto& t2, const auto& y1, const auto& cm){
        return detail::coef_mat_interp_dop853<T, SD>(out, t, t1, t2, y1, cm);
    }, this->t_old(), this->t_new(), this->vector_old(), this->coef_mat_);
}

} // namespace ode
