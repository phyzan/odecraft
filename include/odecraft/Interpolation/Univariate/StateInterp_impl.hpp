#pragma once

#include "odecraft/Toolkit/Tools.hpp"
#include "StateInterp.hpp"
#include <stdexcept>


namespace ode::interp::uni {

template<typename T>
Interval<T>::Interval(const T& a, const T& b, int left_bdr, int right_bdr) : _a(a), _b(b), _h(b-a), _left(sgn(left_bdr)), _right(sgn(right_bdr)), _dir(sgn(a, b)){
    if (a == b){
        throw std::runtime_error("Zero interval not allowed");
    }
}

template<typename T>
bool Interval<T>::contains(const T& t) const{
    return (t==_a) ? _left != -1 : ( (t==_b) ? _right != -1 : (_a*_dir < t*_dir) && (t*_dir < _b*_dir));
}


template<typename T>
const T& Interval<T>::start() const{
    return _a;
}


template<typename T>
const T& Interval<T>::end() const{
    return _b;
}

template<typename T>
const T& Interval<T>::h() const{
    return _h;
}

template<typename T>
int Interval<T>::dir() const{
    return _dir;
}

template<typename T>
int Interval<T>::start_bdr() const{
    return _left;
}

template<typename T>
int Interval<T>::end_bdr() const{
    return _right;
}

template<typename T>
bool Interval<T>::is_point() const{
    return _is_point;
}

template<typename T>
void Interval<T>::open_end(){
    if (_is_point){
        throw std::runtime_error("Cannot open end boundary on a point interval");
    } else {
        _right = -1;
    }
}

template<typename T>
void Interval<T>::close_end(){
    _right = 1;
}


template<typename T>
void Interval<T>::close_start(){
    _left = 1;
}

template<typename T>
void Interval<T>::adjust_start(const T& new_start){
    if (sgn(new_start, _b) == _dir){
        _a = new_start;
        _h = _b - _a;
    } else {
        throw std::runtime_error("Cannot resize interval");
    }
}

template<typename T>
void Interval<T>::adjust_end(const T& new_end){
    if (sgn(_a, new_end) == _dir){
        _b = new_end;
        _h = _b - _a;
    } else {
        throw std::runtime_error("Cannot resize interval");
    }
}


template<typename T>
bool Interval<T>::can_link_with(const Interval<T>& other) const{
    //boundaries and directions must match.
    return (_b == other._a) && (_right*other._left != 1) && (_right != 0 || other._left != 0) && (_dir+other._dir != 0);
}


template<typename T>
void Interval<T>::link_with(Interval<T>& other){
    if (!can_link_with(other)){
        throw std::runtime_error("Intervals cannot link");
    }

    if (_right == 0){
        _right = -other._left;
    } else if (other._left == 0){
        other._left = -_right;
    }
}

template<typename T>
std::string Interval<T>::signature(int prec) const{
    std::ostringstream  oss;
    oss << std::setprecision(prec);
    if (is_point()){
        oss << "[" << _a << "]";
        return oss.str();
    }else{
        oss << (_left == 1 ? "[" : (_left == -1 ? "(" : "|")) << _a << ", " << _b << (_right == 1 ? "]" : (_right == -1 ? ")" : "|"));
        return oss.str();
    }
}


template<typename T, typename SD>
void Interpolator<T, SD>::throw_domain_error(const T &t) const {
    std::ostringstream oss;
    oss << std::setprecision(8) << "Scalar " << t << " is out of bounds in interpolator with range " << interval().signature(8);
    throw std::runtime_error(oss.str());
}


template<typename T, typename SD>
LocalInterpolator<T, SD>::LocalInterpolator(T t, State q) : _interval(t), _tmin(t), _tmax(std::move(t)), _q_old(q), _q(std::move(q)){}


template<typename T, typename SD>
LocalInterpolator<T, SD>::LocalInterpolator(T t1, T t2, State y1, State y2, int left_bdr, int right_bdr) : _interval(t1, t2, left_bdr, right_bdr), _tmin(std::move(t1)), _tmax(std::move(t2)), _q_old(std::move(y1)), _q(std::move(y2)){}

template<typename T, typename SD>
const Interval<T>& LocalInterpolator<T, SD>::interval() const{
    return _interval;
}

template<typename T, typename SD>
const LocalInterpolator<T, SD>::State& LocalInterpolator<T, SD>::q_start() const{
    return _q_old;
}

template<typename T, typename SD>
const LocalInterpolator<T, SD>::State& LocalInterpolator<T, SD>::q_end() const{
    return _q;
}

template<typename T, typename SD>
size_t LocalInterpolator<T, SD>::order() const{
    return 1;
}

template<typename T, typename SD>
int LocalInterpolator<T, SD>::start_bdr() const{
    return _interval.start_bdr();
}

template<typename T, typename SD>
int LocalInterpolator<T, SD>::end_bdr() const{
    return _interval.end_bdr();
}

template<typename T, typename SD>
int LocalInterpolator<T, SD>::dir() const{
    return _interval.dir();
}

template<typename T, typename SD>
const T& LocalInterpolator<T, SD>::t_start() const{
    return _interval.start();
}

template<typename T, typename SD>
const T& LocalInterpolator<T, SD>::t_end() const{
    return _interval.end();
}

template<typename T, typename SD>
bool LocalInterpolator<T, SD>::is_out_of_bounds(const T& t) const{
    return !_interval.contains(t);
}

template<typename T, typename SD>
bool LocalInterpolator<T, SD>::can_link_with(const Interpolator<T, SD>& other) const{
    
    if (const auto* p = dynamic_cast<const LocalInterpolator<T, SD>*>(&other)){
        return _interval.can_link_with(p->_interval);
    } else {
        return (t_end() == other.t_end()) && (end_bdr()*other.start_bdr() != 1) && !(end_bdr() == 0 && other.start_bdr() == 0) && (dir()+other.dir() != 0);
    }
}

template<typename T, typename SD>
std::unique_ptr<Interpolator<T, SD>> LocalInterpolator<T, SD>::clone() const{
    return std::make_unique<LocalInterpolator<T, SD>>(*this);
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::adjust_start(const T& t_start){
    int d = _interval.dir();
    if ( t_start*d < _interval.end()*d && t_start*d >= _tmin*d){
        _interval.adjust_start(t_start);
    }
    else{
        throw std::runtime_error("Cannot adjust interval");
    }
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::adjust_end(const T& t_end){
    int d = _interval.dir();
    if (this->interval().is_point()){
        if (t_end != _interval.start()){
            throw std::runtime_error("Cannot adjust point-interval");
        }
    }else if (_interval.start()*d < t_end*d && t_end*d <= _tmax*d){
        _interval.adjust_end(t_end);
    }
    else{
        throw std::runtime_error("Cannot adjust interval");
    }
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::link_with(Interpolator<T, SD>& other){
    if (auto* p = dynamic_cast<LocalInterpolator<T, SD>*>(&other)){
        _interval.link_with(p->_interval);
    }
    else{
        other.link_after(*this);
    }
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::link_after(Interpolator<T, SD>& other){
    other.link_with(*this);
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::close_end(){
    _interval.close_end();
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::close_start(){
    _interval.close_start();
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::open_end(){
    _interval.open_end();
}

template<typename T, typename SD>
const T& LocalInterpolator<T, SD>::_t_min() const{
    return _tmin;
}

template<typename T, typename SD>
const T& LocalInterpolator<T, SD>::_t_max() const{
    return _tmax;
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::call(State& out, const T& t) const{
    // The interval might have been adjusted, but always lies within the step [_tmin, _tmax]:
    // the stored states are only exact at the step's own ends, anywhere else it interpolates.
    if (!_interval.contains(t)){
        this->throw_domain_error(t);
    } else if (t == _tmin){
        copy_state<SD>(out, _q_old);
    } else if (t == _tmax){
        copy_state<SD>(out, _q);
    } else {
        this->open_interval_call_impl(out, t);
    }
}

template<typename T, typename SD>
void LocalInterpolator<T, SD>::open_interval_call_impl(State& result, const T& t) const{
    lin_interp<T, SD>(result, t, _tmin, _tmax, _q_old, _q);
}

template<typename T, typename SD, typename Callable>
template<typename U>
CustomLocalInterpolator<T, SD, Callable>::CustomLocalInterpolator(U&& callable, T t1, T t2, State y1, State y2, int left_bdr, int right_bdr) : Base(std::move(t1), std::move(t2), std::move(y1), std::move(y2), left_bdr, right_bdr), _callable(std::forward<U>(callable)) {}

template<typename T, typename SD, typename Callable>
std::unique_ptr<Interpolator<T, SD>> CustomLocalInterpolator<T, SD, Callable>::clone() const{
    return std::make_unique<CustomLocalInterpolator<T, SD, Callable>>(*this);
}

template<typename T, typename SD, typename Callable>
void CustomLocalInterpolator<T, SD, Callable>::open_interval_call_impl(State& result, const T& t) const{
    _callable(result, t);
}

template<typename T, typename SD>
LinkedInterpolator<T, SD>::LinkedInterpolator(BoxedInterp<T, SD> segment) : _dir(segment->dir()){
    boxes_.push_back(std::move(segment));
}

template<typename T, typename SD>
LinkedInterpolator<T, SD>::LinkedInterpolator(T t, State q) {
    boxes_.push_back(pbox::make_box<LocalInterpolator<T, SD>>(std::move(t), std::move(q)));
}


template<typename T, typename SD>
const Interval<T>& LinkedInterpolator<T, SD>::interval() const{
    if (_interval_cached){
        return _interval;
    } else if (t_start() == t_end()){
        _interval = Interval<T>(t_start());
        _interval_cached = true;
    } else {
        _interval = Interval<T>(t_start(), t_end(), start_bdr(), end_bdr());
        _interval_cached = true;
    }
    return _interval;
}

template<typename T, typename SD>
size_t LinkedInterpolator<T, SD>::size() const{
    return boxes_.size();
}


template<typename T, typename SD>
const LinkedInterpolator<T, SD>::State& LinkedInterpolator<T, SD>::q_start() const{
    return _get(0).q_start();
}


template<typename T, typename SD>
const LinkedInterpolator<T, SD>::State& LinkedInterpolator<T, SD>::q_end() const{
    return _get_last().q_end();
}

template<typename T, typename SD>
int LinkedInterpolator<T, SD>::start_bdr() const{
    return _get(0).start_bdr();
}

template<typename T, typename SD>
int LinkedInterpolator<T, SD>::end_bdr() const{
    return _get_last().end_bdr();
}


template<typename T, typename SD>
 int LinkedInterpolator<T, SD>::dir() const{
    return _dir;
}

template<typename T, typename SD>
const T& LinkedInterpolator<T, SD>::t_start() const{
    return _get(0).t_start();
}

template<typename T, typename SD>
const T& LinkedInterpolator<T, SD>::t_end() const{
    return _get_last().t_end();
}

template<typename T, typename SD>
const Interpolator<T, SD>& LinkedInterpolator<T, SD>::interpolant(size_t i) const{
    return _get_safe(i);
}

template<typename T, typename SD>
const Interpolator<T, SD>& LinkedInterpolator<T, SD>::last_interpolant() const{
    return _get_last();
}


template<typename T, typename SD>
bool LinkedInterpolator<T, SD>::is_out_of_bounds(const T& t) const {
    if (t == t_start()){
        return _get(0).is_out_of_bounds(t);
    }
    else if (t == t_end()){
        return _get_last().is_out_of_bounds(t);
    }
    else{
        return (t*_dir < t_start()*_dir) || (t*_dir > t_end()*_dir);
    }
}

template<typename T, typename SD>
bool LinkedInterpolator<T, SD>::can_link_with(const Interpolator<T, SD>& interpolant) const{
    return _get_last().can_link_with(interpolant) || _can_replace_last_with(interpolant);
}


template<typename T, typename SD>
std::unique_ptr<Interpolator<T, SD>> LinkedInterpolator<T, SD>::clone() const{
    return std::make_unique<LinkedInterpolator<T, SD>>(*this);
}

template<typename T, typename SD>
void LinkedInterpolator<T, SD>::link_with(Interpolator<T, SD>& interpolant){
    _interval_cached = false;
    _get_last().link_with(interpolant);
}

template<typename T, typename SD>
void LinkedInterpolator<T, SD>::link_after(Interpolator<T, SD>& interpolant){
    _interval_cached = false;
    _get(0).link_after(interpolant);
}

template<typename T, typename SD>
void LinkedInterpolator<T, SD>::adjust_start(const T& t_new) {
    _interval_cached = false;
    _get(0).adjust_start(t_new);
}

template<typename T, typename SD>
void LinkedInterpolator<T, SD>::adjust_end(const T& t_new) {
    _interval_cached = false;
    _get_last().adjust_end(t_new);
}

template<typename T, typename SD>
void LinkedInterpolator<T, SD>::expand(pbox::Box<Interpolator<T, SD>>&& interpolant) {
    _interval_cached = false;
    int dir = interpolant->dir();
    if (_can_replace_last_with(*interpolant)){
        boxes_.back() = std::move(interpolant);
    } else if(_get_last().can_link_with(*interpolant)){
        boxes_.push_back(std::move(interpolant));
        boxes_[size()-2]->link_with(*boxes_[size()-1]);
    } else {
        this->_throw_invalid_interpolant(*interpolant);
    }

    if ((dir != 0) && (_dir == 0)){
        _dir = dir;
    }
}

template<typename T, typename SD>
void LinkedInterpolator<T, SD>::open_end(){
    _interval_cached = false;
    _get_last().open_end();
}


template<typename T, typename SD>
void LinkedInterpolator<T, SD>::close_end(){
    _interval_cached = false;
    _get_last().close_end();
}


template<typename T, typename SD>
void LinkedInterpolator<T, SD>::close_start(){
    _interval_cached = false;
    _get(0).close_end();
}


template<typename T, typename SD>
size_t LinkedInterpolator<T, SD>::_search_index(const T& t) const{
    //no need to check for out of bounds, it must have just been checked before this function call.
    //t must NOT be equal to the _t_end(). This case has already been taken care of.

    size_t left = 0;
    size_t right = size()-1;
    size_t mid;
    const int d = dir();
    while (right > left){
        mid = (right+left) / 2;
        if (!_get(left).is_out_of_bounds(t)){
            return left;
        }
        else if (!_get(mid).is_out_of_bounds(t)){
            return mid;
        }
        else if ( _get(left).t_start()*d < t*d && t*d < _get(mid).t_end()*d){
            left++;
            right = mid-1;
        }
        else{
            left = mid+1;
        }
    }
    //now right == left
    return right;
}

template<typename T, typename SD>
bool LinkedInterpolator<T, SD>::_can_replace_last_with(const Interpolator<T, SD>& other) const {
    //ideally we should check that the vectors match in the boundary, but this might be too expensive, so the user should check it if necessary.
    return (_get_last().t_start() == _get_last().t_end()) && other.start_bdr() == 1 && (_dir + other.dir() != 0);
}

template<typename T, typename SD>
void LinkedInterpolator<T, SD>::call(State& result, const T& t) const{
    size_t ind = _search_index(t);
    _get(ind).call(result, t);
}


template<typename T, typename SD>
Interpolator<T, SD>& LinkedInterpolator<T, SD>::_get(size_t i) {
    return *boxes_[i];
}

template<typename T, typename SD>
const Interpolator<T, SD>& LinkedInterpolator<T, SD>::_get(size_t i) const{
    return *boxes_[i];
}

template<typename T, typename SD>
Interpolator<T, SD>& LinkedInterpolator<T, SD>::_get_last() {
    return *boxes_.back();
}

template<typename T, typename SD>
const Interpolator<T, SD>& LinkedInterpolator<T, SD>::_get_last() const {
    return *boxes_.back();
}

template<typename T, typename SD>
Interpolator<T, SD>& LinkedInterpolator<T, SD>::_get_safe(size_t i){
    return *boxes_.at(i);
}

template<typename T, typename SD>
const Interpolator<T, SD>& LinkedInterpolator<T, SD>::_get_safe(size_t i) const {
    return *boxes_.at(i);
}


template<typename T, typename SD>
void LinkedInterpolator<T, SD>::_throw_invalid_interpolant(const Interpolator<T, SD>& interpolant) const {
    throw std::runtime_error("Invalid interpolant in LinkedInterpolator. Cannot join intervals " + _get_last().interval().signature() + " + " + interpolant.interval().signature());
}


template<typename T, typename SD>
void lin_interp(state_t<T, SD>& result, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1, const state_t<T, SD>& y2){
    assert( (t != t1 && t != t2) && "lin_interp must only be called on the open interval");

    auto interval = t2 - t1;
    auto delta_t = t - t1;

    SD::for_each([&interval, &delta_t](value_t<T, SD>& r_i, const value_t<T, SD>& y1_i, const value_t<T, SD>& y2_i){
        r_i = y1_i + (y2_i - y1_i) / interval * delta_t;
    }, result, y1, y2);
}

template<typename T, typename SD>
void coef_mat_interp(state_t<T, SD>& result, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1, const state_t<T, SD>& /*y2*/, const state_t<T, SD>* coef_mat, size_t order){
    
    assert( (t != t1 && t != t2) && "coef_mat_interp must only be called on the open interval");

    const auto h = t2-t1;
    const auto x = (t-t1)/h;

    using S = value_t<T, SD>;

    SD::for_each([](S& r_i, const S& y1_i){
        r_i = y1_i;
    }, result, y1);

    T x_power = x;
    for (size_t j=0; j<order; ++j){
        SD::for_each([&h, &x_power](S& r_i, const S& c_i){
            r_i += h * c_i * x_power;
        }, result, coef_mat[j]);
        x_power *= x;
    }
}

} // namespace ode::interp::uni
