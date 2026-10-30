#ifndef ODECRAFT_STATE_INTERP_HPP
#define ODECRAFT_STATE_INTERP_HPP

#include <odecraft/Toolkit/Tools.hpp>

namespace ode::interp::uni {

template<typename T>
class Interval{

public:

    Interval(const T& a, const T& b, int left_bdr=0, int right_bdr=0);

    Interval(const T& x) : _a(x), _b(x), _left(1), _right(1), _is_point(true) {}

    Interval() = default;

    DEFAULT_RULE_OF_FOUR(Interval)

    // ACCESSORS

    bool        contains(const T& t) const;

    const T&    start() const;

    const T&    end() const;

    const T&    h() const;

    int         dir() const;

    int         start_bdr() const;

    int         end_bdr() const;

    bool        is_point() const;

    bool        can_link_with(const Interval<T>& other) const;

    std::string signature(int prec = 3) const;

    // MODIFIERS

    void        open_end();

    void        close_end();

    void        close_start();

    void        adjust_start(const T& new_start);

    void        adjust_end(const T& new_end);

    void        link_with(Interval<T>& other);

private:

    T       _a;
    T       _b;
    T       _h = 0;
    int     _left = 0;
    int     _right = 0;
    bool    _is_point = false;
    int     _dir = 0;
};



template<typename T, typename SD>
class Interpolator{

public:

    using State = state_t<T, SD>;

    virtual ~Interpolator() = default;

    Interpolator() = default;

    //ACCESSORS

    virtual const Interval<T>&  interval() const = 0;

    virtual const State&        q_start() const = 0;

    virtual const State&        q_end() const = 0;

    virtual int                 start_bdr() const = 0;

    virtual int                 end_bdr() const = 0;

    virtual int                 dir() const = 0;

    virtual const T&            t_start() const = 0;

    virtual const T&            t_end() const = 0;

    virtual bool                is_out_of_bounds(const T& t) const = 0;

    virtual bool                can_link_with(const Interpolator& other) const = 0;

    void                        call(State& result, const T& t) const;

    virtual std::unique_ptr<Interpolator> clone() const = 0;

    //MODIFIERS

    virtual void link_with(Interpolator& other) = 0;

    virtual void link_after(Interpolator& other) = 0;

    virtual void adjust_start(const T& t_start) = 0;

    virtual void adjust_end(const T& t_end) = 0;

    virtual void close_end() = 0;

    virtual void close_start() = 0;

    virtual void open_end() = 0;

protected:

    DEFAULT_RULE_OF_FOUR(Interpolator)

private:

    virtual void _call_impl(State& result, const T& t) const = 0;
    
};

template<typename T, typename SD>
using BoxedInterp = pbox::Box<Interpolator<T, SD>>;

template<typename T, typename SD>
class LocalInterpolator : public Interpolator<T, SD>{
    //The purpose of this class is to provide a continuous output between the naturally adapted states
    //of an OdeSolver.

public:

    using Base = Interpolator<T, SD>;
    using State = Base::State;

    LocalInterpolator() = delete;

    LocalInterpolator(T t, State q);

    LocalInterpolator(T t1, T t2, State y1, State y2, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(LocalInterpolator)

    //ACCESSORS

    const Interval<T>&          interval() const override;

    const State&        q_start() const override;

    const State&        q_end() const override;

    virtual size_t              order() const;

    int                         start_bdr() const override;

    int                         end_bdr() const override;

    int                         dir() const override;

    const T&                    t_start() const override;

    const T&                    t_end() const override;

    bool                        is_out_of_bounds(const T& t) const override;

    bool                        can_link_with(const Interpolator<T, SD>& other) const override;

    std::unique_ptr<Interpolator<T, SD>>    clone() const override;

    //MODIFIERS

    void    adjust_start(const T& t_start) override;

    void    adjust_end(const T& t_end) override;

    void    link_with(Interpolator<T, SD>& other) override;

    void    link_after(Interpolator<T, SD>& other) override;

    void    close_end() override;

    void    close_start() override;

    void    open_end() override;

protected:

    const T& _t_min() const;

    const T& _t_max() const;

private:

    void _call_impl(State& result, const T& t) const override;

    Interval<T> _interval;
    T _tmin;
    T _tmax;
    State _q_old;
    State _q;

};



template<typename T, typename SD, typename Callable>
class CustomLocalInterpolator : public LocalInterpolator<T, SD>{
    using Base = LocalInterpolator<T, SD>;
public:
    using State = Base::State;
    CustomLocalInterpolator() = delete;

    template<typename U>
    CustomLocalInterpolator(U&& callable, T t1, T t2, State y1, State y2, int left_bdr, int right_bdr);

    DEFAULT_RULE_OF_FOUR(CustomLocalInterpolator)
    
    std::unique_ptr<Interpolator<T, SD>> clone() const override;

    void _call_impl(State& result, const T& t) const override;

private:

    Callable _callable;

};

template<typename T, typename SD, typename Callable>
template<typename U>
CustomLocalInterpolator<T, SD, Callable>::CustomLocalInterpolator(U&& callable, T t1, T t2, State y1, State y2, int left_bdr, int right_bdr) : Base(std::move(t1), std::move(t2), std::move(y1), std::move(y2), left_bdr, right_bdr), _callable(std::forward<U>(callable)) {}

template<typename T, typename SD, typename Callable>
std::unique_ptr<Interpolator<T, SD>> CustomLocalInterpolator<T, SD, Callable>::clone() const{
    return std::make_unique<CustomLocalInterpolator<T, SD, Callable>>(*this);
}

template<typename T, typename State, typename Callable>
void CustomLocalInterpolator<T, State, Callable>::_call_impl(State& result, const T& t) const{
    _callable(result, t);
}


template<typename T, typename SD>
class LinkedInterpolator : public Interpolator<T, SD>{
    using Base = Interpolator<T, SD>;
public:

    using interpolator_t = Base;
    using State = Base::State;

    LinkedInterpolator() = delete;

    LinkedInterpolator(BoxedInterp<T, SD> segment);

    LinkedInterpolator(T t, State q);

    DEFAULT_RULE_OF_FOUR(LinkedInterpolator)

    //ACCESSORS

    const Interval<T>&      interval() const override;

    size_t                  size() const;

    const State&            q_start() const override;

    const State&            q_end() const override;

    int                     start_bdr() const override;

    int                     end_bdr() const override;

    int                     dir() const override;

    const T&                t_start() const override;

    const T&                t_end() const override;

    const interpolator_t&   interpolant(size_t i) const;

    const interpolator_t&   last_interpolant() const;

    bool                    is_out_of_bounds(const T& t) const override;

    bool                    can_link_with(const interpolator_t& interpolant) const override;

    std::unique_ptr<interpolator_t> clone() const override;


    //MODIFIERS

    void    link_with(interpolator_t& interpolant) override;

    void    link_after(interpolator_t& interpolant) override;

    void    adjust_start(const T& t_new) override;

    void    adjust_end(const T& t_new) override;

    void    expand(pbox::Box<interpolator_t>&& interpolant);

    void    open_end() override;

    void    close_end() override;

    void    close_start() override;

private:

    size_t              _search_index(const T& t) const;

    bool                _can_replace_last_with(const interpolator_t& other) const;

    void                _call_impl(State& result, const T& t) const override;

    interpolator_t&       _get(size_t i);

    const interpolator_t& _get(size_t i) const;

    interpolator_t&       _get_last();

    const interpolator_t& _get_last() const;
    
    interpolator_t&       _get_safe(size_t i);

    const interpolator_t& _get_safe(size_t i) const;

    void                _throw_invalid_interpolant(const interpolator_t& interpolant) const;

    std::vector<pbox::owner<interpolator_t>> boxes_ = {};
    int _dir = 0;
    mutable bool _interval_cached = false;
    mutable Interval<T> _interval;

};



template<typename T, typename SD>
void lin_interp(state_t<T, SD>& result, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1, const state_t<T, SD>& y2);

template<typename T, typename SD>
void coef_mat_interp(state_t<T, SD>& result, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1, const state_t<T, SD>& y2, const state_t<T, SD>* coef_mat, size_t order);

template<typename T, typename SD>
using InterpObj = pbox::owner<Interpolator<T, SD>>;

} // namespace ode::interp::uni

#endif // ODECRAFT_STATE_INTERP_HPP