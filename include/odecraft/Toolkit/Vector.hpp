#ifndef ODECRAFT_VECTOR_HPP
#define ODECRAFT_VECTOR_HPP

#include <cstdlib>
#include <utility>
#include <cassert>
#include <vector>
#include <array>

namespace ode{

template<typename T, long N = -1>
class Vector;


template<typename T, long N>
class Vector {
    static_assert(N >= 0, "Vector size must be non-negative");

public:
    
    constexpr Vector() = default;
    
    constexpr Vector(const Vector& arr) = default;
    
    constexpr Vector(Vector&& arr) = default;
    
    constexpr Vector& operator=(const Vector& arr) = default;
    
    constexpr Vector& operator=(Vector&& arr) = default;
    ~Vector() = default;

    // Custom constructors
    
    constexpr Vector(size_t n) {assert(n >= 0 && n == size_t(N) && "Size mismatch in Vector constructor");}
    
    constexpr Vector(size_t n, const T& value) : Vector(private_tag{}, n, std::make_index_sequence<size_t(N)>{}, value) { assert(n == size_t(N) && "Size mismatch in Vector constructor"); }
    
    constexpr Vector(size_t n, T& value) : Vector(n, static_cast<const T&>(value)) {}
    
    constexpr Vector(size_t n, T&& value) : Vector(n, static_cast<const T&>(value)) {}

    template<typename... U>
    
    constexpr Vector(size_t n, const U&... constructor_args) : Vector(private_tag{}, n, std::make_index_sequence<size_t(N)>{}, constructor_args...) {}

    template<typename Int>
    
    constexpr T& operator[](Int i){
        return data_[i];
    }

    template<typename Int>
    
    constexpr const T& operator[](Int i) const{
        return data_[i];
    }


    
    constexpr size_t size() const {
        return data_.size();
    }

    
    constexpr const T* data() const {
        return data_.data();
    }

    
    constexpr T* data() {
        return data_.data();
    }

    
    constexpr auto begin() const {
        return data_.begin();
    }

    
    constexpr auto end() const {
        return data_.end();
    }

    
    constexpr auto begin() {
        return data_.begin();
    }

    
    constexpr auto end() {
        return data_.end();
    }

private:

    template<typename A, typename... U>
    friend Vector<A> make_vector(U&&... items);

    struct private_tag{};

    template<typename... U, size_t... I>
    
    constexpr Vector(private_tag, size_t n, std::index_sequence<I...>, const U&... values) : data_{ { ((void)I, T(values...))... } } {
        assert(n == size_t(N) && "Size mismatch in Vector constructor");
    }

    template<typename... U>
    
    constexpr Vector(private_tag, U&&... items) : data_{ std::forward<U>(items)... } {
        static_assert(sizeof...(U) == N, "Size mismatch in Vector constructor");
    }

    std::array<T, size_t(N)> data_;
};



template<typename T>
class Vector<T, -1> {

public:

    Vector() = default;
    Vector(const Vector& arr) = default;
    Vector(Vector&& arr) = default;
    Vector& operator=(const Vector& arr) = default;
    Vector& operator=(Vector&& arr) = default;
    ~Vector() = default;

    // Custom constructors
    Vector(size_t n) : data_(n) {}

    Vector(size_t n, const T& value) : data_(n, value) {}
    Vector(size_t n, T&& value) : data_(n, static_cast<const T&>(value)) {}
    Vector(size_t n, T& value) : data_(n, static_cast<const T&>(value)) {}

    template<typename... U>
    Vector(size_t n, const U&... constructor_args) {
        data_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            data_.emplace_back(constructor_args...);
        }
    }

    template<std::integral Int>
    inline constexpr T& operator[](Int i){
        return data_[i];
    }

    template<std::integral Int>
    inline constexpr const T& operator[](Int i) const{
        return data_[i];
    }

    inline size_t size() const {
        return data_.size();
    }

    inline const T* data() const {
        return data_.data();
    }

    inline T* data() {
        return data_.data();
    }

    inline auto begin() const {
        return data_.begin();
    }

    inline auto end() const {
        return data_.end();
    }

    inline auto begin() {
        return data_.begin();
    }

    inline auto end() {
        return data_.end();
    }

    inline void resize(size_t n) {
        data_.resize(n);
    }

    inline void reserve(size_t n) {
        data_.reserve(n);
    }

    inline void push_back(const T& value){
        data_.push_back(value);
    }

    inline void push_back(T&& value){
        data_.push_back(std::move(value));
    }

    template<typename... Args>
    inline void emplace_back(Args&&... args){
        data_.emplace_back(std::forward<Args>(args)...);
    }

private:

    template<typename A, typename... U>
    friend Vector<A> make_vector(U&&... items);

    struct private_tag{};

    template<typename... U>
    
    Vector(private_tag, U&&... items) {
        data_.reserve(sizeof...(U));
        (data_.emplace_back(std::forward<U>(items)), ...);
    }

    std::vector<T> data_;

};


template<typename T, typename... U>
inline Vector<T> make_vector(U&&... items){
    return Vector<T>(typename Vector<T>::private_tag{}, std::forward<U>(items)...);
}


} // namespace ode

#endif // ODECRAFT_VECTOR_HPP