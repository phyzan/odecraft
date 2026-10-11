#ifndef ODECRAFT_SCRATCH_HPP
#define ODECRAFT_SCRATCH_HPP


#include "odecraft/Toolkit/Cache.hpp"
#include <vector>
#include <cassert>

namespace ode::detail{

template<typename T>
class ScratchFrame;

template<typename T>
class ScratchStack{
public:

    ScratchFrame<T> request(size_t size){
        return ScratchFrame<T>{*this, size};
    }

    template<typename... Args>
    void update(Args&&... args){
        // Reinitialize all free elements in the current pool
        // when a global default state has been updated
        // or just update them by passing args... in the constructor
        T new_v(std::forward<Args>(args)...);
        std::vector<T>& ch = chunks_[used_];
        for (size_t i=depth_; i < ch.size(); i++){
            ch[i] = new_v;
        }

        for (size_t i=used_+1; i < chunk_count_; i++){
            std::vector<T>& chunk = chunks_[i];
            for (T& value : chunk){
                value = new_v;
            }
        }
    }

    ScratchStack() = default;
    ScratchStack(const ScratchStack&) = delete;
    ScratchStack(ScratchStack&&) = delete;
    ScratchStack& operator=(const ScratchStack&) = delete;
    ScratchStack& operator=(ScratchStack&&) = delete;
    ~ScratchStack() = default;
private:
    friend class ScratchFrame<T>;

    size_t avail_size() const {
        return chunks_[used_].size() - depth_;
    }

    T* find_room(size_t size){
        if (size > avail_size()){
            return alloc(size);
        } else {
            T* result = chunks_[used_].data() + depth_;
            depth_ += size;
            return result;
        }
    }

    // returns a pointer to a fresh block
    // and moves the stack pointer next to it
    T* alloc(size_t size) {
        if (chunk_count_ == used_ + 1 && depth_ > 0){
            // allocate a new chunk and move the stack pointer there
            if (chunks_.size() == chunk_count_){
                chunks_.emplace_back(size);
            } else {
                chunks_[chunk_count_] = std::vector<T>(size);
            }
            used_++;
            depth_ = size;
            return chunks_[chunk_count_++].data();
        } else if (depth_ > 0){
            size_t n = chunks_[used_+1].size();
            for (size_t i = used_ + 2; i < chunk_count_; i++){
                n += chunks_[i].size();
                chunks_[i] = std::vector<T>{};
            }
            n = std::max(n, size);
            chunks_[used_ + 1].resize(n);
            chunk_count_ = used_ + 2;
            depth_ = size;
            return chunks_[++used_].data();
        } else {
            size_t n = chunks_[used_].size();
            for (size_t i = used_ + 1; i < chunk_count_; i++){
                n += chunks_[i].size();
                chunks_[i] = std::vector<T>{};
            }
            n = std::max(n, size);
            chunks_[used_].resize(n);
            chunk_count_ = used_ + 1;
            depth_ = size;
            return chunks_[used_].data();
        }
    }

    std::vector<std::vector<T>> chunks_ = {std::vector<T>{}};
    size_t chunk_count_{}; // effective number of chunks (innermost chunks could be empty due to compacting)
    size_t used_{}; // number of chunks that are unavailable
    size_t depth_{}; // used elements inside the current available chunk, the one sitting next to the used ones
};


template<typename T>
class [[nodiscard("A discarded scratch frame is released immediately")]]
ScratchFrame{
public:
    ScratchFrame(const ScratchFrame&) = delete;
    ScratchFrame(ScratchFrame&&) = delete;
    ScratchFrame& operator=(const ScratchFrame&) = delete;
    ScratchFrame& operator=(ScratchFrame&&) = delete;

    ~ScratchFrame(){
        stack_.used_ = used_stack;
        stack_.depth_ = previous_depth;
    }

    explicit ScratchFrame(ScratchStack<T>& stack, size_t size) :
        used_stack(stack.used_),
        previous_depth(stack.depth_),
        size_(size),
        data_(stack.find_room(size)),
        stack_(stack) {}

    size_t size() const {
        return size_;
    }

    const T& operator[](size_t i) const& {
        assert(i < size_ && "Index out of bounds");
        return data_[i];
    }

    T& operator[](size_t i) & {
        assert(i < size_ && "Index out of bounds");
        return data_[i];
    }

    T& operator[](size_t) && = delete;

    const T* data() const& { return data_; }
    T*       data() &      { return data_; }
    T*       data() &&     = delete;

private:
    size_t used_stack, previous_depth; // Reassign these to the stack in the end
    size_t size_;
    T* data_;
    ScratchStack<T>& stack_;
};


template<typename T>
ScratchStack<T>& scratch_pool() {
    thread_local ScratchStack<T> s;
    return s;
}

template<typename T, typename... Args>
void update_scratch(Args&&... args){
    detail::ScratchStack<T>& pool = detail::scratch_pool<T>();
    pool.update(std::forward<Args>(args)...);
}

} // namespace ode::detail

namespace ode {
    
template<typename T, size_t N>
auto request_scratch(size_t size = N){
    assert((N == 0 || size == N) && "Compile-time with runtime size mismatch");
    if constexpr (detail::scratch_is_static<T, N>){
        (void)size;
        return std::array<T, N>{};
    } else {
        return ode::detail::scratch_pool<T>().request(size);
    }
}

} // namespace ode

#endif // ODECRAFT_SCRATCH_HPP