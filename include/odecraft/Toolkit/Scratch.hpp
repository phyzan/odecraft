#ifndef ODECRAFT_SCRATCH_HPP
#define ODECRAFT_SCRATCH_HPP


#include "odecraft/Toolkit/Cache.hpp"
#include <new>
#include <vector>


namespace ode::detail{

template<typename T>
class ScratchFrame;

template<typename T>
class ScratchPage;

template<typename T>
class ScratchStackFrame;


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


// A frame of the thread-local pool. Only valid as a local variable: frames must be released in
// the reverse order of their requests, which nested scopes guarantee and a frame stored in a
// longer-lived object does not. Debug builds assert it on release.
template<typename T>
class [[nodiscard("A discarded scratch frame is released immediately")]]
ScratchFrame{
public:
    ScratchFrame(const ScratchFrame&) = delete;
    ScratchFrame(ScratchFrame&&) = delete;
    ScratchFrame& operator=(const ScratchFrame&) = delete;
    ScratchFrame& operator=(ScratchFrame&&) = delete;

    ~ScratchFrame(){
        assert(stack_.used_ == top_used && stack_.depth_ == top_depth
               && "Scratch frames must be released in the reverse order of their requests");
        stack_.used_ = used_stack;
        stack_.depth_ = previous_depth;
    }

    explicit ScratchFrame(ScratchStack<T>& stack, size_t size) :
        used_stack(stack.used_),
        previous_depth(stack.depth_),
        size_(size),
        data_(stack.find_room(size)),
#ifndef NDEBUG
        top_used(stack.used_),
        top_depth(stack.depth_),
#endif
        stack_(stack) {}

    size_t size() const {
        return size_;
    }

    const T& operator[](size_t i) const& {
        assert(i < size_ && "Index out of bounds");
        return data()[i];
    }

    T& operator[](size_t i) & {
        assert(i < size_ && "Index out of bounds");
        return data()[i];
    }

    T& operator[](size_t) && = delete;

    const T* data() const& { return data_; }
    T*       data() &      { return data_; }
    T*       data() &&     = delete;

private:

    size_t used_stack, previous_depth; // Reassign these to the stack in the end
    size_t size_;
    T* data_;
#ifndef NDEBUG
    size_t top_used, top_depth; // where the stack's top was right after this request
#endif
    ScratchStack<T>& stack_;
};

template<typename T>
ScratchStack<T>& scratch_pool() {
    thread_local ScratchStack<T> s;
    return s;
}


// Allocates on cache-line boundaries. malloc and new only promise 16 bytes, and from a buffer
// aligned like that every 64-byte (AVX-512) vector load straddles two cache lines.
template<typename T>
struct CacheLineAllocator{
    using value_type = T;
    static constexpr std::align_val_t alignment{
        alignof(T) > 64 ? alignof(T) : 64,
    };

    CacheLineAllocator() = default;

    template<typename U>
    constexpr CacheLineAllocator(const CacheLineAllocator<U>&) noexcept {}

    T* allocate(size_t n){
        return static_cast<T*>(::operator new(n*sizeof(T), alignment));
    }

    void deallocate(T* p, size_t){
        ::operator delete(p, alignment);
    }

    template<typename U>
    bool operator==(const CacheLineAllocator<U>&) const noexcept { return true; }
};


// A fixed-capacity scratch region owned by the object that uses it, so unlike a ScratchFrame
// it can live as long as that object does, be copied with it, and be used from any thread.
// Requests are carved off its front in stack order: a bump of the depth, no allocation and
// no thread-local lookup. The owner sizes it for the most its requests can hold at once;
// asking for more is a bug, caught by an assert.
template<typename T>
class ScratchPage{
public:

    explicit ScratchPage(size_t capacity) : buffer_(capacity) {}

    // A copy gets a buffer of its own: scratch is never shared, and its contents are not
    // worth copying.
    ScratchPage(const ScratchPage& other) : buffer_(other.buffer_.size()) {}
    ScratchPage(ScratchPage&&) noexcept = default;

    ScratchPage& operator=(const ScratchPage& other){
        if (buffer_.size() != other.buffer_.size()){
            buffer_ = std::vector<T, CacheLineAllocator<T>>(other.buffer_.size());
        }
        return *this;
    }
    ScratchPage& operator=(ScratchPage&&) noexcept = default;
    ~ScratchPage() = default;

    size_t capacity() const {
        return buffer_.size();
    }

    template<size_t M>
    ScratchStackFrame<T> request(size_t size){
        assert((M == 0 || size == M) && "Compile-time with runtime size mismatch");
        return ScratchStackFrame<T>{*this, size};
    }

private:
    friend class ScratchStackFrame<T>;
    std::vector<T, CacheLineAllocator<T>> buffer_;
    size_t depth_ = 0; // elements in use, from the front of buffer_
};


template<typename T>
class [[nodiscard("A discarded scratch frame is released immediately")]]
ScratchStackFrame{
public:
    ScratchStackFrame(const ScratchStackFrame&) = delete;
    ScratchStackFrame(ScratchStackFrame&&) = delete;
    ScratchStackFrame& operator=(const ScratchStackFrame&) = delete;
    ScratchStackFrame& operator=(ScratchStackFrame&&) = delete;

    ScratchStackFrame(ScratchPage<T>& page, size_t size):
        size_(size),
        write_back_depth(page.depth_),
        page_(page) {
        assert(size <= page.buffer_.size() - page.depth_ && "Scratch request exceeds the page's capacity");
        data_ = page.buffer_.data() + page.depth_;
        page.depth_ += size;
    }

    ~ScratchStackFrame() {
        assert(page_.depth_ == write_back_depth + size_
               && "Scratch frames must be released in the reverse order of their requests");
        page_.depth_ = write_back_depth;
    }

    size_t size() const {
        return size_;
    }

    const T& operator[](size_t i) const& {
        assert(i < size_ && "Index out of bounds");
        return data()[i];
    }

    T& operator[](size_t i) & {
        assert(i < size_ && "Index out of bounds");
        return data()[i];
    }

    T& operator[](size_t) && = delete;

    const T* data() const& { return data_; }
    T*       data() &      { return data_; }
    T*       data() &&     = delete;

private:
    T* data_;
    size_t size_;
    size_t write_back_depth;
    ScratchPage<T>& page_;
};


// The page of a trivial scalar with a compile-time size: holds nothing, and every request is a
// fresh std::array in automatic storage.
template<typename T>
class StaticScratchPage {
public:
    explicit constexpr StaticScratchPage(size_t /*capacity*/) {}

    template<size_t M>
    std::array<T, M> request([[maybe_unused]] size_t size) const {
        static_assert(M > 0, "A static scratch page serves compile-time sizes only");
        assert(size == M && "Mismatch between compile-time and runtime buffer size");
        return std::array<T, M>{};
    }
};


// Stands in for a page that a class only needs in some configurations.
struct NoScratchPage {
    explicit constexpr NoScratchPage(size_t /*capacity*/) {}
};


template<typename T, size_t N>
using ScratchPageFor = std::conditional_t<scratch_is_static<T, N>, StaticScratchPage<T>, ScratchPage<T>>;

template<bool Enabled, typename T, size_t N>
using ScratchPageIf = std::conditional_t<Enabled, ScratchPageFor<T, N>, NoScratchPage>;


template<typename T, typename... Args>
void update_scratch(Args&&... args){
    detail::ScratchStack<T>& pool = detail::scratch_pool<T>();
    pool.update(std::forward<Args>(args)...);
}

/// @brief A temporary buffer of `size` elements: a std::array for a trivial T and a compile-time N,
/// otherwise a frame of the thread-local pool. Bind it to a local variable only (see ScratchFrame).
template<typename T, size_t N>
auto request_scratch(size_t size = N){
    assert((N == 0 || size == N) && "Compile-time with runtime size mismatch");
    if constexpr (scratch_is_static<T, N>){
        (void)size;
        return std::array<T, N>{};
    } else {
        return scratch_pool<T>().request(size);
    }
}

} // namespace ode::detail

namespace ode {

/// @brief A scratch page of `size` elements, for an object to keep as a member and serve its temporaries from.
/// For a trivial T and a compile-time N it holds nothing, and each `request<M>(m)` returns a std::array<T, M>.
template<typename T, size_t N>
detail::ScratchPageFor<T, N> request_scratch_page(size_t size = N){
    assert((N == 0 || size == N) && "Mismatch between compile-time and runtime buffer size in page request");
    return detail::ScratchPageFor<T, N>(size);
}

} // namespace ode

#endif // ODECRAFT_SCRATCH_HPP