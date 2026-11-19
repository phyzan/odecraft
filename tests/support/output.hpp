#pragma once

/**
 * @file output.hpp
 * @brief What the library prints, captured for a check or kept out of the test log.
 *
 * Shared by both test executables, so it includes nothing of the library.
 */

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace odetest {

/// @brief Redirects std::cout into a string for its lifetime.
class CaptureCout {
public:
    CaptureCout() : previous_(std::cout.rdbuf(buffer_.rdbuf())) {}
    ~CaptureCout(){ std::cout.rdbuf(previous_); }
    CaptureCout(const CaptureCout&) = delete;
    CaptureCout& operator=(const CaptureCout&) = delete;

    std::string str() const { return buffer_.str(); }

private:
    std::ostringstream buffer_;
    std::streambuf* previous_;
};


/**
 * @brief Discards whatever is written to the standard error stream for its lifetime.
 *
 * At the level of the file descriptor, so that it also silences C libraries that print their
 * diagnostics there before the library throws, such as Qhull.
 */
class SilenceStderr {
public:
    SilenceStderr(){
        std::fflush(stderr);
#if defined(_WIN32)
        saved_ = _dup(_fileno(stderr));
        const int sink = _open("NUL", _O_WRONLY);
        if (sink >= 0){
            _dup2(sink, _fileno(stderr));
            _close(sink);
        }
#else
        saved_ = dup(fileno(stderr));
        const int sink = open("/dev/null", O_WRONLY);
        if (sink >= 0){
            dup2(sink, fileno(stderr));
            close(sink);
        }
#endif
    }

    ~SilenceStderr(){
        std::fflush(stderr);
        if (saved_ < 0){
            return;
        }
#if defined(_WIN32)
        _dup2(saved_, _fileno(stderr));
        _close(saved_);
#else
        dup2(saved_, fileno(stderr));
        close(saved_);
#endif
    }

    SilenceStderr(const SilenceStderr&) = delete;
    SilenceStderr& operator=(const SilenceStderr&) = delete;

private:
    int saved_ = -1;
};

} // namespace odetest
