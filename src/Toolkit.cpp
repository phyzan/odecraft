#include <odecraft/Compiled/Toolkit.hpp>

namespace ode::crafted{

void set_mpreal_prec(mpfr_prec_t prec){
#ifdef ODECRAFT_USE_LAZY_MPREAL
    // Sets the MPFR default *and* re-sizes the lazy backend's thread_local scratch buffers.
    lazy::set_default_mpreal_prec(prec);
#else
    mpfr::mpreal::set_default_prec(prec);
#endif
}


mpfr_prec_t get_default_prec(){
    // The lazy backend has no default of its own -- it re-sizes scratch buffers to match MPFR's,
    // so MPFR is the single source of truth either way.
    return mpfr::mpreal::get_default_prec();
}

} // namespace ode::crafted
