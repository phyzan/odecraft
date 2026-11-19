#include "scalars.hpp"

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace odetest {

// Sets the precision of mpfr::mpreal once, before any test case runs. Through lazex, so that its
// scratch buffers for LazyType<mpfr::mpreal> follow as well.
class MultiprecisionSetup : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo& /*info*/) override {
        lazex::set_default_mpreal_prec(mpreal_bits);
    }
};

} // namespace odetest

CATCH_REGISTER_LISTENER(odetest::MultiprecisionSetup)
