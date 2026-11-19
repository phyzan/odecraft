#include <odecraft/Crafted.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include "../support/scalars.hpp"

namespace odetest {

// Sets the precision of the multiprecision scalar once, before any test case runs, through the
// compiled interface's own setter: whichever type mpreal_t is, its precision follows.
class CraftedMultiprecisionSetup : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo& /*info*/) override {
        ode::crafted::set_mpreal_prec(mpreal_bits);
    }
};

} // namespace odetest

CATCH_REGISTER_LISTENER(odetest::CraftedMultiprecisionSetup)
