#pragma once


#include <cstdint>
#include <string_view>


namespace ode{

enum class StepperStatus : std::uint8_t{
    NewStep, // Stepper is running, located at a new adapted step
    BetweenSteps, // Stepper is running, located between two adapted steps
    Uninitialized, // Initial conditions not set
    BadICS, // Invalid ics (nans or infs)
    TinyStep, // Stepsize required was near machine precision
    MinStep, // Stepsize reached minimum set by user
    MaxStep, // Stepsize reached maximum set by user
    TooSmallStep, // t + dt == t due to roundoff error
    Diverges, // State reached nan/inf
    SingularMatrix, // The linear system of an implicit step is singular
    Interrupted, // A user function threw during a step
};


inline constexpr std::string_view status_message(StepperStatus status){
    switch (status){
        case StepperStatus::Uninitialized:
            return "Uninitialized";
        case StepperStatus::BadICS:
            return "Ics contain Nan/Inf";
        case StepperStatus::TinyStep:
            return "Stepsize required is near machine precision";
        case StepperStatus::MinStep:
            return "Stepsize reached its minimum";
        case StepperStatus::MaxStep:
            return "Stepsize reached its maximum";
        case StepperStatus::TooSmallStep:
            return "Stepsize too small relative to the integration variable";
        case StepperStatus::Diverges:
            return "Integrator diverges";
        case StepperStatus::Interrupted:
            return "Step interrupted by an exception from a user function";
        case StepperStatus::SingularMatrix:
            return "Linear system of the implicit step is singular";
        default:
            return "Running";
    }
}

} // namespace ode
