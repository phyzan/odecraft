#pragma once

/**
 * @file Crafted.hpp
 * @brief Single include for the whole pre-compiled `ode::crafted` interface.
 *
 * The header-only interface in namespace `ode` is a separate thing and is not pulled in by
 * this header: use <odecraft/odecraft.hpp> for that. The two can coexist in one program,
 * and even in one translation unit.
 *
 * @author Foivos Zanias
 */

#include "Toolkit/CraftedToolkit.hpp"                               // IWYU pragma: export
#include <odecraft/Interpolation/Univariate/CraftedStateInterp.hpp> // IWYU pragma: export
#include "Events/CraftedEvents.hpp"                                 // IWYU pragma: export
#include "OdeResult/CraftedOdeResult.hpp"                           // IWYU pragma: export
#include "Core/CraftedBase.hpp"                                     // IWYU pragma: export
#include "Steppers/CraftedSteppers.hpp"                             // IWYU pragma: export
#include "DenseOde/CraftedODE.hpp"                                  // IWYU pragma: export
#include <odecraft/Interpolation/MultiDim/CraftedNdInterp.hpp>      // IWYU pragma: export
#include "Chaos/CraftedChaos.hpp"                                   // IWYU pragma: export
