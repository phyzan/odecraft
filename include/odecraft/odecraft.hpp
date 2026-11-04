#pragma once


#include <polybox/polybox.hpp>
#include "Toolkit/Toolkit.hpp" // IWYU pragma: keep
#include "Core/SolverFactory.hpp" // IWYU pragma: keep
#include "DenseOde/OdeInt_impl.hpp" // IWYU pragma: keep

#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include "Events/Events_impl.hpp" // IWYU pragma: keep
#include "Objectives/StaticEventStepper_impl.hpp" // IWYU pragma: keep

#include "OdeResult/OdeResult_impl.hpp" // IWYU pragma: keep

#include <odecraft/Interpolation/MultiDim/NdInterpolator_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Regular/Grids_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Regular/RegularGridInterpolator_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Scattered/Delaunay_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Scattered/ScatteredNdInterpolator_impl.hpp>
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>

#include <odecraft/Interpolation/MultiDim/VectorFields/VectorFields_impl.hpp>

#include "Steppers/Steppers_impl.hpp" // IWYU pragma: keep

#include "Chaos/VariationalSolvers_impl.hpp" // IWYU pragma: keep

// Contracts

#include "Contracts/DenseDescriptor.hpp" // IWYU pragma: keep

/**
 * @file odecraft.hpp
 * @brief Main umbrella for the odecraft library.
 */
