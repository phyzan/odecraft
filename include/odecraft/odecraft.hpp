#ifndef ODECRAFT_HPP
#define ODECRAFT_HPP

#include <polybox/polybox.hpp>
#include <odecraft/Tools.hpp>
#include <odecraft/Core/SolverFactory.hpp>
#include <odecraft/DenseOde/OdeInt_impl.hpp>

#include <odecraft/Core/BaseSolver_impl.hpp>
#include <odecraft/Core/RichBase_impl.hpp>
#include <odecraft/Core/Events_impl.hpp>
#include <odecraft/Core/ObjectiveSolver_impl.hpp>

#include <odecraft/OdeResult/OdeResult_impl.hpp>

#include <odecraft/Interpolation/NdInterpolator_impl.hpp>
#include <odecraft/Interpolation/VectorFields_impl.hpp>
#include <odecraft/Interpolation/Regular/Grids_impl.hpp>
#include <odecraft/Interpolation/Regular/RegularGridInterpolator_impl.hpp>
#include <odecraft/Interpolation/Scattered/Delaunay_impl.hpp>
#include <odecraft/Interpolation/Scattered/ScatteredNdInterpolator_impl.hpp>
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>

#include <odecraft/Integrators/Solvers_impl.hpp>

#include <odecraft/Chaos/VariationalSolvers_impl.hpp>


/**
 * @file odecraft.hpp
 * @brief Main include file for odecraft library.
 *
 * This file includes all necessary headers to use the odecraft library for
 * solving ordinary differential equations (ODEs). It provides access to
 * various solver implementations, tools, and utilities.
 *
 * @author Foivos Zanias
 *
 * @note Make sure to include this file in your project to utilize odecraft's
 *       functionalities.
 * @note Compile with -DODECRAFT_NO_WARN to turn off unnecessary warnings that the solvers may throw on the console
 */

#endif // ODECRAFT_HPP