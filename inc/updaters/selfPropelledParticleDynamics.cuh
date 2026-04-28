#ifndef __SELFPROPELLEDPARTICLEDYNAMICS_CUH__
#define __SELFPROPELLEDPARTICLEDYNAMICS_CUH__

#include "std_include.h"
#include <cuda_runtime.h>

/*!
 \file selfPropelledParticleDynamics.cuh
A file providing an interface to the relevant cuda calls for the selfPropelledParticleDynamics class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!set the vector of displacements from forces and activity
bool gpu_spp_eom_integration(
                    Dscalar2 *forces,
                    Dscalar2 *velocities,
                    Dscalar2 *displacements,
                    Dscalar3 *motility,
		            int *cellType,//PS added
                    Dscalar *cellDirectors,
                    curandState *RNGs,
                    int N,
                    Dscalar deltaT,
                    int Timestep,
                    Dscalar mu);

/** @} */ //end of group declaration
 #endif

