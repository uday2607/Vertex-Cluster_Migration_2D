#ifndef selfPropelledCellVertexDynamicsWithAdhesion_H
#define selfPropelledCellVertexDynamicsWithAdhesion_H

#include "selfPropelledCellVertexDynamics.h"

/*! \file selfPropelledCellVertexDynamicsWithAdhesion.h
 *
 * Extends selfPropelledCellVertexDynamics with viscous cell-cell coupling.
 *
 * The standard overdamped vertex EOM becomes:
 *
 *   mu * r_dot_i + sum_{j in S_i} [xi*(r_ij.r_dot_ij)*r_ij/|r_ij| + eta*(r_dot_i - r_dot_j)]
 *       = -grad_E_i + v0 * n_hat_i
 *
 * which in matrix form is  M v = f,  solved exactly at every timestep.
 * The 2Nv x 2Nv sparse symmetric matrix M has entries:
 *
 *   M_ii^{ab} = (1 + n_i*eta0)*delta_{ab} + xi0 * sum_j  r_ij^a * r_ij^b / |r_ij|
 *   M_ij^{ab} = -eta0*delta_{ab} - xi0 * r_ij^a * r_ij^b / |r_ij|   for j in S_i
 *
 * xi0:  anisotropic longitudinal coupling (Nguyen et al. arXiv:2603.04170)
 * eta0: isotropic viscous coupling (Rozman-style scalar damping per bond)
 *
 * The time step update is then:  r(t+dt) = r(t) + dt * M^{-1} f(t)
 *
 * GPU integration is NOT implemented; the class always falls back to CPU.
 */
class selfPropelledCellVertexDynamicsWithAdhesion : public selfPropelledCellVertexDynamics
{
public:
    selfPropelledCellVertexDynamicsWithAdhesion(int _Ncells, int _Nvertices);

    //! Dimensionless anisotropic viscous adhesion coefficient (Nguyen et al.)
    Dscalar xi0  = 0.0;
    //! Dimensionless isotropic viscous coupling coefficient (Rozman-style)
    Dscalar eta0 = 0.0;

    //! Set the anisotropic viscous adhesion strength
    void setViscousAdhesion(Dscalar xi)    { xi0  = xi;  }
    //! Set the isotropic viscous coupling strength
    void setIsotropicViscosity(Dscalar eta){ eta0 = eta; }

    //! CPU integrator: builds and solves M v = f via Eigen SimplicialLDLT (M is symmetric PD)
    virtual void integrateEquationsOfMotionCPU() override;

    //! GPU path not implemented for adhesion; warns and falls back to CPU
    virtual void integrateEquationsOfMotionGPU() override;
};

#endif
