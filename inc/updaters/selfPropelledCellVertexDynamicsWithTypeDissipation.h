#ifndef selfPropelledCellVertexDynamicsWithTypeDissipation_H
#define selfPropelledCellVertexDynamicsWithTypeDissipation_H

#include "selfPropelledCellVertexDynamics.h"

/*! \file selfPropelledCellVertexDynamicsWithTypeDissipation.h
 *
 * Same bond-projected viscous-dissipation EOM as
 * selfPropelledCellVertexDynamicsWithAdhesion (Nguyen et al.), but the
 * coefficient is no longer a single global xi0 -- it depends on which pair
 * of CELLS borders each vertex-vertex bond:
 *
 *   chi_ij = chiSame  if the two cells bordering edge (i,j) are the same type
 *          = chiDiff  otherwise
 *
 *   mu * r_dot_i + sum_j chi_ij * |r_ij| * (r_ij_hat (x) r_ij_hat) * r_dot_ij
 *       = -grad_E_i + v0 * n_hat_i
 *
 * The interfacial line tension gamma_ij is untouched by this class -- run
 * with gamma1=gamma2=gamma3=0 (driver's -G/-H/-I flags) to isolate purely
 * kinetic (dissipative) heterogeneity from the usual energetic one.
 *
 * The two cells bordering edge (i,j) are found as the intersection of vertex
 * i's and vertex j's three neighboring cells (vertexCellNeighbors) -- every
 * interior vertex-model edge is shared by exactly two cells, so this
 * intersection has size 2.
 */
class selfPropelledCellVertexDynamicsWithTypeDissipation : public selfPropelledCellVertexDynamics
{
public:
    selfPropelledCellVertexDynamicsWithTypeDissipation(int _Ncells, int _Nvertices);

    //! Dissipation coefficient for edges bordered by two cells of the same type
    Dscalar chiSame = 0.0;
    //! Dissipation coefficient for edges bordered by two cells of different types
    Dscalar chiDiff = 0.0;

    void setHomotypicDissipation(Dscalar c) { chiSame = c; }
    void setHeterotypicDissipation(Dscalar c) { chiDiff = c; }

    //! CPU integrator: builds and solves M v = f via Eigen SimplicialLDLT (M is symmetric PD)
    virtual void integrateEquationsOfMotionCPU() override;

    //! GPU path not implemented; warns and falls back to CPU
    virtual void integrateEquationsOfMotionGPU() override;
};

#endif
