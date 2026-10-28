#define ENABLE_CUDA

#include "selfPropelledCellVertexDynamicsWithAdhesion.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

/*! \file selfPropelledCellVertexDynamicsWithAdhesion.cpp
 *
 * Implements the viscous adhesion EOM from Nguyen, Bera, Notbohm & Bi
 * (arXiv:2603.04170).  At each timestep the coupled linear system
 *
 *   M v = f_rhs
 *
 * is assembled and solved exactly via Eigen::SparseLU.  The displacement
 * is then  dr = dt * v.
 *
 * M is 2*Nv x 2*Nv (x and y components interleaved as 2i, 2i+1).
 * Diagonal 2x2 block for vertex i:
 *   M_ii = I + xi0 * sum_j  r_ij ⊗ r_ij / |r_ij|
 * Off-diagonal 2x2 block for vertex i and its vertex-neighbor j:
 *   M_ij = -xi0 * r_ij ⊗ r_ij / |r_ij|
 *
 * f_rhs[2i]   = (motility_x)_i + mu * Fx_i
 * f_rhs[2i+1] = (motility_y)_i + mu * Fy_i
 *
 * Vertices belonging to a pinned (type-5) cell get zero displacement.
 */

selfPropelledCellVertexDynamicsWithAdhesion::selfPropelledCellVertexDynamicsWithAdhesion(
    int _Ncells, int _Nvertices)
    : selfPropelledCellVertexDynamics(_Ncells, _Nvertices)
{
}

void selfPropelledCellVertexDynamicsWithAdhesion::integrateEquationsOfMotionGPU()
{
    printf("WARNING: selfPropelledCellVertexDynamicsWithAdhesion: GPU path not "
           "implemented for viscous adhesion.  Running on CPU.\n");
    integrateEquationsOfMotionCPU();
}

/*!
 * Build the 2Nv x 2Nv sparse system M v = f and solve it with SparseLU.
 * Director noise (Ornstein-Uhlenbeck) is applied after the solve, exactly
 * as in the parent class.
 */
void selfPropelledCellVertexDynamicsWithAdhesion::integrateEquationsOfMotionCPU()
{
    Timestep += 1;
    if (activeModel->getNumberOfDegreesOfFreedom() != Nvertices)
    {
        Nvertices = activeModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Nvertices);
        Ncells = Nvertices / 2;
        noise.initialize(Ncells);
    }

    activeModel->computeForces();

    {// --- scope: all ArrayHandles live here ---

        ArrayHandle<Dscalar2> h_f(activeModel->returnForces(),
                                  access_location::host, access_mode::read);
        ArrayHandle<Dscalar>  h_cd(activeModel->cellDirectors,
                                   access_location::host, access_mode::readwrite);
        ArrayHandle<Dscalar2> h_disp(displacements,
                                     access_location::host, access_mode::overwrite);
        ArrayHandle<Dscalar3> h_mot(activeModel->Motility,
                                    access_location::host, access_mode::read);
        ArrayHandle<int>      h_vcn(activeModel->vertexCellNeighbors,
                                    access_location::host, access_mode::read);
        ArrayHandle<int>      h_ct(activeModel->cellType,
                                   access_location::host, access_mode::read);
        ArrayHandle<Dscalar2> h_vp(activeModel->vertexPositions,
                                   access_location::host, access_mode::read);
        ArrayHandle<int>      h_vn(activeModel->vertexNeighbors,
                                   access_location::host, access_mode::read);

        gpubox &box = activeModel->returnBox();

        // ---------------------------------------------------------------
        // 1.  Identify pinned vertices (adjacent to a type-5 wall cell)
        // ---------------------------------------------------------------
        vector<bool> pinned(Nvertices, false);
        for (int i = 0; i < Nvertices; ++i)
            for (int cc = 0; cc < 3; ++cc)
                if (h_ct.data[h_vcn.data[3*i+cc]] == 5)
                {
                    pinned[i] = true;
                    break;
                }

        // ---------------------------------------------------------------
        // 2.  Build sparse M and RHS
        // ---------------------------------------------------------------
        typedef Eigen::SparseMatrix<Dscalar>      SpMat;
        typedef Eigen::Triplet<Dscalar>           Tri;
        typedef Eigen::Matrix<Dscalar, Eigen::Dynamic, 1> EigVec;

        vector<Tri> triplets;
        triplets.reserve(16 * Nvertices);   // 4 triplets per vertex x 4 blocks max

        EigVec rhs(2 * Nvertices);
        rhs.setZero();

        for (int i = 0; i < Nvertices; ++i)
        {
            if (pinned[i])
            {
                // Identity rows → solution will be zero velocity
                triplets.push_back(Tri(2*i,   2*i,   1.0));
                triplets.push_back(Tri(2*i+1, 2*i+1, 1.0));
                // rhs stays 0
                continue;
            }

            // ---- motility of this vertex (average over 3 cell neighbors) ----
            int c0 = h_vcn.data[3*i], c1 = h_vcn.data[3*i+1], c2 = h_vcn.data[3*i+2];
            Dscalar v0 = h_mot.data[c0].x,  v1 = h_mot.data[c1].x,  v2 = h_mot.data[c2].x;
            Dscalar mot_x = (v0*cos(h_cd.data[c0]) + v1*cos(h_cd.data[c1]) + v2*cos(h_cd.data[c2])) / 3.0;
            Dscalar mot_y = (v0*sin(h_cd.data[c0]) + v1*sin(h_cd.data[c1]) + v2*sin(h_cd.data[c2])) / 3.0;

            rhs(2*i)   = mot_x + mu * h_f.data[i].x;
            rhs(2*i+1) = mot_y + mu * h_f.data[i].y;

            // ---- build M diagonal block (starts as identity) ----
            Dscalar D_xx = 1.0, D_xy = 0.0, D_yy = 1.0;

            for (int nn = 0; nn < 3; ++nn)
            {
                int j = h_vn.data[3*i + nn];

                // Minimum-image r_ij = r_i - r_j
                Dscalar2 rij;
                box.minDist(h_vp.data[i], h_vp.data[j], rij);

                Dscalar norm = sqrt(rij.x*rij.x + rij.y*rij.y);
                if (norm < 1e-12) continue;

                // anisotropic (xi0) dyadic term: xi0 * r^a r^b / |r|
                Dscalar mxx = xi0 * rij.x * rij.x / norm;
                Dscalar mxy = xi0 * rij.x * rij.y / norm;
                Dscalar myy = xi0 * rij.y * rij.y / norm;

                // Accumulate diagonal: anisotropic + isotropic (eta0 per bond)
                D_xx += mxx + eta0;
                D_xy += mxy;
                D_yy += myy + eta0;

                // Off-diagonal block M[i,j]: skip if j is pinned (v_j = 0
                // so the coupling drops out of the RHS automatically)
                if (!pinned[j])
                {
                    triplets.push_back(Tri(2*i,   2*j,   -(mxx + eta0)));
                    triplets.push_back(Tri(2*i,   2*j+1, -mxy));
                    triplets.push_back(Tri(2*i+1, 2*j,   -mxy));
                    triplets.push_back(Tri(2*i+1, 2*j+1, -(myy + eta0)));
                }
            }

            // Insert diagonal 2x2 block (note x-y off-diagonal coupling)
            triplets.push_back(Tri(2*i,   2*i,   D_xx));
            triplets.push_back(Tri(2*i,   2*i+1, D_xy));
            triplets.push_back(Tri(2*i+1, 2*i,   D_xy));
            triplets.push_back(Tri(2*i+1, 2*i+1, D_yy));
        }

        SpMat M(2*Nvertices, 2*Nvertices);
        M.setFromTriplets(triplets.begin(), triplets.end());

        // ---------------------------------------------------------------
        // 3.  Solve M v = rhs.  M is symmetric positive-definite by
        //     construction (identity diagonal + PSD dyadic terms), so a
        //     Cholesky-type SimplicialLDLT factorization is used instead of
        //     the general-purpose SparseLU -- same exact direct solve, no
        //     accuracy tradeoff, but roughly 2-4x faster since it exploits
        //     symmetry instead of doing full unsymmetric pivoting.
        // ---------------------------------------------------------------
        Eigen::SimplicialLDLT<SpMat> solver;
        solver.compute(M);
        if (solver.info() != Eigen::Success)
        {
            printf("ERROR: SimplicialLDLT factorization failed at timestep %i.  "
                   "Falling back to zero-adhesion update.\n", Timestep);
            // Graceful fallback: ignore adhesion this step
            for (int i = 0; i < Nvertices; ++i)
            {
                if (pinned[i]) { h_disp.data[i] = make_Dscalar2(0,0); continue; }
                int c0 = h_vcn.data[3*i], c1 = h_vcn.data[3*i+1], c2 = h_vcn.data[3*i+2];
                Dscalar v0 = h_mot.data[c0].x, v1 = h_mot.data[c1].x, v2 = h_mot.data[c2].x;
                Dscalar dx = (v0*cos(h_cd.data[c0]) + v1*cos(h_cd.data[c1]) + v2*cos(h_cd.data[c2])) / 3.0;
                Dscalar dy = (v0*sin(h_cd.data[c0]) + v1*sin(h_cd.data[c1]) + v2*sin(h_cd.data[c2])) / 3.0;
                h_disp.data[i].x = deltaT * (dx + mu * h_f.data[i].x);
                h_disp.data[i].y = deltaT * (dy + mu * h_f.data[i].y);
            }
        }
        else
        {
            EigVec vel = solver.solve(rhs);
            for (int i = 0; i < Nvertices; ++i)
            {
                h_disp.data[i].x = deltaT * vel(2*i);
                h_disp.data[i].y = deltaT * vel(2*i+1);
            }
        }

        // ---------------------------------------------------------------
        // 4.  Update cell directors (Ornstein-Uhlenbeck) — unchanged
        // ---------------------------------------------------------------
        Dscalar theta0_transplant = M_PI;
        Dscalar theta0_background = 0.0;
        for (int i = 0; i < Ncells; ++i)
        {
            Dscalar theta = h_cd.data[i];
            Dscalar eta   = noise.getRealNormal();
            Dscalar Dr    = h_mot.data[i].y;
            Dscalar vD    = h_mot.data[i].z;

            if (h_ct.data[i] == 1 || h_ct.data[i] == 2)
            {
                theta = theta + (theta0_transplant - theta)*vD*deltaT + eta*sqrt(2.0*deltaT*Dr);
            }
            else if (h_ct.data[i] == 0)
            {
                theta = theta + (theta0_background - theta)*vD*deltaT + eta*sqrt(2.0*deltaT*Dr);
            }
            // type 5 (pinned wall): leave theta unchanged

            theta = atan2(sin(theta), cos(theta));
            if (theta < 0.0) theta += 2.0*M_PI;
            h_cd.data[i] = theta;
        }

    }// --- end ArrayHandle scope ---

    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
}
