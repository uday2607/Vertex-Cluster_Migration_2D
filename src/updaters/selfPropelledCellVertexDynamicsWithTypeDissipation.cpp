#define ENABLE_CUDA

#include "selfPropelledCellVertexDynamicsWithTypeDissipation.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

/*! \file selfPropelledCellVertexDynamicsWithTypeDissipation.cpp
 *
 * Same M v = f_rhs solve as selfPropelledCellVertexDynamicsWithAdhesion, but
 * the bond-projected dissipation coefficient chi_ij is looked up per-edge
 * from the two cell types bordering that edge, instead of being a single
 * global xi0. See the header for the full equation.
 *
 * Finding the two cells bordering edge (i,j): vertex i touches 3 cells and
 * vertex j touches 3 cells; every interior edge is shared by exactly 2
 * cells, and those 2 cells must appear in BOTH vertex i's and vertex j's
 * lists of 3 neighboring cells. So the bordering pair is just the
 * intersection of the two 3-element cell-neighbor sets.
 */

selfPropelledCellVertexDynamicsWithTypeDissipation::selfPropelledCellVertexDynamicsWithTypeDissipation(
    int _Ncells, int _Nvertices)
    : selfPropelledCellVertexDynamics(_Ncells, _Nvertices)
{
}

void selfPropelledCellVertexDynamicsWithTypeDissipation::integrateEquationsOfMotionGPU()
{
    printf("WARNING: selfPropelledCellVertexDynamicsWithTypeDissipation: GPU path not "
           "implemented for type-dependent dissipation. Running on CPU.\n");
    integrateEquationsOfMotionCPU();
}

void selfPropelledCellVertexDynamicsWithTypeDissipation::integrateEquationsOfMotionCPU()
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
        triplets.reserve(16 * Nvertices);

        EigVec rhs(2 * Nvertices);
        rhs.setZero();

        for (int i = 0; i < Nvertices; ++i)
        {
            if (pinned[i])
            {
                triplets.push_back(Tri(2*i,   2*i,   1.0));
                triplets.push_back(Tri(2*i+1, 2*i+1, 1.0));
                continue;
            }

            // ---- motility of this vertex (average over 3 cell neighbors) ----
            int c0 = h_vcn.data[3*i], c1 = h_vcn.data[3*i+1], c2 = h_vcn.data[3*i+2];
            Dscalar v0 = h_mot.data[c0].x,  v1 = h_mot.data[c1].x,  v2 = h_mot.data[c2].x;
            Dscalar mot_x = (v0*cos(h_cd.data[c0]) + v1*cos(h_cd.data[c1]) + v2*cos(h_cd.data[c2])) / 3.0;
            Dscalar mot_y = (v0*sin(h_cd.data[c0]) + v1*sin(h_cd.data[c1]) + v2*sin(h_cd.data[c2])) / 3.0;

            rhs(2*i)   = mot_x + mu * h_f.data[i].x;
            rhs(2*i+1) = mot_y + mu * h_f.data[i].y;

            Dscalar D_xx = 1.0, D_xy = 0.0, D_yy = 1.0;

            int ci[3] = {c0, c1, c2};

            for (int nn = 0; nn < 3; ++nn)
            {
                int j = h_vn.data[3*i + nn];

                Dscalar2 rij;
                box.minDist(h_vp.data[i], h_vp.data[j], rij);

                Dscalar norm = sqrt(rij.x*rij.x + rij.y*rij.y);
                if (norm < 1e-12) continue;

                // ---- find the two cells bordering edge (i,j): the cells
                // common to vertex i's and vertex j's three neighboring
                // cells (every interior edge is shared by exactly two) ----
                int cj[3] = {h_vcn.data[3*j], h_vcn.data[3*j+1], h_vcn.data[3*j+2]};
                int common[2];
                int nCommon = 0;
                for (int a = 0; a < 3 && nCommon < 2; ++a)
                    for (int b = 0; b < 3; ++b)
                        if (ci[a] == cj[b]) { common[nCommon++] = ci[a]; break; }

                // fallback (should not occur for interior edges): treat as heterotypic
                Dscalar chi_ij = chiDiff;
                if (nCommon == 2)
                {
                    int typeA = h_ct.data[common[0]];
                    int typeB = h_ct.data[common[1]];
                    chi_ij = (typeA == typeB) ? chiSame : chiDiff;
                }

                // bond-projected (longitudinal) dyadic term, same form as xi0
                Dscalar mxx = chi_ij * rij.x * rij.x / norm;
                Dscalar mxy = chi_ij * rij.x * rij.y / norm;
                Dscalar myy = chi_ij * rij.y * rij.y / norm;

                D_xx += mxx;
                D_xy += mxy;
                D_yy += myy;

                if (!pinned[j])
                {
                    triplets.push_back(Tri(2*i,   2*j,   -mxx));
                    triplets.push_back(Tri(2*i,   2*j+1, -mxy));
                    triplets.push_back(Tri(2*i+1, 2*j,   -mxy));
                    triplets.push_back(Tri(2*i+1, 2*j+1, -myy));
                }
            }

            triplets.push_back(Tri(2*i,   2*i,   D_xx));
            triplets.push_back(Tri(2*i,   2*i+1, D_xy));
            triplets.push_back(Tri(2*i+1, 2*i,   D_xy));
            triplets.push_back(Tri(2*i+1, 2*i+1, D_yy));
        }

        SpMat M(2*Nvertices, 2*Nvertices);
        M.setFromTriplets(triplets.begin(), triplets.end());

        // ---------------------------------------------------------------
        // 3.  Solve M v = rhs.  M is symmetric positive-definite by
        //     construction (identity diagonal + PSD chi_ij dyadic terms),
        //     same as selfPropelledCellVertexDynamicsWithAdhesion -- use
        //     SimplicialLDLT instead of general SparseLU for the same
        //     ~2-4x speedup from exploiting symmetry.
        // ---------------------------------------------------------------
        Eigen::SimplicialLDLT<SpMat> solver;
        solver.compute(M);
        if (solver.info() != Eigen::Success)
        {
            printf("ERROR: SimplicialLDLT factorization failed at timestep %i.  "
                   "Falling back to zero-dissipation update.\n", Timestep);
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

            theta = atan2(sin(theta), cos(theta));
            if (theta < 0.0) theta += 2.0*M_PI;
            h_cd.data[i] = theta;
        }

    }// --- end ArrayHandle scope ---

    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
}
