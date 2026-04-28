#define ENABLE_CUDA

#include "vertexQuadraticEnergyWithTension.h"
#include "vertexQuadraticEnergyWithTension.cuh"
/*! \file vertexQuadraticEnergyWithTension.cpp */

/*!
This function defines a matrix, \gamma_{i,j}, describing the imposed tension between cell types i and
j. This function both sets that matrix and sets the flag telling the computeForces function to call
the more general tension force computations.
\pre the vector has n^2 elements, where n is the number of types, and the values of type in the system
must be of the form 0, 1, 2, ...n. The input vector should be laid out as:
gammas[0] = g_{0,0}  (an irrelevant value that is never called)
gammas[1] = g_{0,1}
gammas[n] = g_{0,n}
gammas[n+1] = g_{1,0} (physically, this better be the same as g_{0,1})
gammas[n+2] = g_{1,1} (again, never used)
...
gammas[n^2-1] = g_{n,n}
*/
void VertexQuadraticEnergyWithTension::setSurfaceTension(vector<Dscalar> gammas)
    {
    simpleTension = false;
    //set the tension matrix to the right size, and the indexer
    tensionMatrix.resize(gammas.size());
    int n = sqrt(gammas.size());
    cellTypeIndexer = Index2D(n);

    ArrayHandle<Dscalar> tensions(tensionMatrix,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < gammas.size(); ++ii)
        {
        int typeI = ii/n;
        int typeJ = ii - typeI*n;
        tensions.data[cellTypeIndexer(typeJ,typeI)] = gammas[ii];
        };
    };

/*!
goes through the process of computing the forces on either the CPU or GPU, either with or without
exclusions, as determined by the flags. Assumes the geometry has NOT yet been computed.
\post the geometry is computed, and force per cell is computed.
*/
void VertexQuadraticEnergyWithTension::computeForces()
    {
    if(forcesUpToDate)
       return; 
    forcesUpToDate = true;
    computeGeometry();
    if (GPUcompute)
        {
        if (simpleTension)
            computeVertexSimpleTensionForceGPU();
        else if (Tension)
            computeVertexTensionForceGPU();
        }
    else
        {
        if(Tension)
                computeVertexTensionForcesCPU();
        else
            computeForcesCPU();
        };
    };

/*!
Use the data pre-computed in the geometry routine to rapidly compute the net force on each vertex...for the cpu part combine the simple and complex tension routines
*/
void VertexQuadraticEnergyWithTension::computeVertexTensionForcesCPU()
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_tm(tensionMatrix,access_location::host,access_mode::read);

    // Update the cell positions
    getCellCentroids();
    ArrayHandle<Dscalar2> h_cpos(cellPositions, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_f(vertexForces,access_location::host, access_mode::overwrite);

    //first, compute the contribution to the force on each vertex from each of its three cells
    Dscalar2 vlast,vcur,vnext;
    Dscalar2 dEdv;
    Dscalar Adiff, Pdiff;
    for(int fsidx = 0; fsidx < Nvertices*3; ++fsidx)
        {
        //for the change in the energy of the cell, just repeat the vertexQuadraticEnergy part
        int cellIdx1 = h_vcn.data[fsidx];
        Dscalar Adiff = KA*(h_AP.data[cellIdx1].x - h_APpref.data[cellIdx1].x);
        Dscalar Pdiff = KP*(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y);
        vcur = h_vc.data[fsidx];
        vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
        vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;

        //computeForceSetVertexModel is defined in inc/utility/functions.h
        computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
        h_fs.data[fsidx].x = dEdv.x;
        h_fs.data[fsidx].y = dEdv.y;

        //first, determine the index of the cell other than cellIdx1 that contains both vcur and vnext
        int cellNeighs = h_cvn.data[cellIdx1];
        //find the index of vcur and vnext
        int vCurIdx = fsidx/3;
        int vNextInt = 0;
        if (h_cv.data[n_idx(cellNeighs-1,cellIdx1)] != vCurIdx)
            {
            for (int nn = 0; nn < cellNeighs-1; ++nn)
                {
                int idx = h_cv.data[n_idx(nn,cellIdx1)];
                if (idx == vCurIdx)
                    vNextInt = nn +1;
                };
            };
        int vNextIdx = h_cv.data[n_idx(vNextInt,cellIdx1)];

        //vcur belongs to three cells... which one isn't cellIdx1 and has both vcur and vnext?
        int cellIdx2 = 0;
        int cellOfSet = fsidx-3*vCurIdx;
        for (int cc = 0; cc < 3; ++cc)
            {
            if (cellOfSet == cc) continue;
            int cell2 = h_vcn.data[3*vCurIdx+cc];
            int cNeighs = h_cvn.data[cell2];
            for (int nn = 0; nn < cNeighs; ++nn)
                if (h_cv.data[n_idx(nn,cell2)] == vNextIdx)
                    cellIdx2 = cell2;
            }
        //now, determine the types of the two relevant cells, and add an extra force if needed
        int cellType1 = h_ct.data[cellIdx1];
        int cellType2 = h_ct.data[cellIdx2];
        
        //add line tension force regardless of cell types
        Dscalar gammaEdge;
        if (simpleTension)
            gammaEdge = gamma;
        else
            gammaEdge = h_tm.data[cellTypeIndexer(cellType1,cellType2)];
        Dscalar2 dnext = vcur-vnext;
        Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        h_fs.data[fsidx].x -= gammaEdge*dnext.x/dnnorm;
        h_fs.data[fsidx].y -= gammaEdge*dnext.y/dnnorm;
        };

    //now sum these up to get the force on each vertex
    for (int v = 0; v < Nvertices; ++v)
        {
        Dscalar2 ftemp = make_Dscalar2(0.0,0.0);
        for (int ff = 0; ff < 3; ++ff)
            {
            ftemp.x += h_fs.data[3*v+ff].x;
            ftemp.y += h_fs.data[3*v+ff].y;
            };
        h_f.data[v] = ftemp;
        };

    //add cell-cell repulsion forces if enabled
    if(repulsion) 
        {
        for (int cellIdx1 = 0; cellIdx1 < Ncells; ++cellIdx1) 
            {
            Dscalar2 cellPos1 = h_cpos.data[cellIdx1];
            int cellNeighs1 = h_cvn.data[cellIdx1];
            
            for (int cellIdx2 = cellIdx1 + 1; cellIdx2 < Ncells; ++cellIdx2) 
                {
                Dscalar2 delta;
                Box->minDist(cellPos1, h_cpos.data[cellIdx2], delta);
                Dscalar r = sqrt(delta.x * delta.x + delta.y * delta.y);

                if (r < repulsionRange) 
                    {
                    Dscalar F = -0.5 * repulsionStrength * (r - repulsionRange);
                    Dscalar2 force = make_Dscalar2(F * delta.x / r, F * delta.y / r);

                    //apply forces to vertices of both cells
                    for (int nn = 0; nn < cellNeighs1; ++nn) 
                        {
                        int vidx = h_cv.data[n_idx(nn, cellIdx1)];
                        h_f.data[vidx].x += force.x;
                        h_f.data[vidx].y += force.y;
                        };

                    int cellNeighs2 = h_cvn.data[cellIdx2];
                    for (int nn = 0; nn < cellNeighs2; ++nn) 
                        {
                        int vidx = h_cv.data[n_idx(nn, cellIdx2)];
                        h_f.data[vidx].x -= force.x;
                        h_f.data[vidx].y -= force.y;
                        };
                    };
                };
            };
        };
    };

/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A_i-A_i,0)^2 + K_P(P_i-P_i,0)^2 + \sum_{[i]\neq[j]} \gamma_{[i][j]}l_{ij}
By UR: This is a very lazy implementation, but it should work.
*/
Dscalar VertexQuadraticEnergyWithTension::computeEnergy() {
    if (!forcesUpToDate)
        computeForces();

    ArrayHandle<int> h_vcn(vertexCellNeighbors, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur, access_location::host, access_mode::read);
    ArrayHandle<Dscalar4> h_vln(voroLastNext, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_m(Moduli, access_location::host, access_mode::overwrite);
    ArrayHandle<int> h_ct(cellType, access_location::host, access_mode::read);
    ArrayHandle<int> h_cv(cellVertices, access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum, access_location::host, access_mode::read);
    ArrayHandle<Dscalar> h_tm(tensionMatrix, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_cpos(cellPositions, access_location::host, access_mode::read);

    Dscalar totalEnergy = 0.0;

    for (int cellIdx1 = 0; cellIdx1 < Ncells; ++cellIdx1) 
        {
        Dscalar2 AP = h_AP.data[cellIdx1];
        Dscalar2 APpref = h_APpref.data[cellIdx1];
        Dscalar2 m = h_m.data[cellIdx1];
        Dscalar2 cellPos1 = h_cpos.data[cellIdx1];
        int cellNeighs1 = h_cvn.data[cellIdx1];
        int cellType1 = h_ct.data[cellIdx1];

        // Area and perimeter energy
        totalEnergy += m.x * (AP.x - APpref.x) * (AP.x - APpref.x) +
                      m.y * (AP.y - APpref.y) * (AP.y - APpref.y);

        // Process vertices and tension energy
        for (int nn = 0; nn < cellNeighs1; ++nn) 
            {
            int vCurIdx = h_cv.data[n_idx(nn, cellIdx1)];
            int vNextIdx = (nn == cellNeighs1 - 1) ? 
                          h_cv.data[n_idx(0, cellIdx1)] : 
                          h_cv.data[n_idx(nn + 1, cellIdx1)];

            for (int cc = 0; cc < 3; ++cc) 
                {
                int fsidx = 3 * vCurIdx + cc;
                int cellOfSet = fsidx - 3 * vCurIdx;
                
                if (cellOfSet == cc) continue;

                Dscalar2 vcur = h_vc.data[fsidx];
                Dscalar2 vnext = make_Dscalar2(h_vln.data[fsidx].z, h_vln.data[fsidx].w);

                // Find second cell
                int cellIdx2 = -1;
                for (int i = 0; i < 3; ++i) 
                    {
                    if (i == cellOfSet) continue;
                    int cell2 = h_vcn.data[3 * vCurIdx + i];
                    int cNeighs = h_cvn.data[cell2];
                    
                    for (int k = 0; k < cNeighs; ++k) 
                        {
                        if (h_cv.data[n_idx(k, cell2)] == vNextIdx) 
                            {
                            cellIdx2 = cell2;
                            break;
                            }
                        }
                    if (cellIdx2 != -1) break;
                    }

                // Compute tension energy
                Dscalar gammaEdge = simpleTension ? 
                                 gamma : 
                                 h_tm.data[cellTypeIndexer(cellType1, h_ct.data[cellIdx2])];
                Dscalar2 dnext = vcur - vnext;
                Dscalar dnnorm = sqrt(dnext.x * dnext.x + dnext.y * dnext.y);
                totalEnergy += (1.0 / 3.0) * gammaEdge * dnnorm;

                // Cell-cell repulsion energy
                if(repulsion) 
                {
                    for (int cellIdx2 = cellIdx1 + 1; cellIdx2 < Ncells; ++cellIdx2) 
                    {
                        Dscalar2 delta = make_Dscalar2(
                            cellPos1.x - h_cpos.data[cellIdx2].x,
                            cellPos1.y - h_cpos.data[cellIdx2].y
                        );
                        Dscalar r = sqrt(delta.x * delta.x + delta.y * delta.y);
        
                        if (r < repulsionRange) {
                            totalEnergy += 0.5 * repulsionStrength * (r - 0.5) * (r - 0.5);
                            }
                        }
                    }
                }

                }
            }

    return totalEnergy;
    }

void VertexQuadraticEnergyWithTension::computeVertexSimpleTensionForceGPU()
    {
    printf("computeVertexSimpleTensionForceGPU function not written. Very sorry\n");
    throw std::exception();
    };
void VertexQuadraticEnergyWithTension::computeVertexTensionForceGPU()
    {
    printf("computeVertexTensionForceGPU function not written. Very sorry\n");
    throw std::exception();
    };

