#define ENABLE_CUDA

#include "selfPropelledCellVertexDynamics.h"
#include "selfPropelledParticleDynamics.cuh"
#include "selfPropelledCellVertexDynamics.cuh"
/*! \file selfPropelledCellVertexDynamics.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
selfPropelledCellVertexDynamics::selfPropelledCellVertexDynamics(int _Ncells, int _Nvertices)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = true;
    mu = 1.0;
    Ndof = _Nvertices;
    Nvertices = _Nvertices;
    Ncells = _Ncells;
    noise.initialize(Ncells);
    displacements.resize(Nvertices);
    };

/*!
Advances self-propelled dynamics with random noise in the director by one time step
*/
void selfPropelledCellVertexDynamics::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (activeModel->getNumberOfDegreesOfFreedom() != Nvertices)
        {
        Nvertices = activeModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Nvertices);
        Ncells = Nvertices / 2;
        noise.initialize(Ncells);
        };
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
    }

/*!
The straightforward GPU implementation
*/
void selfPropelledCellVertexDynamics::integrateEquationsOfMotionGPU()
    {
    activeModel->computeForces();
    {//scope for array Handles
    ArrayHandle<Dscalar2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<Dscalar> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar3> d_motility(activeModel->Motility,access_location::device,access_mode::read);
    ArrayHandle<int> d_vcn(activeModel->vertexCellNeighbors,access_location::device,access_mode::read);

    ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);

    gpu_spp_cellVertex_eom_integration(d_f.data,
                 d_disp.data,
                 d_motility.data,
                 d_cd.data,
                 d_vcn.data,
                 d_RNG.data,
                 Nvertices,Ncells,
                 deltaT,
                 Timestep,
                 mu);
    };//end array handle scope
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    };

/*!
Move every vertex according to the net force on it and its motility...CPU routine
*/
void selfPropelledCellVertexDynamics::integrateEquationsOfMotionCPU()
{
    activeModel->computeForces();
    {//scope for array Handles
        ArrayHandle<Dscalar2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
        ArrayHandle<Dscalar> h_cd(activeModel->cellDirectors,access_location::host,access_mode::readwrite);
        ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);
        ArrayHandle<Dscalar3> h_motility(activeModel->Motility,access_location::host,access_mode::read);
        ArrayHandle<int> h_vcn(activeModel->vertexCellNeighbors,access_location::host,access_mode::read);
        ArrayHandle<int> h_ct(activeModel->cellType,access_location::host,access_mode::read);

        Dscalar directorx, directory;
        for (int i = 0; i < Nvertices; ++i)
        {
            Dscalar v1 = h_motility.data[h_vcn.data[3*i]].x;
            Dscalar v2 = h_motility.data[h_vcn.data[3*i+1]].x;
            Dscalar v3 = h_motility.data[h_vcn.data[3*i+2]].x;
            //for uniform v0, the vertex director is the straight average of the directors of the cell neighbors
            directorx  = v1*cos(h_cd.data[ h_vcn.data[3*i] ]);
            directorx += v2*cos(h_cd.data[ h_vcn.data[3*i+1] ]);
            directorx += v3*cos(h_cd.data[ h_vcn.data[3*i+2] ]);
            directorx /= 3.0;
            directory  = v1*sin(h_cd.data[ h_vcn.data[3*i] ]);
            directory += v2*sin(h_cd.data[ h_vcn.data[3*i+1] ]);
            directory += v3*sin(h_cd.data[ h_vcn.data[3*i+2] ]);
            directory /= 3.0;

            //UR pins type 5 cells to the wall
            // Is a bit tricky - if a vertex is shared by type 5 cell, pin it!
            bool pin_vertex = false;
            for (int cc = 0; cc < 3; ++cc)
            {
                // Type 5 cell found!
                if (h_ct.data[h_vcn.data[3*i+cc]] == 5)
                {
                    pin_vertex = true;
                }
            }
            if (pin_vertex)
            {
                h_disp.data[i].x = 0;
                h_disp.data[i].y = 0;
            }
            else
            {
                h_disp.data[i].x = deltaT*(directorx + mu*h_f.data[i].x);
                h_disp.data[i].y = deltaT*(directory + mu*h_f.data[i].y);
            }
        };

        //update cell directors
        Dscalar theta0_0 = M_PI; // Transplant
        Dscalar theta0_1 = 0.0; // Background
        for (int i = 0; i < Ncells; ++i)
        {
            Dscalar theta = h_cd.data[i];
            Dscalar randomNumber = noise.getRealNormal();
            Dscalar Dr = h_motility.data[i].y;
            Dscalar vD = h_motility.data[i].z;

            // Ornstein - Uhlenbeck Process
            if (h_ct.data[i] == 1 or h_ct.data[i] == 2)
            {
                theta = theta + (theta0_0 - theta)*vD*deltaT + randomNumber*sqrt(2.0*deltaT*Dr); 
                // Angles in [0, 2.PI]     
                theta = atan2(sin(theta), cos(theta)); 
                if (theta < 0.0)
                {
                    theta = 2.0*M_PI + theta;
                }
                h_cd.data[i] = theta;
            } 
            else if (h_ct.data[i] == 0)
            {
                theta = theta + (theta0_1 - theta)*vD*deltaT + randomNumber*sqrt(2.0*deltaT*Dr);
                // Angles in [0, 2.PI]     
                theta = atan2(sin(theta), cos(theta)); 
                if (theta < 0.0)
                {
                    theta = 2.0*M_PI + theta;
                }
                h_cd.data[i] = theta;
            }
        };
    }//end array handle scoping
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
};
