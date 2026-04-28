#define ENABLE_CUDA

#include "selfPropelledParticleDynamics.h"
#include "selfPropelledParticleDynamics.cuh"
/*! \file selfPropelledParticleDynamics.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
selfPropelledParticleDynamics::selfPropelledParticleDynamics(int _N, double _mu)
    {
    Timestep = 0;
    deltaT = 0.001;
    GPUcompute = true;
    mu = _mu;
    Ndof = _N;
    noise.initialize(Ndof);
    displacements.resize(Ndof);
    };

/*!
When spatial sorting is performed, re-index the array of cuda RNGs... This function is currently
commented out, for greater flexibility (i.e., to not require that the indexToTag (or Itt) be the
re-indexing array), since that assumes cell and not particle-based dynamics
*/
void selfPropelledParticleDynamics::spatialSorting()
    {
    //reIndexing = activeModel->returnItt();
    //reIndexRNG(noise.RNGs);
    };

/*!
Set the shared pointer of the base class to passed variable; cast it as an active cell model
*/
void selfPropelledParticleDynamics::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    activeModel = dynamic_pointer_cast<Simple2DActiveCell>(model);
    }

/*!
Advances self-propelled dynamics with random noise in the director by one time step
*/
void selfPropelledParticleDynamics::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (activeModel->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = activeModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        noise.initialize(Ndof);
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
The straightforward CPU implementation
*/
void selfPropelledParticleDynamics::integrateEquationsOfMotionCPU()
{
    activeModel->computeForces();
    {
        //scope for array Handles
        ArrayHandle<Dscalar2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
        ArrayHandle<Dscalar> h_cd(activeModel->cellDirectors);
        ArrayHandle<Dscalar2> h_v(activeModel->cellVelocities);
        ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);
        ArrayHandle<Dscalar3> h_motility(activeModel->Motility,access_location::host,access_mode::read);

        //PS call celltype additionally to selctively pin type 5
        ArrayHandle<int> h_ct(activeModel->cellType,access_location::host,access_mode::read);

        //UR assumes that d\theta_i = vD*(theta0_i - theta_i) + sqrt(2*Dr)*W_i
        Dscalar theta0_0 = M_PI; // Transplant
        Dscalar theta0_1 = 0.0; // Background
 
        for (int ii = 0; ii < Ndof; ++ii)
        {
            //displace according to current velocities and forces
            Dscalar v0i = h_motility.data[ii].x;
            Dscalar Dri = h_motility.data[ii].y;
            Dscalar vD = h_motility.data[ii].z;
            h_v.data[ii].x =  v0i * cos(h_cd.data[ii]);
            h_v.data[ii].y =  v0i * sin(h_cd.data[ii]);
            Dscalar2 Vcur = h_v.data[ii];
	    
            //PS pins type 5
	        if (h_ct.data[ii] == 5)
            {
	            h_disp.data[ii].x = 0;
	            h_disp.data[ii].y = 0;        	
	        }
	        else
            {
                h_disp.data[ii].x = deltaT*(Vcur.x + mu * h_f.data[ii].x);
                h_disp.data[ii].y = deltaT*(Vcur.y + mu * h_f.data[ii].y);
	        }
	
            Dscalar theta = h_cd.data[ii];

            Dscalar randomNumber = noise.getRealNormal();

            // Ornstein - Uhnbleck Process
            if (h_ct.data[ii] == 1 or h_ct.data[ii] == 2)
            {
	            theta = theta + (theta0_0 - theta)*vD*deltaT + randomNumber*sqrt(2.0*deltaT*Dri); 
                // Angles in [0, 2.PI]     
                theta = atan2(sin(theta), cos(theta)); 
                if (theta < 0.0)
                {
                    theta = 2.0*M_PI + theta;
                }
                h_cd.data[ii] = theta;

	        } 
            else if (h_ct.data[ii] == 0)
            {
                theta = theta + (theta0_1 - theta)*vD*deltaT + randomNumber*sqrt(2.0*deltaT*Dri);
                // Angles in [0, 2.PI]     
                theta = atan2(sin(theta), cos(theta)); 
                if (theta < 0.0)
                {
                    theta = 2.0*M_PI + theta;
                }
                h_cd.data[ii] = theta;
            }
            
        };
    }//end array handle scoping
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    //vector of displacements is mu*forces*timestep + v0's*timestep
};

/*!
The straightforward GPU implementation
*/
void selfPropelledParticleDynamics::integrateEquationsOfMotionGPU()
{
    activeModel->computeForces();
    {//scope for array Handles
        ArrayHandle<Dscalar2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
        ArrayHandle<Dscalar> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
        ArrayHandle<Dscalar2> d_v(activeModel->cellVelocities,access_location::device,access_mode::readwrite);
        ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::overwrite);
        ArrayHandle<Dscalar3> d_motility(activeModel->Motility,access_location::device,access_mode::read);
        ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);

        //PS call celltype additionally to selctively pin type 10  
        ArrayHandle<int> d_ct(activeModel->cellType,access_location::host,access_mode::read);
    
        gpu_spp_eom_integration(d_f.data,
                     d_v.data,
                     d_disp.data,
                     d_motility.data,
	    	         d_ct.data, //PS added celltype pointer(?)
                     d_cd.data,
                     d_RNG.data,
                     Ndof,
                     deltaT,
                     Timestep,
                     mu);
    };//end array handle scope
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
};
