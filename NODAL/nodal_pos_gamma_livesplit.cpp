/*
 * nodal_pos_gamma_livesplit.cpp
 * Same as nodal_pos_gamma plus an in-simulation, per-integration-step cluster-split detector (writes <name>_split_time_live.txt next to the .nc).
 *
 * Two flags are common to every driver in this repository:
 *   -O <dir>   root output directory (default: DATA). Trajectories are written to
 *              <dir>/NODAL_POS_GAMMA_LIVESPLIT/Data_t_<...parameters...>_ensem_<k>.nc
 *   -S <time>  time between saved frames during production, in simulation time
 *              units (default 5.0). Smaller values give smoother movies and bigger files.
 * All other flags are listed in the README.md of this folder.
 */
#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergyWithTension.h"
#include "selfPropelledParticleDynamics.h"
#include "analysisPackage.h"
#include "DatabaseNetCDFSPV.h"

#include <set>
#include <map>
#include <fstream>

/*
HELPER FUNCTION: Find nearest k points to a point in 2D
By UR
*/
std::vector<int> kClosest(ArrayHandle<Dscalar2> points, int num_points, Dscalar2 c_point, int k)
{

    // In multimap values gets
    // automatically sorted based on
    // their keys which is distance here
    std::multimap<double, int> mp;
    for (int i = 0; i < num_points; i++)
    {
        double x = points.data[i].x, y = points.data[i].y;
        mp.insert({sqrt((x - c_point.x) * (x - c_point.x) +
                        (y - c_point.y) * (y - c_point.y)),
                   i});
    }

    std::vector<int> output;

    for (auto it = mp.begin();
         it != mp.end() && k > 0;
         it++, k--)
    {
        output.push_back(it->second);
    }

    return output;
}

/*
HELPER FUNCTION: Find nearest k points to a point in 2D and sort by x co-ordinates
By UR
*/
std::vector<int> kClosest_x(ArrayHandle<Dscalar2> points, int num_points, Dscalar2 c_point, int k)
{

    // In multimap values gets
    // automatically sorted based on
    // their keys which is distance here
    std::multimap<double, int> mp;
    for (int i = 0; i < num_points; i++)
    {
        double x = points.data[i].x, y = points.data[i].y;
        mp.insert({sqrt((x - c_point.x) * (x - c_point.x) +
                        (y - c_point.y) * (y - c_point.y)),
                   i});
    }

    // Use second multimap for sorting using x-coords
    std::multimap<double, int> mp2;
    int i = 0;
    int k_ = k;
    for (auto it = mp.begin();
         it != mp.end() && k_ > 0;
         it++, k_--)
    {
        mp2.insert({points.data[it->second].x, it->second});
        i = i+1;
    }


    std::vector<int> output;
    for (auto it = mp2.begin();
         it != mp2.end() && k > 0;
         it++, k--)
    {
        output.push_back(it->second);
    }

    return output;
}

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will
move the positions. If you want the forces and the positions to be sync'ed, you should call the
Voronoi model's computeForces() funciton right before saving a state.
*/
#include <cerrno>
#include <cstring>
#include <sys/stat.h>

// Create every missing directory in the parent path of `outputPath` (like `mkdir -p`).
namespace
{
bool createDirectoryIfMissing(const std::string &path)
    {
    struct stat info;
    if (stat(path.c_str(), &info) == 0)
        return S_ISDIR(info.st_mode);
    if (mkdir(path.c_str(), 0755) == 0)
        return true;
    return errno == EEXIST;
    }

bool ensureParentDirectoryExists(const std::string &outputPath)
    {
    const std::size_t lastSlash = outputPath.find_last_of('/');
    if (lastSlash == std::string::npos)
        return true;
    std::string current;
    for (std::size_t i = 0; i <= lastSlash; ++i)
        {
        if (outputPath[i] == '/')
            {
            if (!current.empty() && !createDirectoryIfMissing(current))
                return false;
            }
        current.push_back(outputPath[i]);
        }
    return true;
    }
}


int main(int argc, char *argv[])
{
    //...some default parameters
    int numpts = 400;       // number of cells
    int migratingcells = 8; // number of cells in the migrating cluster-- can safely go upto 35!
                            // read more on issue#7; try to stick with even numbers please
    int activecells = migratingcells/2; // number of active cells in the migrating cluster 
                                        // Added by UR    
    int USE_GPU = -1;       // 0 or greater uses a gpu, any negative number runs on the cpu
                      //  only reason PS switched back to CPU was pinning was not as trivial
    int c;
    int tSteps = 500000; // number of time steps to run after initialization
    int ensemble = 999;
    int IC_switch = 1;        // Disc IC
    int Averaging_switch = 1; // averaging ie not reproducible
    // int program_switch = -1; //various settings control output
    int IV_switch = 1; // initial velocity allotment

    Dscalar dt = 0.0002; // the time step size (changed by UR from 0.01 -> 0.001)
    Dscalar v0 = 2;      // the self-propulsion, of the more active celltype in the transplant
    Dscalar u0 = 0.5;    // activity of the other celltype in the transplant
    Dscalar Dr = 1.0;    // the rotational diffusion constant of the cell directors
    Dscalar vD = 0.0;    // Drift constant of cell directors
    /*
    Cell directors are now modelled as a ornstein-uhlenbeck process so that the director is always
    pointing towards theta(t = 0). This makes more sense from physics POV for experimental data
    */
    Dscalar Dr_back = 0.05;                 // the rotational diffusion constant of the background cell directors (by UR)
    Dscalar v_back = 0.01; // velocity of the background cells (by UR)
    Dscalar vD_back = 0.0;                  // Drift constant of cell director of Background cells (Ornstein - Uhlenbeck process) -> By UR
    Dscalar Dr_transplant = 0.005;        // the rotational diffusion constant of transplant cell directors (by UR)
    Dscalar vD_transplant = 0.02;           // Drift constant of cell director of transplant cells (Ornstein - Uhlenbeck process) -> By UR
    int initSteps = (1.0 / (dt * 1.0)) + 1; // number of initialization steps
    int granularity = (100.0 / (dt * 1.0)); // 100000/10 at the equilibration phase just get 10 linear points

    Dscalar alpha = 1;
    Dscalar s0a = 3.7; // two different shapes possible
    Dscalar s0b = 3.7;
    Dscalar p0a = s0a;
    Dscalar p0b = s0b;
    Dscalar a0a = 1.0;
    Dscalar a0b = 1.0;
    Dscalar gamma1 = 2;
    Dscalar gamma2 = 0.25;
    Dscalar gamma3 = 0.5;
    Dscalar Ka = 10.0;
    Dscalar Kp = 1.0;
    int num_types = 5;
    int RNDseed = -1; //UR: If RNDSeed = -1, then the current time plus the ensemble number is taken as the seed
                      // Or a seed can be passed to the code.

    // The defaults can be overridden from the command line
    std::string outRoot = "DATA";   // -O
    Dscalar saveInterval = 5.0;      // -S
    while ((c = getopt(argc, argv, "a:n:b:t:G:H:I:i:e:s:q:r:u:v:k:p:f:x:m:d:j:y:c:l:S:O:")) != -1)
        switch (c)
        {
        case 'n':
            numpts = atoi(optarg);
            break;
        case 'm':
            migratingcells = atoi(optarg);
            break;
        case 'i':
            IC_switch = atoi(optarg);
            break;
        case 't':
            tSteps = atoi(optarg);
            break;
        case 'G':
            gamma1 = atof(optarg);
            break;
        case 'H':
            gamma2 = atof(optarg);
            break;
        case 'I':
            gamma3 = atof(optarg);
            break;        
        case 'e':
            dt = atof(optarg);
            break;
        case 's':
            s0a = atof(optarg);
            break;
        case 'q':
            s0b = atof(optarg);
            break;
        case 'u':
            u0 = atof(optarg);
            break;
        case 'v':
            v0 = atof(optarg);
            break;
        case 'k':
            Ka = atof(optarg);
            break;
        case 'p':
            Kp = atof(optarg);
            break;    
        case 'f':
            granularity = atof(optarg);
            break;
        case 'x':
            ensemble = atoi(optarg);
            break;
        case 'd':
            Dr_back = atof(optarg);
            break; // Added by UR
        case 'y':
            v_back = atof(optarg);
            break; // Added by UR
        case 'b':
            Dr_transplant = atof(optarg);
            break; // Added by UR
        case 'j':
            vD_transplant = atof(optarg);
            break; // Added by UR
        case 'c':
            RNDseed = atoi(optarg);
            break; // Added by UR
        case 'l':
            activecells = atoi(optarg);
            break; // Added by UR        
        case 'S':
            saveInterval = atof(optarg);
            break;
        case 'O':
            outRoot = optarg;
            break;
        case '?':
            if (optopt == 'c')
                std::cerr << "Option -" << optopt << "requires an argument.\n";
            else if (isprint(optopt))
                std::cerr << "Unknown option '-" << optopt << "'.\n";
            else
                std::cerr << "Unknown option character.\n";
            return 1;
        default:
            abort();
        };

    a0a = 2 / (1 + alpha);
    p0a = s0a * sqrt(a0a);
    a0b = 2 * alpha / (1 + alpha);
    p0b = s0b * sqrt(a0b);

    // UR: Initialize a vector of n^2 for gamma values between different cell types
    // Checkout: https://dmsussman.gitlab.io/cellGPUdocumentation/classVoronoiQuadraticEnergyWithTension.html
    vector<Dscalar> gammas((num_types+1)*(num_types+1), 0.0);
    // Line Tension is between Cell Types 0 <-> 1, 0 <-> 2, 1 <-> 2, 1 <-> 5, 2 <-> 5
    // Gamma[0, 1] = Gamma[1, 0] = gamma1
    gammas[1] = gamma1;
    gammas[num_types+1] = gamma1;
    // Gamma[2, 0] = Gamma[0, 2] = gamma2
    gammas[2] = gamma2;
    gammas[2*num_types+2] = gamma2;
    // Gamma[1, 2] = Gamma[2, 1] = gamma3
    gammas[num_types+3] = gamma3;
    gammas[2*num_types+3] = gamma3;
    // Gamma[1, 5] = Gamma[5, 1] = 20
    gammas[num_types+6] = 20;
    gammas[(num_types+1)*5+1] = 20;
    // Gamma[2, 5] = Gamma[5, 2] = 20
    gammas[2*num_types+7] = 20;
    gammas[(num_types+1)*5+2] = 20;

    // Log the full surface-tension matrix so the physical meaning of the -G/-H/-I flags can be checked in the run log.
    cout << "Surface tension matrix (row/col = cell type: 0 background, 1-2 migrating types, 5 wall):" << endl;
    for (int ti = 0; ti <= num_types; ++ti)
        {
        for (int tj = 0; tj <= num_types; ++tj)
            cout << gammas[ti*(num_types+1) + tj] << (tj < num_types ? " " : "");
        cout << endl;
        }

    std::stringstream posixpath;
    posixpath << outRoot << "/NODAL_POS_GAMMA_LIVESPLIT/Data_t_" << tSteps << "_N_" << numpts << "_dt_" << dt << "_Ka_" << Ka << "_Kp_" << Kp << "_A0_" << a0a << "-" << a0b << "_P0_" << p0a << "-" << p0b << "_s0_" << s0a << "-" << s0b << "_c_Size_" << migratingcells << "_a_Cells_" << activecells << "_va_" << v0 << "_vp_" << u0 << "_vb_" << v_back << "_T1_" << gamma1 << "_T2_" << gamma2 << "_T3_" << gamma3 << "_Dr_b_" << Dr_back << "_Dr_tp_" << Dr_transplant << "_vD_tp_" << vD_transplant << "_IC_" << IC_switch << "_ensem_" << ensemble << ".nc";

    clock_t t1, t2; // clocks for timing information

    bool reproducible;
    if (Averaging_switch == 0)
    {
        reproducible = true;
    } // if you want random numbers with a more random seed each run, set this to false
    else
    {
        reproducible = false;
    }

    // check to see if we should run on a GPU
    bool initializeGPU = true;
    if (USE_GPU >= 0)
    {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu)
            return 0;
        cudaSetDevice(USE_GPU);
    }
    else
    {
        initializeGPU = false;
    }
    if (!ensureParentDirectoryExists(posixpath.str()))
    {
        std::cerr << "Failed to create output directory for " << posixpath.str()
                  << ": " << std::strerror(errno) << std::endl;
        return 1;
    }
    SPVDatabaseNetCDF ncdat(numpts, posixpath.str(), NcFile::Replace, false);

    // define an equation of motion object...here for self-propelled cells
    EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts, 1.0);
    // define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergyWithTension> spv = make_shared<VoronoiQuadraticEnergyWithTension>(numpts, 1.0, 4.0, reproducible);
    // UR: uses a random seed number for every simulation
    if (RNDseed == -1)
    {
        RNDseed = int(clock()) + ensemble;
    }
    //UR: Both E.O.M object and Voronoi object has two differnt seeds. Set them inividually
    spv->noise.setReproducibleSeed(RNDseed);
    spp->noise.setReproducibleSeed(RNDseed);

    // set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    // spv->setCellPreferencesUniform(1.0,p0);
    // set the cell activity to have D_r = 1. and a given v_0 and vD
    spv->setv0DrvD(0.1, Dr, vD); // PS fixes the v0 for the initial stead-state situation as the standard 0.1. You can reduce it later. Be mindful the time for SS will increase.
                                   // PS decreases v0 by ten times but did not increase the SS time yet!!
                                   // UR fixed vD = 0.0 for equilibration

    // PS sets the migration geomtry up
    ArrayHandle<int> h_ct(spv->cellType, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_p(spv->cellPositions, access_location::host, access_mode::read);
    ArrayHandle<Dscalar2> h_a(spv->AreaPeriPreferences, access_location::host, access_mode::overwrite);

    Dscalar boxsize = sqrt((Dscalar)numpts);
    int cellcount = 0;
    std::vector<int> mc_points;

    if (IC_switch == 0)
    {
        // Type 1
        for (int ii = 0; ii < numpts/2; ++ii)
        {
            h_ct.data[ii] = 1;
            h_a.data[ii].x = a0a;
            h_a.data[ii].y = p0a;
        }

        // Type 2
        for (int ii = numpts/2; ii < numpts; ++ii)
        {
            h_ct.data[ii] = 0;
            h_a.data[ii].x = a0b;
            h_a.data[ii].y = p0b;
        }
    }

    // circular disc contains all the smaller cells (random order - simulation)
    else if (IC_switch == 1)
    {
        Dscalar2 c_point;
        c_point.x = 4.0*boxsize/5.0, c_point.y = boxsize/2.0;
        // Find the closest points to the point above
        // By UR: Instead of drawing some arbitrary circle and finding points inside it, it is better
        // to find the closest points to a given points. That way, the cluster number is always equal to the
        // number of migrating cells and we will get a circular cluster.
        mc_points = kClosest(h_p, numpts, c_point, migratingcells);  
        for (int ii = 0; ii < numpts; ++ii)
        {  
            if (std::find(mc_points.begin(), mc_points.end(), ii) != mc_points.end())
            {
                cellcount = cellcount + 1;
                if (cellcount <= activecells) 
                {
                    h_ct.data[ii] = 1;
                } 
                else 
                {
                    h_ct.data[ii] = 2;
                } 
                h_a.data[ii].x = a0a;
                h_a.data[ii].y = p0a;
            }
            else // either the nuclei is outside the disc region OR we already have enough cluster cells
            {
                h_ct.data[ii] = 0;
                h_a.data[ii].x = a0b;
                h_a.data[ii].y = p0b;
            }
        }
    }
    // circular disc contains all the smaller cells (active cells to the front)
    else if (IC_switch == 2)
    {
        Dscalar2 c_point;
        c_point.x = 4.0*boxsize/5.0, c_point.y = boxsize/2.0;
        // Find the closest points to the point above
        // By UR: Instead of drawing some arbitrary circle and finding points inside it, it is better
        // to find the closest points to a given points. That way, the cluster number is always equal to the
        // number of migrating cells and we will get a circular cluster.

        // Use x-coordinates for sorting the list
        mc_points = kClosest_x(h_p, numpts, c_point, migratingcells);

        for (int ii = 0; ii < numpts; ++ii)
        {  
            // Initialize everything to be the same
            h_ct.data[ii] = 0;
            h_a.data[ii].x = a0b;
            h_a.data[ii].y = p0b;
        }

        // Iterate through the points in the chosen circle
        for (int ii : mc_points)
        {
            cellcount = cellcount + 1;
            if (cellcount <= activecells) 
            {
                h_ct.data[ii] = 1;
            } 
            else 
            {
                h_ct.data[ii] = 2;
            } 
            h_a.data[ii].x = a0a;
            h_a.data[ii].y = p0a;
        }
    }
    // circular disc contains all the smaller cells (active cells to the back)
    else if (IC_switch == 3)
    {
        Dscalar2 c_point;
        c_point.x = 4.0*boxsize/5.0, c_point.y = boxsize/2.0;
        // Find the closest points to the point above
        // By UR: Instead of drawing some arbitrary circle and finding points inside it, it is better
        // to find the closest points to a given points. That way, the cluster number is always equal to the
        // number of migrating cells and we will get a circular cluster.

        // Use x-coordinates for sorting the list
        mc_points = kClosest_x(h_p, numpts, c_point, migratingcells);

        for (int ii = 0; ii < numpts; ++ii)
        {  
            // Initialize everything to be the same
            h_ct.data[ii] = 0;
            h_a.data[ii].x = a0b;
            h_a.data[ii].y = p0b;
        }

        // Iterate through the points in the chosen circle
        for (int ii : mc_points)
        {
            cellcount = cellcount + 1;
            if (cellcount <= (migratingcells - activecells)) 
            {
                h_ct.data[ii] = 2;
            } 
            else 
            {
                h_ct.data[ii] = 1;
            } 
            h_a.data[ii].x = a0a;
            h_a.data[ii].y = p0a;
        }
    }
    else
    {
        for (int ii = 0; ii < numpts; ++ii)
        {
            if (h_p.data[ii].x < (boxsize/2))
            {
                h_ct.data[ii] = 0;
                h_a.data[ii].x = a0a;
                h_a.data[ii].y = p0a;
            }
            else
            {
                h_ct.data[ii] = 1;
                h_a.data[ii].x = a0b;
                h_a.data[ii].y = p0b;
            }
        }
    }

    // combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(spp, spv);
    // set the time step size
    sim->setIntegrationTimestep(dt);
    // initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    // sim->setSortPeriod(initSteps/10);
    // set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);
    spv->setModuliUniform(Ka, Kp);
    // UR: Use non-uniform gamma values
    spv->setSurfaceTension(gammas);
    spv->setUseSurfaceTension(true);
    // run for a few initialization timesteps
    printf("starting initialization with Random Seed: %i\n", RNDseed);
    for (int ii = 0; ii < initSteps; ++ii)
    {
        sim->performTimestep();
        spv->computeForces();
        // Write the initialization data to the output file
        if (ii % ((int)(0.2/ dt)) == 0)
        {
            ncdat.WriteState(spv);
        }
    };
    printf("Finished with initialization\n");
    
    // cout << "current q = " << spv->reportq() << endl;
    // the reporting of the force should yield a number that is numerically close to zero.
    // spv->reportMeanCellForce(false);
    if (!initializeGPU)
    {
        spv->setCPU(false); // turn off globabl-ony mode
    }

    // PS switches activity ON
    // Assuming that the tissue is now equilibrated to the tension and the shape differences, I now switch ON activity (and switch it OFF for not migrating cells)
    ArrayHandle<Dscalar> h_cd(spv->cellDirectors, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_v(spv->cellVelocities);
    ArrayHandle<Dscalar3> h_mot(spv->Motility, access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_m(spv->Moduli, access_location::host, access_mode::overwrite);

    // PS chooses what cells to PIN as wall
    for (int ii = 0; ii < numpts; ++ii)
    {
        // PIN the cells here
        // UR changed it from 2.5 to 2
        if (h_p.data[ii].x < 2.0)
        {
            // PS assumes you will not have more than 5 different celltypes in your simulations ever
            // fingers crossed! dont have to be dramatic you can always set the 5th type as 6 :P
            h_ct.data[ii] = 5;
            // UR: The wall cells are very "incompressible". Hence high KA.
            h_m.data[ii].x = 1000;
        }
    }

    // Set the activity for every cell type
    for (int ii = 0; ii < numpts; ++ii)
    {
        // transplant celltype is 1 or 2
        if (h_ct.data[ii] == 1)
        {
            // do we even have SPP dynamics activated?? or is it still brownian is which case this wont work // for now I am putting the value of pi
            h_cd.data[ii] = M_PI;
            // do a back of the envelope calc to make sure this value is small enough
            // Added by UR (0 -> Dr_back)
            // UR assumes that the Dr of all the transplant cells to be very small
            h_mot.data[ii].y = Dr_transplant;
            // UR -> cell directors have a small drift velocity now
            h_mot.data[ii].z = vD_transplant;

            // Added by UR
            // first half in the list are more active folks
            h_mot.data[ii].x = v0;
        }
        else if (h_ct.data[ii] == 2)
        {
            // do we even have SPP dynamics activated?? or is it still brownian is which case this wont work // for now I am putting the value of pi
            h_cd.data[ii] = M_PI;
            // do a back of the envelope calc to make sure this value is small enough
            // Added by UR (0 -> Dr_back)
            // UR assumes that the Dr of all the transplant cells to be very small
            h_mot.data[ii].y = Dr_transplant;
            // UR -> cell directors have a small drift velocity now
            h_mot.data[ii].z = vD_transplant;

            // Added by UR
            // second half is the less active one
            h_mot.data[ii].x = u0;
        }

        // Added by UR
        // background celltype is one
        else if (h_ct.data[ii] == 0)
        {
            h_cd.data[ii] = spv->noise.getRealUniform(-M_PI, M_PI);
            // Added by UR (0 -> Dr_back)
            // UR assumes that for v = 0.01, Dr = 0.05 for the background cells is enough for the
            // uncaging of the cells
            // Also, background cells only show Brownian motion -> vD_back = 0
            h_mot.data[ii].x = v_back;
            h_mot.data[ii].y = Dr_back;
            h_mot.data[ii].z = vD_back;
        }
        else 
        {
            h_cd.data[ii] = 0.0;
            // Added by UR (0 -> Dr_back)
            // For pinned cells, velocity is zero
            h_mot.data[ii].x = 0.0;
            h_mot.data[ii].y = 0.0;
            h_mot.data[ii].z = 0.0;
        }
    } 

    // ---- Added by UR (2026-10-14): live, in-simulation cluster-split
    // detection. Replaces the old post-hoc Python approach (which could only
    // snap the split moment to whatever saved frame happened to be nearest,
    // every 5 sim-time-units) with a per-INTEGRATION-STEP connectivity check
    // on the migrating cluster (mc_points), using the same
    // Simple2DCell::getCellNeighs() accessor already proven in
    // pos_gamma_adhesion_codes_t1_diagnostic_DELETE_ME.cpp's inline T1
    // tracking. Writes ONE extra small text file per run -- no extra .nc
    // frames saved.
    //
    // Debounce: a raw single-step disconnect is not immediately trusted (the
    // live triangulation can flicker near-degenerate configurations for one
    // step and immediately repair itself -- the same noise the T1 diagnostic
    // file's hysteresis state machine was built to reject). A candidate
    // split must remain disconnected continuously for SPLIT_HOLD_TIME sim-
    // time-units before being confirmed; if it reconnects first, it's
    // discarded as noise and monitoring resumes.
    //
    // Search window: matches the existing Python convention exactly (see
    // compute_cluster_split_time.py's `20.0 - X >= 15.0`, i.e. any migrating
    // cell's x-position dropping to <= 5.0) -- stop watching for a split
    // once any migrating cell gets this close to the wall/YSL region, same
    // as the post-hoc analysis's boundary-crossing cutoff.
    const Dscalar SPLIT_BOUNDARY_X = 5.0;
    const Dscalar SPLIT_HOLD_TIME = 0.05; // sim-time-units, same constant as the T1 diagnostic file

    std::set<int> mcSet(mc_points.begin(), mc_points.end());
    int n_mc = (int)mc_points.size();
    std::vector<int> mcCellType(n_mc);
    for (int k = 0; k < n_mc; ++k) mcCellType[k] = h_ct.data[mc_points[k]];

    auto buildMigEdges = [&]() {
        std::set<std::pair<int,int>> edges;
        vector<int> neighs; int nNeighs;
        for (int idx : mc_points)
        {
            spv->getCellNeighs(idx, nNeighs, neighs);
            for (int jn = 0; jn < nNeighs; ++jn)
            {
                int j = neighs[jn];
                if (mcSet.count(j))
                {
                    std::pair<int,int> e = (idx < j) ? std::make_pair(idx,j) : std::make_pair(j,idx);
                    edges.insert(e);
                }
            }
        }
        return edges;
    };
    // 2026-10-23, added by UR: label every migrating cell with its connected
    // component id for a given edge set, and return the number of distinct
    // components. Needed for the component-crossing fix below -- counting
    // "which edges disappeared" is not the same as "which edges actually
    // caused fragmentation": a sub-group that stays fully connected can
    // still lose some of its own internal edges to ordinary neighbor
    // reshuffling in the same step, and that must NOT be counted as part of
    // a split (confirmed via direct snapshot inspection on the Python side,
    // see SNAPSHOTS/INDUCED50_SHIELD_CHECK/ and CLAUDE.md 2026-10-23).
    auto getComponents = [&](const std::set<std::pair<int,int>> &edges, std::map<int,int> &compId) {
        std::map<int, std::vector<int>> adj;
        for (auto &e : edges) { adj[e.first].push_back(e.second); adj[e.second].push_back(e.first); }
        compId.clear();
        int nextId = 0;
        for (int idx : mc_points)
        {
            if (compId.count(idx)) continue;
            std::vector<int> stack = {idx};
            compId[idx] = nextId;
            while (!stack.empty())
            {
                int u = stack.back(); stack.pop_back();
                for (int v : adj[u]) if (!compId.count(v)) { compId[v] = nextId; stack.push_back(v); }
            }
            nextId++;
        }
        return nextId;
    };

    std::stringstream splitpath;
    splitpath << outRoot << "/NODAL_POS_GAMMA_LIVESPLIT/Data_t_" << tSteps << "_N_" << numpts << "_dt_" << dt << "_Ka_" << Ka << "_Kp_" << Kp << "_A0_" << a0a << "-" << a0b << "_P0_" << p0a << "-" << p0b << "_s0_" << s0a << "-" << s0b << "_c_Size_" << migratingcells << "_a_Cells_" << activecells << "_va_" << v0 << "_vp_" << u0 << "_vb_" << v_back << "_T1_" << gamma1 << "_T2_" << gamma2 << "_T3_" << gamma3 << "_Dr_b_" << Dr_back << "_Dr_tp_" << Dr_transplant << "_vD_tp_" << vD_transplant << "_IC_" << IC_switch << "_ensem_" << ensemble << "_split_time_live.txt";

    // 2026-10-23, rewritten by UR: track EVERY fragmentation event over the
    // run (not just the first -- "save data after first split too... this
    // way we store all splits data", explicit user request), and classify
    // each event's lost edges by final-component-crossing, not raw set
    // difference (same fix as the Python side). The running baseline is
    // "the component structure as of the last confirmed state" -- a new
    // candidate split is "component count increased beyond that baseline";
    // a reconnect back to <= that baseline's count is debounce noise, not
    // a reverse-split.
    // 2026-10-23, extended: homotypic is split by actual type (hom1 =
    // type1-type1, hom2 = type2-type2) instead of one collapsed "hom"
    // bucket -- answers "is the homotypic loss mostly within type1 or
    // type2" directly from the live per-step detector, at the same
    // resolution as the het/hom classification itself (raw .nc frames are
    // only saved every 5 sim-time-units, so a post-hoc script reading
    // them back loses exactly the fine-grained ordering this live check
    // exists to capture -- same reasoning as the original het/hom fix).
    struct SplitEvent { Dscalar time; long long het; long long hom1; long long hom2; };
    std::vector<SplitEvent> splitEvents;
    bool alreadyBrokenAtStart = false;
    bool windowEnded = false;
    bool havePendingCandidate = false;
    Dscalar pendingCandidateTime = 0.0;
    std::set<std::pair<int,int>> pendingEdgesBefore;
    std::set<std::pair<int,int>> migEdgesStable = buildMigEdges(); // last confirmed stable edge set
    int lastLocalTopoUpdates_split = spv->localTopologyUpdates;

    std::map<int,int> compId0;
    int lastComponentCount = getComponents(migEdgesStable, compId0);
    if (lastComponentCount > 1)
    {
        // Migrating cluster isn't even connected at t=0 -- not a real split
        // event (same exclusion rule as the Python post-hoc script).
        alreadyBrokenAtStart = true;
        windowEnded = true;
    }

    auto writeSplitFile = [&]() {
        std::ofstream sfile(splitpath.str());
        sfile << "n_events " << splitEvents.size() << "\n";
        sfile << "already_broken_at_start " << (alreadyBrokenAtStart ? 1 : 0) << "\n";
        sfile << "censored " << ((splitEvents.empty() && !alreadyBrokenAtStart) ? 1 : 0) << "\n";
        for (auto &ev : splitEvents)
            sfile << "event " << ev.time << " " << ev.het << " " << ev.hom1 << " " << ev.hom2 << "\n";
        sfile.close();
    };

    // Classify lost edges (before - after) using AFTER's final component
    // membership: a lost edge only counts as a real split-edge if its two
    // endpoints land in DIFFERENT components of `after`. An edge lost
    // between cells that remain in the same final component is incidental
    // internal reshuffling, not a real split, and must be excluded.
    auto classifyAndCountLostEdges = [&](const std::set<std::pair<int,int>> &before,
                                          const std::set<std::pair<int,int>> &after,
                                          long long &hetOut, long long &hom1Out, long long &hom2Out) {
        std::map<int,int> compIdAfter;
        getComponents(after, compIdAfter);
        hetOut = 0; hom1Out = 0; hom2Out = 0;
        for (auto &e : before)
        {
            if (after.count(e)) continue; // still present, not lost
            if (compIdAfter[e.first] == compIdAfter[e.second]) continue; // internal reshuffle, not a real split
            int ta = h_ct.data[e.first];
            int tb = h_ct.data[e.second];
            if (ta != tb) { hetOut++; continue; }
            if (ta == 1) hom1Out++; else hom2Out++;
        }
    };
    // ---- end live cluster-split setup ----

    // run for additional timesteps, compute dynamical features, and record timing information
    dynamicalFeatures dynFeat(spv->returnPositions(), spv->Box);
    // logSpacedIntegers logInts(0,0.05);
    std::cout << "Data_t_" << tSteps << "_N_" << numpts << "_dt_" << dt << "_Ka_" << Ka << "_Kp_" << Kp << "_A0_" << a0a << "-" << a0b << "_P0_" << p0a << "-" << p0b << "_s0_" << s0a << "-" << s0b << "_c_Size_" << migratingcells << "_a_Cells_" << activecells << "_va_" << v0 << "_vp_" << u0 << "_vb_" << v_back << "_T1_" << gamma1 << "_T2_" << gamma2 << "_T3_" << gamma3 << "_Dr_b_" << Dr_back << "_Dr_tp_" << Dr_transplant << "_vD_tp_" << vD_transplant << "_IC_" << IC_switch << "_ensem_" << ensemble << endl;
    t1 = clock();
    for (int ii = 0; ii < tSteps; ++ii)
    {
        if (ii % ((int)(saveInterval / dt)) == 0)
        {
            printf("timestep %i\t\t energy %f \t msd %f \t overlap %f topoUpdates %i \n", ii, spv->computeEnergy(), dynFeat.computeMSD(spv->returnPositions()), dynFeat.computeOverlapFunction(spv->returnPositions()), spv->localTopologyUpdates);
            // logInts.update();
            ncdat.WriteState(spv);
        };
        sim->performTimestep();
        spv->computeForces();

        // ---- Added by UR (2026-10-14, extended 2026-10-23 for multi-split) ----
        if (!windowEnded && n_mc > 1)
        {
            Dscalar tNow = (ii + 1) * dt;

            // boundary-window check first (cheap: just a position read)
            {
                ArrayHandle<Dscalar2> h_posChk(spv->cellPositions, access_location::host, access_mode::read);
                Dscalar minX = 1e30;
                for (int idx : mc_points) if (h_posChk.data[idx].x < minX) minX = h_posChk.data[idx].x;
                if (minX <= SPLIT_BOUNDARY_X)
                {
                    if (havePendingCandidate)
                    {
                        // candidate never had time to confirm before the window
                        // closed -- still more fragmented than baseline, so
                        // confirm it now.
                        std::set<std::pair<int,int>> edgesNow = buildMigEdges();
                        long long het, hom1, hom2;
                        classifyAndCountLostEdges(pendingEdgesBefore, edgesNow, het, hom1, hom2);
                        splitEvents.push_back({pendingCandidateTime, het, hom1, hom2});
                        havePendingCandidate = false;
                    }
                    windowEnded = true; // nothing more to check
                }
            }

            if (!windowEnded && havePendingCandidate)
            {
                // currently waiting out the debounce hold time -- check every
                // step (cheap, this state is rare/brief) whether it reconnects
                // back to (at most) the last confirmed baseline (discard as
                // noise) or survives the hold time (confirm as a real event).
                std::set<std::pair<int,int>> edgesNow = buildMigEdges();
                std::map<int,int> compIdNow;
                int nowCompCount = getComponents(edgesNow, compIdNow);
                if (nowCompCount <= lastComponentCount)
                {
                    // reconnected before confirmation -- was noise, resume
                    // normal gated monitoring from this restored state.
                    havePendingCandidate = false;
                    migEdgesStable = edgesNow;
                    lastComponentCount = nowCompCount;
                }
                else if (tNow - pendingCandidateTime >= SPLIT_HOLD_TIME)
                {
                    long long het, hom1, hom2;
                    classifyAndCountLostEdges(pendingEdgesBefore, edgesNow, het, hom1, hom2);
                    splitEvents.push_back({pendingCandidateTime, het, hom1, hom2});
                    migEdgesStable = edgesNow;
                    lastComponentCount = nowCompCount;
                    havePendingCandidate = false;
                    // keep monitoring: further fragmentation beyond this new
                    // baseline is still possible and must also be recorded.
                }
            }
            else if (!windowEnded && spv->localTopologyUpdates != lastLocalTopoUpdates_split)
            {
                // only pay for the full neighbor-list rebuild on steps where
                // the library itself confirms a real local repair happened
                // (same gating trick as the T1 diagnostic file -- avoids
                // floating-point chatter in near-degenerate triangulations).
                lastLocalTopoUpdates_split = spv->localTopologyUpdates;
                std::set<std::pair<int,int>> edgesNow = buildMigEdges();
                std::map<int,int> compIdNow;
                int nowCompCount = getComponents(edgesNow, compIdNow);
                if (nowCompCount > lastComponentCount)
                {
                    havePendingCandidate = true;
                    pendingCandidateTime = tNow;
                    pendingEdgesBefore = migEdgesStable;
                }
                else
                {
                    migEdgesStable = edgesNow;
                    lastComponentCount = nowCompCount;
                }
            }
        }
        // ---- end live cluster-split check ----
    };

    // If the run ended (tSteps exhausted) before a pending candidate resolved,
    // confirm whatever fragmentation state exists at the final step.
    if (!windowEnded && n_mc > 1 && !alreadyBrokenAtStart && havePendingCandidate)
    {
        std::set<std::pair<int,int>> edgesNow = buildMigEdges();
        long long het, hom1, hom2;
        classifyAndCountLostEdges(pendingEdgesBefore, edgesNow, het, hom1, hom2);
        splitEvents.push_back({pendingCandidateTime, het, hom1, hom2});
    }
    writeSplitFile();

    // Write last time step data to file
    printf("timestep %i\t\t energy %f \t msd %f \t overlap %f topoUpdates %i \n", tSteps, spv->computeEnergy(), dynFeat.computeMSD(spv->returnPositions()), dynFeat.computeOverlapFunction(spv->returnPositions()), spv->localTopologyUpdates);
    ncdat.WriteState(spv);

    t2 = clock();
    Dscalar steptime = (t2 - t1) / (Dscalar)CLOCKS_PER_SEC / tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    cout << "Total time taken: " << (t2 - t1) / (Dscalar)CLOCKS_PER_SEC << endl;
    cout << spv->reportq() << endl;
    cout << "number of local topology updates per cell per tau = " << spv->localTopologyUpdates * (1.0 / numpts) * (1.0 / tSteps / dt) << endl;

    if (initializeGPU)
    {
        cudaDeviceReset();
    }

    return 0;
};
