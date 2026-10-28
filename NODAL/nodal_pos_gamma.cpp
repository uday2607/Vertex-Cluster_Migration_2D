/*
 * nodal_pos_gamma.cpp
 * SPV model with Nodal-signalling levels, POSITIVE-gamma rule, ONE or TWO cell types in the cluster (leaders type 1, followers type 2). Three types: see nodal_pos_gamma_three_cell.
 *
 * Two flags are common to every driver in this repository:
 *   -O <dir>   root output directory (default: DATA). Trajectories are written to
 *              <dir>/NODAL_POS_GAMMA/Data_t_<...parameters...>_ensem_<k>.nc
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
    posixpath << outRoot << "/NODAL_POS_GAMMA/Data_t_" << tSteps << "_N_" << numpts << "_dt_" << dt << "_Ka_" << Ka << "_Kp_" << Kp << "_A0_" << a0a << "-" << a0b << "_P0_" << p0a << "-" << p0b << "_s0_" << s0a << "-" << s0b << "_c_Size_" << migratingcells << "_a_Cells_" << activecells << "_va_" << v0 << "_vp_" << u0 << "_vb_" << v_back << "_T1_" << gamma1 << "_T2_" << gamma2 << "_T3_" << gamma3 << "_Dr_b_" << Dr_back << "_Dr_tp_" << Dr_transplant << "_vD_tp_" << vD_transplant << "_IC_" << IC_switch << "_ensem_" << ensemble << ".nc";

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
    };

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
