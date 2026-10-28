/*
 * nodal_pos_gamma_three_cell_livesplit.cpp
 * Same as nodal_pos_gamma_three_cell plus an in-simulation, per-integration-step cluster-split detector (writes <name>_split_time_live.txt next to the .nc).
 *
 * Two flags are common to every driver in this repository:
 *   -O <dir>   root output directory (default: DATA). Trajectories are written to
 *              <dir>/NODAL_POS_GAMMA_THREE_CELL_LIVESPLIT/Data_t_<...parameters...>_ensem_<k>.nc
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
    int type1cells = migratingcells/3; // number of type 1 cells in the migrating cluster
    int type2cells = migratingcells/3; // number of type 2 cells in the migrating cluster
    int type3cells = migratingcells/3; // number of type 3 cells in the migrating cluster
                                        
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
    Dscalar v0 = 2;      // the self-propulsion, of the first active celltype in the transplant
    Dscalar u0 = 0.5;    // activity of the second celltype in the transplant
    Dscalar w0 = 1.0;    // activity of the third celltype in the transplant
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
    Dscalar s0c = 3.7; // Third shape parameter
    Dscalar p0a = s0a;
    Dscalar p0b = s0b;
    Dscalar p0c = s0c;
    Dscalar a0a = 1.0;
    Dscalar a0b = 1.0;
    Dscalar a0c = 1.0;
    Dscalar gamma1 = 2;    // Tension between type 1 and background
    Dscalar gamma2 = 0.25; // Tension between type 2 and background
    Dscalar gamma3 = 1.0;  // Tension between type 3 and background
    Dscalar gamma4 = 0.5;  // Tension between type 1 and type 2
    Dscalar gamma5 = 0.75; // Tension between type 1 and type 3
    Dscalar gamma6 = 0.6;  // Tension between type 2 and type 3
    Dscalar Ka = 10.0;
    Dscalar Kp = 1.0;
    int num_types = 5;     // background(0), type1(1), type2(2), type3(3), wall(5)
    int RNDseed = -1; //UR: If RNDSeed = -1, then the current time plus the ensemble number is taken as the seed
                      // Or a seed can be passed to the code.

    // The defaults can be overridden from the command line
    std::string outRoot = "DATA";   // -O
    Dscalar saveInterval = 5.0;      // -S
    while ((c = getopt(argc, argv, "a:n:b:t:G:H:I:J:K:L:i:e:s:q:r:u:v:w:k:p:f:x:m:d:j:y:c:l:o:z:S:O:")) != -1)
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
        case 'J':
            gamma4 = atof(optarg);
            break;
        case 'K':
            gamma5 = atof(optarg);
            break;
        case 'L':
            gamma6 = atof(optarg);
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
        case 'r':
            s0c = atof(optarg);
            break;
        case 'u':
            u0 = atof(optarg);
            break;
        case 'v':
            v0 = atof(optarg);
            break;
        case 'w':
            w0 = atof(optarg);
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
            type1cells = atoi(optarg);
            break; // Type1 cells count
        case 'o':
            type2cells = atoi(optarg);
            break; // Type2 cells count
        case 'z':
            type3cells = atoi(optarg);
            break; // Type3 cells count        
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

    // Check if cell counts add up to migrating cells
    if (type1cells + type2cells + type3cells != migratingcells) {
        std::cerr << "Error: type1cells + type2cells + type3cells must equal migratingcells" << std::endl;
        std::cerr << "Current values: type1cells=" << type1cells 
                  << ", type2cells=" << type2cells 
                  << ", type3cells=" << type3cells 
                  << ", migratingcells=" << migratingcells << std::endl;
        return 1;
    }

    a0a = 2 / (1 + alpha);
    p0a = s0a * sqrt(a0a);
    a0b = 2 * alpha / (1 + alpha);
    p0b = s0b * sqrt(a0b);
    a0c = 2 * alpha / (1 + alpha);
    p0c = s0c * sqrt(a0c);

    // UR: Initialize a vector of n^2 for gamma values between different cell types
    // Checkout: https://dmsussman.gitlab.io/cellGPUdocumentation/classVoronoiQuadraticEnergyWithTension.html
    vector<Dscalar> gammas((num_types+1)*(num_types+1), 0.0);
    
    // Line Tension is between Cell Types 0 <-> 1, 0 <-> 2, 0 <-> 3, 1 <-> 2, 1 <-> 3, 2 <-> 3, 1/2/3 <-> 5
    
    // Gamma[0, 1] = Gamma[1, 0] = gamma1 (background-type1)
    gammas[0*(num_types+1) + 1] = gamma1;
    gammas[1*(num_types+1) + 0] = gamma1;
    
    // Gamma[0, 2] = Gamma[2, 0] = gamma2 (background-type2)
    gammas[0*(num_types+1) + 2] = gamma2;
    gammas[2*(num_types+1) + 0] = gamma2;
    
    // Gamma[0, 3] = Gamma[3, 0] = gamma3 (background-type3)
    gammas[0*(num_types+1) + 3] = gamma3;
    gammas[3*(num_types+1) + 0] = gamma3;
    
    // Gamma[1, 2] = Gamma[2, 1] = gamma4 (type1-type2)
    gammas[1*(num_types+1) + 2] = gamma4;
    gammas[2*(num_types+1) + 1] = gamma4;
    
    // Gamma[1, 3] = Gamma[3, 1] = gamma5 (type1-type3)
    gammas[1*(num_types+1) + 3] = gamma5;
    gammas[3*(num_types+1) + 1] = gamma5;
    
    // Gamma[2, 3] = Gamma[3, 2] = gamma6 (type2-type3)
    gammas[2*(num_types+1) + 3] = gamma6;
    gammas[3*(num_types+1) + 2] = gamma6;
    
    // Gamma[1/2/3, 5] = Gamma[5, 1/2/3] = 20 (wall interaction)
    gammas[1*(num_types+1) + 5] = 20;
    gammas[5*(num_types+1) + 1] = 20;
    
    gammas[2*(num_types+1) + 5] = 20;
    gammas[5*(num_types+1) + 2] = 20;
    
    gammas[3*(num_types+1) + 5] = 20;
    gammas[5*(num_types+1) + 3] = 20;

    // 2026-10-23, added by UR: live per-integration-step cluster-split
    // detector for the 3-cell-type (Triple) condition, adapted from
    // Nodal_pos_gamma_adhesion_codes_livesplit.cpp's 2-type version
    // (component-crossing classification + tracks ALL splits, not just
    // the first). Output redirected to a separate directory -- this
    // binary re-simulates from scratch with a fresh seed each run, so
    // reusing the production DATA/NODAL_POS_GAMMA_THREE/ path would
    // silently overwrite already-analyzed data.
    // Log the full surface-tension matrix so the physical meaning of the -G..-L flags can be checked in the run log.
    cout << "Surface tension matrix (row/col = cell type: 0 background, 1-3 migrating types, 5 wall):" << endl;
    for (int ti = 0; ti <= num_types; ++ti)
        {
        for (int tj = 0; tj <= num_types; ++tj)
            cout << gammas[ti*(num_types+1) + tj] << (tj < num_types ? " " : "");
        cout << endl;
        }

    std::stringstream posixpath;
    posixpath << outRoot << "/NODAL_POS_GAMMA_THREE_CELL_LIVESPLIT/Data_t_" << tSteps
              << "_N_" << numpts 
              << "_dt_" << dt 
              << "_Ka_" << Ka 
              << "_Kp_" << Kp 
              << "_A0_" << a0a << "-" << a0b << "-" << a0c
              << "_P0_" << p0a << "-" << p0b << "-" << p0c
              << "_s0_" << s0a << "-" << s0b << "-" << s0c
              << "_c_Size_" << migratingcells 
              << "_t1_" << type1cells 
              << "_t2_" << type2cells 
              << "_t3_" << type3cells 
              << "_v1_" << v0 
              << "_v2_" << u0 
              << "_v3_" << w0
              << "_vb_" << v_back 
              << "_G1_" << gamma1 
              << "_G2_" << gamma2 
              << "_G3_" << gamma3 
              << "_G4_" << gamma4 
              << "_G5_" << gamma5 
              << "_G6_" << gamma6
              << "_Dr_b_" << Dr_back 
              << "_Dr_tp_" << Dr_transplant 
              << "_vD_tp_" << vD_transplant 
              << "_IC_" << IC_switch 
              << "_ensem_" << ensemble << ".nc";

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
        for (int ii = 0; ii < numpts/3; ++ii)
        {
            h_ct.data[ii] = 1;
            h_a.data[ii].x = a0a;
            h_a.data[ii].y = p0a;
        }

        // Type 2
        for (int ii = numpts/3; ii < 2*numpts/3; ++ii)
        {
            h_ct.data[ii] = 2;
            h_a.data[ii].x = a0b;
            h_a.data[ii].y = p0b;
        }
        
        // Type 3
        for (int ii = 2*numpts/3; ii < numpts; ++ii)
        {
            h_ct.data[ii] = 3;
            h_a.data[ii].x = a0c;
            h_a.data[ii].y = p0c;
        }
    }

    // circular disc contains all the cell types (random order)
    else if (IC_switch == 1)
    {
        Dscalar2 c_point;
        c_point.x = 4.0*boxsize/5.0, c_point.y = boxsize/2.0;
        // Find the closest points to the point above
        mc_points = kClosest(h_p, numpts, c_point, migratingcells);  
        
        // Initialize all cells as background
        for (int ii = 0; ii < numpts; ++ii)
        {
            h_ct.data[ii] = 0;
            h_a.data[ii].x = a0b;
            h_a.data[ii].y = p0b;
        }
        
        // Assign migrating cell types
        cellcount = 0;
        for (int ii : mc_points)
        {
            if (cellcount < type1cells) {
                h_ct.data[ii] = 1;
                h_a.data[ii].x = a0a;
                h_a.data[ii].y = p0a;
            } 
            else if (cellcount < type1cells + type2cells) {
                h_ct.data[ii] = 2;
                h_a.data[ii].x = a0b;
                h_a.data[ii].y = p0b;
            }
            else {
                h_ct.data[ii] = 3;
                h_a.data[ii].x = a0c;
                h_a.data[ii].y = p0c;
            }
            cellcount++;
        }
    }
    
    // circular disc with sorted cell types (ordered by x-coordinate)
    else if (IC_switch == 2)
    {
        Dscalar2 c_point;
        c_point.x = 4.0*boxsize/5.0, c_point.y = boxsize/2.0;
        // Find the closest points to the point above and sort by x-coordinate
        mc_points = kClosest_x(h_p, numpts, c_point, migratingcells);

        // Initialize all cells as background
        for (int ii = 0; ii < numpts; ++ii)
        {
            h_ct.data[ii] = 0;
            h_a.data[ii].x = a0b;
            h_a.data[ii].y = p0b;
        }
        
        // Assign migrating cell types - ordered from left to right
        cellcount = 0;
        for (int ii : mc_points)
        {
            if (cellcount < type1cells) {
                h_ct.data[ii] = 1;
                h_a.data[ii].x = a0a;
                h_a.data[ii].y = p0a;
            } 
            else if (cellcount < type1cells + type2cells) {
                h_ct.data[ii] = 2;
                h_a.data[ii].x = a0b;
                h_a.data[ii].y = p0b;
            }
            else {
                h_ct.data[ii] = 3;
                h_a.data[ii].x = a0c;
                h_a.data[ii].y = p0c;
            }
            cellcount++;
        }
    }
    
    // circular disc with sorted cell types (ordered from right to left)
    else if (IC_switch == 3)
    {
        Dscalar2 c_point;
        c_point.x = 4.0*boxsize/5.0, c_point.y = boxsize/2.0;
        // Find the closest points to the point above and sort by x-coordinate
        mc_points = kClosest_x(h_p, numpts, c_point, migratingcells);
        std::reverse(mc_points.begin(), mc_points.end()); // Reverse the order

        // Initialize all cells as background
        for (int ii = 0; ii < numpts; ++ii)
        {
            h_ct.data[ii] = 0;
            h_a.data[ii].x = a0b;
            h_a.data[ii].y = p0b;
        }
        
        // Assign migrating cell types - ordered from right to left
        cellcount = 0;
        for (int ii : mc_points)
        {
            if (cellcount < type1cells) {
                h_ct.data[ii] = 1;
                h_a.data[ii].x = a0a;
                h_a.data[ii].y = p0a;
            } 
            else if (cellcount < type1cells + type2cells) {
                h_ct.data[ii] = 2;
                h_a.data[ii].x = a0b;
                h_a.data[ii].y = p0b;
            }
            else {
                h_ct.data[ii] = 3;
                h_a.data[ii].x = a0c;
                h_a.data[ii].y = p0c;
            }
            cellcount++;
        }
    }
    // Layered structure (half/half)
    else if (IC_switch == 4)
    {
        for (int ii = 0; ii < numpts; ++ii)
        {
            if (h_p.data[ii].x < (boxsize/2))
            {
                h_ct.data[ii] = 0;
                h_a.data[ii].x = a0b;
                h_a.data[ii].y = p0b;
            }
            else
            {
                // Assign the three types in equal proportions
                int third = (ii % 3);
                if (third == 0) {
                    h_ct.data[ii] = 1;
                    h_a.data[ii].x = a0a;
                    h_a.data[ii].y = p0a;
                }
                else if (third == 1) {
                    h_ct.data[ii] = 2;
                    h_a.data[ii].x = a0b;
                    h_a.data[ii].y = p0b;
                }
                else {
                    h_ct.data[ii] = 3;
                    h_a.data[ii].x = a0c;
                    h_a.data[ii].y = p0c;
                }
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
        // Cell type 1 (most active)
        if (h_ct.data[ii] == 1)
        {
            h_cd.data[ii] = M_PI;
            h_mot.data[ii].y = Dr_transplant;
            h_mot.data[ii].z = vD_transplant;
            h_mot.data[ii].x = v0;
        }
        // Cell type 2 (medium activity)
        else if (h_ct.data[ii] == 2)
        {
            h_cd.data[ii] = M_PI;
            h_mot.data[ii].y = Dr_transplant;
            h_mot.data[ii].z = vD_transplant;
            h_mot.data[ii].x = u0;
        }
        // Cell type 3 (low activity)
        else if (h_ct.data[ii] == 3)
        {
            h_cd.data[ii] = M_PI;
            h_mot.data[ii].y = Dr_transplant;
            h_mot.data[ii].z = vD_transplant;
            h_mot.data[ii].x = w0;
        }
        // Background celltype (0)
        else if (h_ct.data[ii] == 0)
        {
            h_cd.data[ii] = spv->noise.getRealUniform(-M_PI, M_PI);
            h_mot.data[ii].x = v_back;
            h_mot.data[ii].y = Dr_back;
            h_mot.data[ii].z = vD_back;
        }
        // Wall cells (type 5)
        else 
        {
            h_cd.data[ii] = 0.0;
            h_mot.data[ii].x = 0.0;
            h_mot.data[ii].y = 0.0;
            h_mot.data[ii].z = 0.0;
        }
    } 

    // ---- Added by UR (2026-10-23): live, in-simulation cluster-split
    // detection for 3 cell types. Same per-integration-step connectivity
    // check, debounce-against-flicker state machine, and "track every
    // split, not just the first" architecture as the 2-type livesplit
    // binary -- only the classification step differs (6 categories:
    // 1-1/2-2/3-3 homotypic, 1-2/1-3/2-3 heterotypic, instead of 2).
    const Dscalar SPLIT_BOUNDARY_X = 5.0;
    const Dscalar SPLIT_HOLD_TIME = 0.05;

    std::set<int> mcSet(mc_points.begin(), mc_points.end());
    int n_mc = (int)mc_points.size();

    auto buildMigEdges3 = [&]() {
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
    auto getComponents3 = [&](const std::set<std::pair<int,int>> &edges, std::map<int,int> &compId) {
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

    std::stringstream splitpath3;
    splitpath3 << outRoot << "/NODAL_POS_GAMMA_THREE_CELL_LIVESPLIT/Data_t_" << tSteps << "_N_" << numpts << "_dt_" << dt << "_Ka_" << Ka << "_Kp_" << Kp
               << "_A0_" << a0a << "-" << a0b << "-" << a0c << "_P0_" << p0a << "-" << p0b << "-" << p0c
               << "_s0_" << s0a << "-" << s0b << "-" << s0c << "_c_Size_" << migratingcells
               << "_t1_" << type1cells << "_t2_" << type2cells << "_t3_" << type3cells
               << "_v1_" << v0 << "_v2_" << u0 << "_v3_" << w0 << "_vb_" << v_back
               << "_G1_" << gamma1 << "_G2_" << gamma2 << "_G3_" << gamma3
               << "_G4_" << gamma4 << "_G5_" << gamma5 << "_G6_" << gamma6
               << "_Dr_b_" << Dr_back << "_Dr_tp_" << Dr_transplant << "_vD_tp_" << vD_transplant
               << "_IC_" << IC_switch << "_ensem_" << ensemble << "_split_time_live.txt";

    struct SplitEvent3 { Dscalar time; long long c11, c22, c33, c12, c13, c23; };
    std::vector<SplitEvent3> splitEvents3;
    bool alreadyBrokenAtStart3 = false;
    bool windowEnded3 = false;
    bool havePendingCandidate3 = false;
    Dscalar pendingCandidateTime3 = 0.0;
    std::set<std::pair<int,int>> pendingEdgesBefore3;
    std::set<std::pair<int,int>> migEdgesStable3 = buildMigEdges3();
    int lastLocalTopoUpdates3 = spv->localTopologyUpdates;

    std::map<int,int> compId03;
    int lastComponentCount3 = getComponents3(migEdgesStable3, compId03);
    if (lastComponentCount3 > 1)
    {
        alreadyBrokenAtStart3 = true;
        windowEnded3 = true;
    }

    auto writeSplitFile3 = [&]() {
        std::ofstream sfile(splitpath3.str());
        sfile << "n_events " << splitEvents3.size() << "\n";
        sfile << "already_broken_at_start " << (alreadyBrokenAtStart3 ? 1 : 0) << "\n";
        sfile << "censored " << ((splitEvents3.empty() && !alreadyBrokenAtStart3) ? 1 : 0) << "\n";
        for (auto &ev : splitEvents3)
            sfile << "event " << ev.time << " " << ev.c11 << " " << ev.c22 << " " << ev.c33
                  << " " << ev.c12 << " " << ev.c13 << " " << ev.c23 << "\n";
        sfile.close();
    };

    auto classifyAndCountLostEdges3 = [&](const std::set<std::pair<int,int>> &before,
                                           const std::set<std::pair<int,int>> &after,
                                           SplitEvent3 &out) {
        std::map<int,int> compIdAfter;
        getComponents3(after, compIdAfter);
        out.c11 = out.c22 = out.c33 = out.c12 = out.c13 = out.c23 = 0;
        for (auto &e : before)
        {
            if (after.count(e)) continue;
            if (compIdAfter[e.first] == compIdAfter[e.second]) continue;
            int ta = h_ct.data[e.first], tb = h_ct.data[e.second];
            int lo = std::min(ta, tb), hi = std::max(ta, tb);
            if (lo == 1 && hi == 1) out.c11++;
            else if (lo == 2 && hi == 2) out.c22++;
            else if (lo == 3 && hi == 3) out.c33++;
            else if (lo == 1 && hi == 2) out.c12++;
            else if (lo == 1 && hi == 3) out.c13++;
            else if (lo == 2 && hi == 3) out.c23++;
        }
    };
    // ---- end live cluster-split setup ----

    // run for additional timesteps, compute dynamical features, and record timing information
    dynamicalFeatures dynFeat(spv->returnPositions(), spv->Box);
    // logSpacedIntegers logInts(0,0.05);
    std::cout << "Data_t_" << tSteps << "_N_" << numpts << "_dt_" << dt << "_Ka_" << Ka << "_Kp_" << Kp
              << "_A0_" << a0a << "-" << a0b << "-" << a0c
              << "_P0_" << p0a << "-" << p0b << "-" << p0c
              << "_s0_" << s0a << "-" << s0b << "-" << s0c
              << "_c_Size_" << migratingcells
              << "_t1_" << type1cells
              << "_t2_" << type2cells
              << "_t3_" << type3cells
              << "_v1_" << v0
              << "_v2_" << u0
              << "_v3_" << w0
              << "_vb_" << v_back
              << "_G1_" << gamma1
              << "_G2_" << gamma2
              << "_G3_" << gamma3
              << "_G4_" << gamma4
              << "_G5_" << gamma5
              << "_G6_" << gamma6
              << "_Dr_b_" << Dr_back
              << "_Dr_tp_" << Dr_transplant
              << "_vD_tp_" << vD_transplant
              << "_IC_" << IC_switch
              << "_ensem_" << ensemble << endl;
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

        // ---- live cluster-split check, every step ----
        if (!windowEnded3 && n_mc > 1)
        {
            Dscalar tNow = (ii + 1) * dt;
            {
                ArrayHandle<Dscalar2> h_posChk(spv->cellPositions, access_location::host, access_mode::read);
                Dscalar minX = 1e30;
                for (int idx : mc_points) if (h_posChk.data[idx].x < minX) minX = h_posChk.data[idx].x;
                if (minX <= SPLIT_BOUNDARY_X)
                {
                    if (havePendingCandidate3)
                    {
                        std::set<std::pair<int,int>> edgesNow = buildMigEdges3();
                        SplitEvent3 ev; ev.time = pendingCandidateTime3;
                        classifyAndCountLostEdges3(pendingEdgesBefore3, edgesNow, ev);
                        splitEvents3.push_back(ev);
                        havePendingCandidate3 = false;
                    }
                    windowEnded3 = true;
                }
            }

            if (!windowEnded3 && havePendingCandidate3)
            {
                std::set<std::pair<int,int>> edgesNow = buildMigEdges3();
                std::map<int,int> compIdNow;
                int nowCompCount = getComponents3(edgesNow, compIdNow);
                if (nowCompCount <= lastComponentCount3)
                {
                    havePendingCandidate3 = false;
                    migEdgesStable3 = edgesNow;
                    lastComponentCount3 = nowCompCount;
                }
                else if (tNow - pendingCandidateTime3 >= SPLIT_HOLD_TIME)
                {
                    SplitEvent3 ev; ev.time = pendingCandidateTime3;
                    classifyAndCountLostEdges3(pendingEdgesBefore3, edgesNow, ev);
                    splitEvents3.push_back(ev);
                    migEdgesStable3 = edgesNow;
                    lastComponentCount3 = nowCompCount;
                    havePendingCandidate3 = false;
                }
            }
            else if (!windowEnded3 && spv->localTopologyUpdates != lastLocalTopoUpdates3)
            {
                lastLocalTopoUpdates3 = spv->localTopologyUpdates;
                std::set<std::pair<int,int>> edgesNow = buildMigEdges3();
                std::map<int,int> compIdNow;
                int nowCompCount = getComponents3(edgesNow, compIdNow);
                if (nowCompCount > lastComponentCount3)
                {
                    havePendingCandidate3 = true;
                    pendingCandidateTime3 = tNow;
                    pendingEdgesBefore3 = migEdgesStable3;
                }
                else
                {
                    migEdgesStable3 = edgesNow;
                    lastComponentCount3 = nowCompCount;
                }
            }
        }
        // ---- end live cluster-split check ----
    };

    if (!windowEnded3 && n_mc > 1 && !alreadyBrokenAtStart3 && havePendingCandidate3)
    {
        std::set<std::pair<int,int>> edgesNow = buildMigEdges3();
        SplitEvent3 ev; ev.time = pendingCandidateTime3;
        classifyAndCountLostEdges3(pendingEdgesBefore3, edgesNow, ev);
        splitEvents3.push_back(ev);
    }
    writeSplitFile3();

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