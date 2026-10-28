/*
 * avm_viscous_dissipation_type.cpp
 * AVM (wet vertex model) with cell-type-dependent junction dissipation (heterotypic vs homotypic), gamma = 0.
 *
 * Two flags are common to every driver in this repository:
 *   -O <dir>   root output directory (default: DATA). Trajectories are written to
 *              <dir>/AVM_VISCOUS_DISSIPATION_TYPE/Data_t_<...parameters...>_ensem_<k>.nc
 *   -S <time>  time between saved frames during production, in simulation time
 *              units (default 5.0). Smaller values give smoother movies and bigger files.
 * All other flags are listed in the README.md of this folder.
 */
#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "vertexQuadraticEnergyWithTension.h"
#include "selfPropelledCellVertexDynamicsWithTypeDissipation.h"
#include "analysisPackage.h"
#include "DatabaseNetCDFAVM.h"

/*
 * Vertex model with TYPE-DEPENDENT viscous dissipation, and no line tension
 * (run with -G 0 -H 0 -I 0 to zero out gamma_ij entirely).
 *
 * Tests whether purely kinetic (dissipative) heterogeneity between cell
 * types -- with zero energetic (tension) heterogeneity -- is enough on its
 * own to reproduce cluster cohesion/invasion behavior. Same bond-projected
 * form as the xi0 model (Nguyen et al.), but the coefficient now depends on
 * which two cell types border each edge:
 *
 *   chi_ij = chiSame (-X)  if the two cells bordering edge (i,j) are same type
 *          = chiDiff (-Y)  otherwise
 *
 * All other parameters and initial-condition switches are identical to
 * avm_viscous_adhesion.cpp / avm_pos_gamma.cpp.
 */

// ---- helper: k closest vertices (by Euclidean distance) ----------------

std::vector<int> kClosest(ArrayHandle<Dscalar2> points, int num_points,
                          Dscalar2 c_point, int k)
{
    std::multimap<double, int> mp;
    for (int i = 0; i < num_points; i++)
    {
        double dx = points.data[i].x - c_point.x;
        double dy = points.data[i].y - c_point.y;
        mp.insert({sqrt(dx*dx + dy*dy), i});
    }
    std::vector<int> out;
    for (auto it = mp.begin(); it != mp.end() && k > 0; ++it, --k)
        out.push_back(it->second);
    return out;
}

// ---- helper: k closest, sorted by x-coordinate (active cells to front) -

std::vector<int> kClosest_x(ArrayHandle<Dscalar2> points, int num_points,
                             Dscalar2 c_point, int k)
{
    std::multimap<double, int> mp;
    for (int i = 0; i < num_points; i++)
    {
        double dx = points.data[i].x - c_point.x;
        double dy = points.data[i].y - c_point.y;
        mp.insert({sqrt(dx*dx + dy*dy), i});
    }
    std::multimap<double, int> mp2;
    int k_ = k;
    for (auto it = mp.begin(); it != mp.end() && k_ > 0; ++it, --k_)
        mp2.insert({points.data[it->second].x, it->second});

    std::vector<int> out;
    for (auto it = mp2.begin(); it != mp2.end() && k > 0; ++it, --k)
        out.push_back(it->second);
    return out;
}

// ========================================================================

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
    // ---- defaults -------------------------------------------------------
    int numpts        = 400;
    int migratingcells = 8;
    int activecells   = migratingcells / 2;
    int USE_GPU       = -1;
    int tSteps        = 500000;
    int ensemble      = 999;
    int IC_switch     = 1;
    int Averaging_switch = 1;
    int IV_switch     = 1;

    Dscalar dt            = 0.0002;
    Dscalar v0            = 2.0;
    Dscalar u0            = 0.5;
    Dscalar Dr            = 1.0;
    Dscalar vD            = 0.0;
    Dscalar Dr_back       = 0.05;
    Dscalar v_back        = 0.01;
    Dscalar vD_back       = 0.0;
    Dscalar Dr_transplant = 0.005;
    Dscalar vD_transplant = 0.02;

    // Type-dependent bond-projected dissipation (no tension in this driver)
    Dscalar chiSame = 0.0;
    Dscalar chiDiff = 0.0;
    Dscalar mu      = 1.0;

    Dscalar alpha = 1.0;
    Dscalar s0a   = 3.7;
    Dscalar s0b   = 3.7;
    Dscalar a0a, p0a, a0b, p0b;
    Dscalar Ka    = 10.0;
    Dscalar Kp    = 1.0;
    Dscalar gamma1 = 0.0;
    Dscalar gamma2 = 0.0;
    Dscalar gamma3 = 0.0;

    int num_types = 5;
    int RNDseed   = -1;

    int initSteps  = (int)(1.0 / (dt * 1.0)) + 1;
    int granularity = (int)(100.0 / (dt * 1.0));

    // ---- command-line overrides -----------------------------------------
    int c;
    std::string outRoot = "DATA";   // -O
    Dscalar saveInterval = 5.0;      // -S
    while ((c = getopt(argc, argv, "a:n:b:t:G:H:I:i:e:s:q:r:u:v:k:p:f:x:m:d:j:y:c:l:X:Y:M:S:O:")) != -1)
        switch (c)
        {
        case 'n': numpts = atoi(optarg); break;
        case 'm': migratingcells = atoi(optarg); break;
        case 'i': IC_switch = atoi(optarg); break;
        case 't': tSteps = atoi(optarg); break;
        case 'G': gamma1 = atof(optarg); break;
        case 'H': gamma2 = atof(optarg); break;
        case 'I': gamma3 = atof(optarg); break;
        case 'e': dt = atof(optarg); break;
        case 's': s0a = atof(optarg); break;
        case 'q': s0b = atof(optarg); break;
        case 'u': u0 = atof(optarg); break;
        case 'v': v0 = atof(optarg); break;
        case 'k': Ka = atof(optarg); break;
        case 'p': Kp = atof(optarg); break;
        case 'f': granularity = (int)atof(optarg); break;
        case 'x': ensemble = atoi(optarg); break;
        case 'd': Dr_back = atof(optarg); break;
        case 'y': v_back = atof(optarg); break;
        case 'b': Dr_transplant = atof(optarg); break;
        case 'j': vD_transplant = atof(optarg); break;
        case 'c': RNDseed = atoi(optarg); break;
        case 'l': activecells = atoi(optarg); break;
        case 'X': chiSame = atof(optarg); break;
        case 'Y': chiDiff = atof(optarg); break;
        case 'M': mu      = atof(optarg); break;
        case 'S':
            saveInterval = atof(optarg);
            break;
        case 'O':
            outRoot = optarg;
            break;
        case '?':
            if (isprint(optopt))
                std::cerr << "Unknown option '-" << optopt << "'.\n";
            return 1;
        default: abort();
        }

    a0a = 2.0 / (1.0 + alpha);
    p0a = s0a * sqrt(a0a);
    a0b = 2.0 * alpha / (1.0 + alpha);
    p0b = s0b * sqrt(a0b);

    // ---- line-tension matrix (zero by default in this driver) -----------
    vector<Dscalar> gammas((num_types+1)*(num_types+1), 0.0);
    gammas[1]                   = gamma1;   // Gamma[0,1]
    gammas[num_types+1]         = gamma1;   // Gamma[1,0]
    gammas[2]                   = gamma2;   // Gamma[0,2]
    gammas[2*(num_types+1)]     = gamma2;   // Gamma[2,0]
    gammas[num_types+3]         = gamma3;   // Gamma[1,2]
    gammas[2*num_types+3]       = gamma3;   // Gamma[2,1]

    // ---- output path ----------------------------------------------------
    std::stringstream posixpath;
    posixpath << outRoot << "/AVM_VISCOUS_DISSIPATION_TYPE/Data"
              << "_t_"     << tSteps
              << "_N_"     << numpts
              << "_dt_"    << dt
              << "_Ka_"    << Ka
              << "_Kp_"    << Kp
              << "_A0_"    << a0a << "-" << a0b
              << "_P0_"    << p0a << "-" << p0b
              << "_s0_"    << s0a << "-" << s0b
              << "_cSz_"   << migratingcells
              << "_aC_"    << activecells
              << "_va_"    << v0
              << "_vp_"    << u0
              << "_vb_"    << v_back
              << "_T1_"    << gamma1
              << "_T2_"    << gamma2
              << "_T3_"    << gamma3
              << "_chiSame_" << chiSame
              << "_chiDiff_" << chiDiff
              << "_mu_"    << mu
              << "_Dr_b_"  << Dr_back
              << "_Dr_tp_" << Dr_transplant
              << "_vD_tp_" << vD_transplant
              << "_IC_"    << IC_switch
              << "_ens_"   << ensemble
              << ".nc";

    // ---- GPU / CPU ------------------------------------------------------
    bool initializeGPU = false;
    if (USE_GPU >= 0)
    {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        initializeGPU = true;
    }

    bool reproducible = (Averaging_switch == 0);

    int Nvert = 2 * numpts;
    if (!ensureParentDirectoryExists(posixpath.str()))
    {
        std::cerr << "Failed to create output directory for " << posixpath.str()
                  << ": " << std::strerror(errno) << std::endl;
        return 1;
    }
    AVMDatabaseNetCDF ncdat(Nvert, posixpath.str(), NcFile::Replace);

    // ---- build model and EOM updater ------------------------------------
    bool runSPV = false;

    shared_ptr<selfPropelledCellVertexDynamicsWithTypeDissipation> spp =
        make_shared<selfPropelledCellVertexDynamicsWithTypeDissipation>(numpts, Nvert);

    shared_ptr<VertexQuadraticEnergyWithTension> avm =
        make_shared<VertexQuadraticEnergyWithTension>(numpts, 1.0, 4.0, reproducible, runSPV);

    if (RNDseed == -1) RNDseed = (int)clock() + ensemble;
    avm->noise.setReproducibleSeed(RNDseed);
    spp->noise.setReproducibleSeed(RNDseed);

    // Set type-dependent dissipation coefficients
    spp->setHomotypicDissipation(chiSame);
    spp->setHeterotypicDissipation(chiDiff);
    spp->setMu(mu);

    avm->setv0DrvD(0.1, Dr, vD);
    avm->setT1Threshold(0.1);

    // ---- initial conditions (cell types) --------------------------------
    {
        ArrayHandle<int>      h_ct(avm->cellType,         access_location::host, access_mode::overwrite);
        ArrayHandle<Dscalar2> h_p (avm->cellPositions,    access_location::host, access_mode::read);
        ArrayHandle<Dscalar2> h_a (avm->AreaPeriPreferences, access_location::host, access_mode::overwrite);

        Dscalar boxsize = sqrt((Dscalar)numpts);
        int cellcount = 0;
        std::vector<int> mc_points;

        if (IC_switch == 0)
        {
            for (int ii = 0; ii < numpts/2; ++ii)
            {
                h_ct.data[ii] = 1;
                h_a.data[ii].x = a0a; h_a.data[ii].y = p0a;
            }
            for (int ii = numpts/2; ii < numpts; ++ii)
            {
                h_ct.data[ii] = 0;
                h_a.data[ii].x = a0b; h_a.data[ii].y = p0b;
            }
        }
        else if (IC_switch == 1)
        {
            Dscalar2 c_point = {4.0*boxsize/5.0, boxsize/2.0};
            mc_points = kClosest(h_p, numpts, c_point, migratingcells);
            for (int ii = 0; ii < numpts; ++ii)
            {
                if (std::find(mc_points.begin(), mc_points.end(), ii) != mc_points.end())
                {
                    ++cellcount;
                    h_ct.data[ii] = (cellcount <= activecells) ? 1 : 2;
                    h_a.data[ii].x = a0a; h_a.data[ii].y = p0a;
                }
                else
                {
                    h_ct.data[ii] = 0;
                    h_a.data[ii].x = a0b; h_a.data[ii].y = p0b;
                }
            }
        }
        else if (IC_switch == 2)
        {
            Dscalar2 c_point = {4.0*boxsize/5.0, boxsize/2.0};
            mc_points = kClosest_x(h_p, numpts, c_point, migratingcells);
            for (int ii = 0; ii < numpts; ++ii)
            {
                h_ct.data[ii] = 0;
                h_a.data[ii].x = a0b; h_a.data[ii].y = p0b;
            }
            for (int ii : mc_points)
            {
                ++cellcount;
                h_ct.data[ii] = (cellcount <= activecells) ? 1 : 2;
                h_a.data[ii].x = a0a; h_a.data[ii].y = p0a;
            }
        }
        else if (IC_switch == 3)
        {
            Dscalar2 c_point = {4.0*boxsize/5.0, boxsize/2.0};
            mc_points = kClosest_x(h_p, numpts, c_point, migratingcells);
            for (int ii = 0; ii < numpts; ++ii)
            {
                h_ct.data[ii] = 0;
                h_a.data[ii].x = a0b; h_a.data[ii].y = p0b;
            }
            for (int ii : mc_points)
            {
                ++cellcount;
                h_ct.data[ii] = (cellcount <= (migratingcells - activecells)) ? 2 : 1;
                h_a.data[ii].x = a0a; h_a.data[ii].y = p0a;
            }
        }
        else
        {
            for (int ii = 0; ii < numpts; ++ii)
            {
                if (h_p.data[ii].x < sqrt((Dscalar)numpts)/2.0)
                {
                    h_ct.data[ii] = 0;
                    h_a.data[ii].x = a0a; h_a.data[ii].y = p0a;
                }
                else
                {
                    h_ct.data[ii] = 1;
                    h_a.data[ii].x = a0b; h_a.data[ii].y = p0b;
                }
            }
        }
    }// end IC handle scope

    // ---- assemble simulation --------------------------------------------
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->addUpdater(spp, avm);
    sim->setIntegrationTimestep(dt);
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);
    avm->setModuliUniform(Ka, Kp);
    avm->setSurfaceTension(gammas);
    avm->setUseSurfaceTension(true);

    // ---- equilibration --------------------------------------------------
    printf("Starting equilibration (RNGseed=%i, chiSame=%.4f, chiDiff=%.4f)\n",
           RNDseed, chiSame, chiDiff);
    for (int ii = 0; ii < initSteps; ++ii)
    {
        sim->performTimestep();
        avm->computeForces();
        if (ii % ((int)(0.2 / dt)) == 0)
            ncdat.WriteState(avm);
    }
    printf("Equilibration done.\n");

    if (!initializeGPU) avm->setCPU(false);

    // ---- switch on activity and pinning ---------------------------------
    {
        ArrayHandle<Dscalar>  h_cd(avm->cellDirectors,  access_location::host, access_mode::overwrite);
        ArrayHandle<Dscalar2> h_v (avm->cellVelocities, access_location::host, access_mode::read);
        ArrayHandle<Dscalar3> h_mot(avm->Motility,      access_location::host, access_mode::overwrite);
        ArrayHandle<Dscalar2> h_m (avm->Moduli,         access_location::host, access_mode::overwrite);
        ArrayHandle<int>      h_ct(avm->cellType,       access_location::host, access_mode::readwrite);
        ArrayHandle<Dscalar2> h_p (avm->cellPositions,  access_location::host, access_mode::read);

        for (int ii = 0; ii < numpts; ++ii)
            if (h_p.data[ii].x < 2.0)
            {
                h_ct.data[ii] = 5;
                h_m.data[ii].x = 1000.0;
            }

        for (int ii = 0; ii < numpts; ++ii)
        {
            if (h_ct.data[ii] == 1)
            {
                h_cd.data[ii]    = M_PI;
                h_mot.data[ii].x = v0;
                h_mot.data[ii].y = Dr_transplant;
                h_mot.data[ii].z = vD_transplant;
            }
            else if (h_ct.data[ii] == 2)
            {
                h_cd.data[ii]    = M_PI;
                h_mot.data[ii].x = u0;
                h_mot.data[ii].y = Dr_transplant;
                h_mot.data[ii].z = vD_transplant;
            }
            else if (h_ct.data[ii] == 0)
            {
                h_cd.data[ii]    = avm->noise.getRealUniform(-M_PI, M_PI);
                h_mot.data[ii].x = v_back;
                h_mot.data[ii].y = Dr_back;
                h_mot.data[ii].z = vD_back;
            }
            else  // type 5 (pinned wall)
            {
                h_cd.data[ii]    = 0.0;
                h_mot.data[ii].x = 0.0;
                h_mot.data[ii].y = 0.0;
                h_mot.data[ii].z = 0.0;
            }
        }
    }// end activity handle scope

    // ---- production run -------------------------------------------------
    dynamicalFeatures dynFeat(avm->returnPositions(), avm->Box);
    std::cout << posixpath.str() << "\n";

    clock_t t1 = clock();
    for (int ii = 0; ii < tSteps; ++ii)
    {
        if (ii % ((int)(saveInterval / dt)) == 0)
        {
            printf("t=%i  E=%.6f  MSD=%.6f\n",
                   ii, avm->computeEnergy(),
                   dynFeat.computeMSD(avm->returnPositions()));
            ncdat.WriteState(avm);
        }
        sim->performTimestep();
        avm->computeForces();
    }

    printf("t=%i  E=%.6f  MSD=%.6f\n",
           tSteps, avm->computeEnergy(),
           dynFeat.computeMSD(avm->returnPositions()));
    ncdat.WriteState(avm);

    clock_t t2 = clock();
    Dscalar steptime = (t2 - t1) / (Dscalar)CLOCKS_PER_SEC / tSteps;
    cout << "Time per step: " << steptime << " s\n";
    cout << "Total time:    " << (t2 - t1) / (Dscalar)CLOCKS_PER_SEC << " s\n";
    cout << avm->reportq() << "\n";

    if (initializeGPU) cudaDeviceReset();
    return 0;
}
