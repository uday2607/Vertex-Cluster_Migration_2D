#define ENABLE_CUDA
#include "DatabaseNetCDFSPV_Force.h"
/*! \file DatabaseNetCDFSPV_Force.cpp */

SPVDatabaseNetCDF_Force::SPVDatabaseNetCDF_Force(int np, string fn, NcFile::FileMode mode, bool exclude)
    : BaseDatabaseNetCDF(fn,mode),
      Nv(np),
      Current(0),
      exclusions(exclude)
    {
    switch(Mode)
        {
        case NcFile::ReadOnly:
            break;
        case NcFile::Write:
            GetDimVar();
            break;
        case NcFile::Replace:
            SetDimVar();
            break;
        case NcFile::New:
            SetDimVar();
            break;
        default:
            ;
        };
    }

void SPVDatabaseNetCDF_Force::SetDimVar()
    {
    //Set the dimensions
    recDim = File.add_dim("rec");
    NvDim  = File.add_dim("Nv",  Nv);
    dofDim = File.add_dim("dof", Nv*2);
    boxDim = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    timeVar       = File.add_var("time", ncDscalar,recDim, unitDim);
    means0Var     = File.add_var("means0", ncDscalar,recDim, unitDim);
    posVar        = File.add_var("pos", ncDscalar,recDim, dofDim);
    forceVar      = File.add_var("force", ncDscalar,recDim, dofDim);
    typeVar       = File.add_var("type", ncInt,recDim, NvDim );
    directorVar   = File.add_var("director", ncDscalar,recDim, NvDim );
    BoxMatrixVar  = File.add_var("BoxMatrix", ncDscalar,recDim, boxDim);
    if(exclusions)
        exVar     = File.add_var("externalForce", ncDscalar,recDim, dofDim);
    }

void SPVDatabaseNetCDF_Force::GetDimVar()
    {
    //Get the dimensions
    recDim = File.get_dim("rec");
    boxDim = File.get_dim("boxdim");
    NvDim  = File.get_dim("Nv");
    dofDim = File.get_dim("dof");
    unitDim = File.get_dim("unit");
    //Get the variables
    posVar       = File.get_var("pos");
    forceVar     = File.get_var("force");
    typeVar      = File.get_var("type");
    directorVar  = File.get_var("director");
    means0Var    = File.get_var("means0");
    BoxMatrixVar = File.get_var("BoxMatrix");
    timeVar      = File.get_var("time");
    if(exclusions)
        exVar    = File.get_var("externalForce");
    }

void SPVDatabaseNetCDF_Force::WriteState(STATE s, Dscalar time, int rec)
    {
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = s->currentTime;

    std::vector<Dscalar> boxdat(4,0.0);
    Dscalar x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    std::vector<Dscalar> posdat(2*Nv);
    std::vector<Dscalar> apdat(2*Nv);
    std::vector<Dscalar> forcedat(2*Nv);
    std::vector<Dscalar> directordat(Nv);
    std::vector<int> typedat(Nv);
    int idx = 0;
    Dscalar means0=0.0;

    ArrayHandle<Dscalar2> h_p(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(s->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(s->cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_ex(s->exclusions,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        Dscalar px = h_p.data[pidx].x;
        Dscalar py = h_p.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        double fx = h_f.data[pidx].x;
        double fy = h_f.data[pidx].y;
        forcedat[(2*idx)] = fx;
        forcedat[(2*idx)+1] = fy;
        directordat[ii] = h_cd.data[pidx];
        if(h_ex.data[ii] == 0)
            typedat[ii] = h_ct.data[pidx];
        else
            typedat[ii] = h_ct.data[pidx]-5;
        idx +=1;
        };
//    means0 = means0/Nv;
    means0 = s->reportq();

    //Write all the data
    means0Var   ->put_rec(&means0, rec);
    timeVar     ->put_rec(&time, rec);
    posVar      ->put_rec(&posdat[0], rec);
    forceVar    ->put_rec(&forcedat[0], rec);
    typeVar     ->put_rec(&typedat[0], rec);
    directorVar ->put_rec(&directordat[0], rec);
    BoxMatrixVar->put_rec(&boxdat[0], rec);
    if(exclusions)
        {
        ArrayHandle<Dscalar2> h_ef(s->external_forces,access_location::host,access_mode::read);
        std::vector<Dscalar> exdat(2*Nv);
        int id = 0;
        for (int ii = 0; ii < Nv; ++ii)
            {
            int pidx = s->tagToIdx[ii];
            Dscalar px = h_ef.data[pidx].x;
            Dscalar py = h_ef.data[pidx].y;
            exdat[(2*id)] = px;
            exdat[(2*id)+1] = py;
            id +=1;
            };
        exVar      ->put_rec(&exdat[0],     rec);
        };

    File.sync();
    }