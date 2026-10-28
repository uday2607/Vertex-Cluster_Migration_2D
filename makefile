# Copyright (c) 2016 - 2019 Daniel M. Sussman
# Copyright (C) 2021        Alois Schloegl

#standard places to find cuda files
CUDA_INC = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib64
CUDA_LIB2 = $(CUDA_HOME)/lib

CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc

INCLUDES = -I. -I./src/ -I./ext_src/ -I./inc/ -I$(CUDA_INC)
INCLUDES += -I$(CGAL_DIR)/Periodic_2_triangulation_2/include/
LIB_CUDA = -L. -L$(CUDA_LIB) -L$(CUDA_LIB2) -lcuda -lcudart
LIB_CGAL += -L$(CGAL_DIR)/lib -L/usr/lib/x86_64-linux-gnu/
LIB_CGAL += -L$(CGAL_DIR)/lib -lCGAL -lCGAL_Core -lgmp -lmpfr
LIB_NETCDF = -lnetcdf -lnetcdf_c++ -L/opt/local/lib

#common flags
COMMONFLAGS += $(INCLUDES) -std=c++14 -DCGAL_DISABLE_ROUNDING_MATH_CHECK -O3
NVCCFLAGS += -D_FORCE_INLINES $(COMMONFLAGS) -Wno-deprecated-gpu-targets -arch=sm_50 #-Xptxas -fmad=false#-O0#-dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w -frounding-math -march=native
CFLAGS += $(COMMONFLAGS) -frounding-math -march=native

CUOBJ_DIR=obj/cuobj
MODULES = databases models updaters utility
INCLUDES += -I./inc/databases -I./inc/models -I./inc/updaters -I./inc/utility -I./inc/analysis
OBJ_DIR=obj
SRC_DIR=src
BIN_DIR=bin

.SECONDARY:

# Every driver lives in one of the model folders; any new .cpp dropped into
# SPV/, AVM/ or NODAL/ is picked up automatically and built to bin/<name>.out
PROG_DIRS := SPV AVM NODAL
PROGS := $(foreach d,$(PROG_DIRS),$(wildcard $(d)/*.cpp))
PROG_NAMES := $(notdir $(basename $(PROGS)))
PROG_OBJS := $(addprefix $(OBJ_DIR)/,$(addsuffix .main.o,$(PROG_NAMES)))
PROG_MAINS := $(addprefix $(BIN_DIR)/,$(addsuffix .out,$(PROG_NAMES)))
vpath %.cpp $(PROG_DIRS)


CPP_FILES := $(wildcard src/*/*.cpp)
CPP_FILES += $(wildcard src/*.cpp)
CU_FILES := $(wildcard src/*/*.cu)
CU_FILES += $(wildcard src/*.cu)

CLASS_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_FILES))


#cuda objects
$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu 
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

#cpp class objects
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#program objects
$(OBJ_DIR)/%.main.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#Programs
$(BIN_DIR)/%.out: $(OBJ_DIR)/%.main.o $(CLASS_OBJS) $(CU_OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+

#target rules

all:build

float: CXXFLAGS += -DSCALARFLOAT
float: NVCCFLAGS += -DSCALARFLOAT
float: build

debug: CXXFLAGS += -g -DCUDATHREADSYNC -DDEBUGFLAGUP
debug: NVCCFLAGS += -g -lineinfo -Xptxas --generate-line-info -DDEBUGFLAGUP # -G note that in debug mode noise will always be reproducible
debug: build
build: $(CLASS_OBJS) $(CU_OBJS) $(PROG_MAINS)

# build one model family:  make spv | make avm | make nodal
spv:   $(filter $(BIN_DIR)/spv%,$(PROG_MAINS))
avm:   $(filter $(BIN_DIR)/avm%,$(PROG_MAINS))
nodal: $(filter $(BIN_DIR)/nodal%,$(PROG_MAINS))

clean: 
	rm -f $(PROG_OBJS) $(CLASS_OBJS) $(CU_OBJS) $(PROG_OBJS) $(PROG_MAINS)

print-%  : ; @echo $* = $($*)

