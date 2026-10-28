# Source this file to get the build/run environment:   source load_modules.sh
# On a cluster with environment modules it loads the libraries the binaries need (the compiled
# programs need the CUDA runtime library even though the simulations run on the CPU). Edit the module
# names to match your site. On a workstation with everything installed, set CUDA_HOME and CGAL_DIR instead.
if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    module load cuda/12.0.0 cgal/4.9.1 eigen/3.4.0 netcdf/4.1.3
fi
