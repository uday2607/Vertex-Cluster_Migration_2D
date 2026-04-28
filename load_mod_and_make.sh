#!/bin/bash
# Script by UR. Loads all the modules and compiles all the files

set -euo pipefail

module load cuda/12.0.0 cgal/4.9.1 eigen/3.4.0 netcdf/4.1.3

make -Bj4
