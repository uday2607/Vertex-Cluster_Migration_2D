#!/bin/bash
# Compile every driver (SPV/, AVM/, NODAL/) into bin/<name>.out.
# Boost headers are required by CGAL (libboost-dev on Debian/Ubuntu, or the boost module on a cluster).
#
# Usage:   bash load_mod_and_make.sh            # build everything
#          bash load_mod_and_make.sh spv        # only SPV drivers (also: avm, nodal)
set -euo pipefail
cd "$(dirname "$0")"
source load_modules.sh
make -j4 "${1:-build}"
