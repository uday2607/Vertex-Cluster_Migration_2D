#!/bin/bash
# End-to-end demo: compile -> simulate three interfacial tensions in SPV and in AVM -> movie.
#
#   bash run_all.sh            # quick demo: 50,000 steps (T = 10), a few minutes
#   bash run_all.sh full       # paper length: 500,000 steps (T = 100), ~20 min per run
#
# Output goes to DATA/demo/ (not tracked by git) and RESULTS/demo/.
# Set PYTHON to pick an interpreter (needs numpy, scipy, matplotlib, freud-analysis, networkx, numba, ffmpeg).
set -euo pipefail
cd "$(dirname "$0")"
source load_modules.sh      # the binaries need the CUDA runtime library at run time

MODE="${1:-quick}"
STEPS=50000;  SAVE=0.5      # quick: save a frame every 0.5 time units so the short run still makes a movie
[ "$MODE" = "full" ] && { STEPS=500000; SAVE=5; }
PYTHON="${PYTHON:-python3}"
OUT=DATA/demo
mkdir -p "$OUT" RESULTS/demo logs

echo "=== [1/3] Compiling ==="
bash load_mod_and_make.sh spv
bash load_mod_and_make.sh avm

echo "=== [2/3] Simulating (6 runs in parallel, $STEPS steps each) ==="
# Homogeneous clusters: 8 cells, all active (-m 8 -l 8), no passive followers (-u 0).
# -G = interfacial tension gamma between the active cells and the background tissue.
for g in 0 0.2 5; do
    bin/spv_pos_gamma.out -t $STEPS -e 0.0002 -n 400 -m 8 -l 8 -v 1.5 -u 0 -i 1 -x 5 \
        -G $g -H 0 -I 0 -S $SAVE -O $OUT > logs/spv_gamma$g.log 2>&1 &
done
for g in 0 2 5; do
    bin/avm_pos_gamma.out -t $STEPS -e 0.0002 -n 400 -m 8 -l 8 -v 0.75 -u 0 -i 1 -x 5 \
        -G $g -H 0 -I 0 -S $SAVE -O $OUT > logs/avm_gamma$g.log 2>&1 &
done
wait
echo "simulations done:"; ls -1 $OUT/*/*.nc

echo "=== [3/3] Movie ==="
SPV=($(ls $OUT/SPV_POS_GAMMA/*_T1_0_T2*.nc $OUT/SPV_POS_GAMMA/*_T1_0.2_T2*.nc $OUT/SPV_POS_GAMMA/*_T1_5_T2*.nc))
AVM=($(ls $OUT/AVM_POS_GAMMA/*_T1_0_T2*.nc $OUT/AVM_POS_GAMMA/*_T1_2_T2*.nc $OUT/AVM_POS_GAMMA/*_T1_5_T2*.nc))
"$PYTHON" - "${SPV[@]}" "${AVM[@]}" <<'PY'
import json, sys
f = sys.argv[1:]
spec = json.load(open("PYTHON_ANALYSIS/movies/specs/fig2.json"))
spec["files"] = f
spec["title"] = "Demo run: interfacial tension regimes"
json.dump(spec, open("DATA/demo/demo_spec.json", "w"), indent=2)
PY
"$PYTHON" PYTHON_ANALYSIS/movies/make_grid_movie.py --spec DATA/demo/demo_spec.json --out RESULTS/demo/regimes_demo.mp4
echo "=== ALL DONE: RESULTS/demo/regimes_demo.mp4 ==="
