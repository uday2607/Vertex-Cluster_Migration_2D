#!/bin/bash
# Compile, simulate (6 parallel), then process all results.
# Run: nohup bash run_all.sh > logs/master.log 2>&1 &

set -euo pipefail
mkdir -p logs DATA/POS_GAMMA_HETERO DATA/POS_GAMMA_VERTEX_HETERO

echo "=== [1/3] Compiling ===" | tee logs/master.log
module load cuda/12.0.0 cgal/4.9.1 eigen/3.4.0 netcdf/4.1.3
make spv_pos_gamma_hetero_adhesion.out spv_neg_gamma_hetero_adhesion.out \
     avm_pos_gamma_hetero_adhesion.out avm_neg_gamma_hetero_adhesion.out 2>&1 | tee -a logs/master.log
echo "=== Compile done ===" | tee -a logs/master.log

echo "=== [2/3] Removing old nc files ===" | tee -a logs/master.log
rm -f DATA/POS_GAMMA_HETERO/*.nc DATA/POS_GAMMA_VERTEX_HETERO/*.nc

echo "=== [3/3] Launching 6 simulations ===" | tee -a logs/master.log
for gamma in 0 0.8 1.5; do
    ./spv_pos_gamma_hetero_adhesion.out -t 100000 -e 0.0002 -n 400 -v 3 -u 0 -y 0.01 \
        -b 0.005 -j 0.02 -i 3 -G $gamma -H $gamma -I 0 -x 1 \
        > logs/spv_gamma${gamma}.log 2>&1 &
    echo "[LAUNCHED] SPV gamma=$gamma (pid $!)" | tee -a logs/master.log

    ./avm_pos_gamma_hetero_adhesion.out -t 100000 -e 0.0002 -n 400 -v 3 -u 0 -y 0.01 \
        -b 0.005 -j 0.02 -i 3 -G $gamma -H $gamma -I 0 -x 1 \
        > logs/avm_gamma${gamma}.log 2>&1 &
    echo "[LAUNCHED] AVM gamma=$gamma (pid $!)" | tee -a logs/master.log
done

echo "Waiting for all 6 sims..." | tee -a logs/master.log
wait
echo "=== All sims done ===" | tee -a logs/master.log
ls -lh DATA/POS_GAMMA_HETERO/ DATA/POS_GAMMA_VERTEX_HETERO/ | tee -a logs/master.log

echo "=== Processing results ===" | tee -a logs/master.log
PYTHON=/nfs/scistore18/hannegrp/ugubbala/cell_env/bin/python
for f in DATA/POS_GAMMA_HETERO/*.nc DATA/POS_GAMMA_VERTEX_HETERO/*.nc; do
    $PYTHON -m PYTHON_ANALYSIS.cellgpu_analysis run-all \
        --input "$f" --voronoi --stride 1 --fps 5 --dpi 200 \
        >> logs/master.log 2>&1 &
done
wait
echo "=== ALL DONE ===" | tee -a logs/master.log
ls -lh RESULTS/MOVIES/ RESULTS/PLOTS/ | tee -a logs/master.log
