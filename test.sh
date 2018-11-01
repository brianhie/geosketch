#!/bin/bash

stdbuf -oL python3 bin/artificial_volume.py > artificial_volume.log 2>&1
echo "art_vol"

stdbuf -oL python3 bin/artificial_density.py > artificial_density.log 2>&1
echo "art_den"

stdbuf -oL python3 bin/pbmc.py > pbmc.log 2>&1
echo "pbmc"

stdbuf -oL python3 bin/simulate_varied.py > simulate_varied.log 2>&1
echo "sim_var"

stdbuf -oL python3 bin/tabula_ss2.py > tabula_ss2.log 2>&1
echo "tab_ss2"

stdbuf -oL python3 bin/tabula_10x.py > tabula_10x.log 2>&1
echo "tab_10x"

stdbuf -oL python3 bin/mouse_brain.py > mouse_brain_raw.log 2>&1
echo "mou_bra"
