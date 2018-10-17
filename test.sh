#!/bin/bash

stdbuf -oL python bin/tabula_ss2.py > tabula_ss2.log 2>&1 &
stdbuf -oL python bin/tabula_10x.py > tabula_10x.log 2>&1 &
stdbuf -oL python bin/simulate_varied.py > simulate_varied.log 2>&1 &
stdbuf -oL python bin/mouse_brain.py > mouse_brain_raw.log 2>&1 &

wait
