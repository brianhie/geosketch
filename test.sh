#!/bin/bash

stdbuf -oL python bin/ica.py > ica.log 2>&1 &
stdbuf -oL python bin/mouse_brain.py > mouse_brain.log 2>&1 &
stdbuf -oL python bin/pallium.py > pallium.log 2>&1 &
stdbuf -oL python bin/pbmc.py > pbmc.log 2>&1 &
stdbuf -oL python bin/simulate_varied.py > simulate_varied.log 2>&1 &

wait
