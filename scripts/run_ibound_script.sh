#!/bin/bash
MODEL_TYPE="complete"
ALGORITHMS="ijgp mbe wmbe mbr gbr"

for i in {1..10};
do
  tmux new-session -d -s "ibound_complete$i" "python3 run_ibound.py --model-type $MODEL_TYPE --algorithms $ALGORITHMS --seed $i"
done

MODEL_TYPE="grid"
ALGORITHMS="ijgp mbe wmbe mbr gbr"

for i in {1..10};
do
  tmux new-session -d -s "ibound_grid$i" "python3 run_ibound.py --model-type $MODEL_TYPE --algorithms $ALGORITHMS --seed $i"
done
