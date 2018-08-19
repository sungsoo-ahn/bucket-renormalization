#!/bin/bash
MODEL_TYPE="complete"
ALGORITHMS="mf bp ijgp mbe wmbe mbr gbr"

for i in {1..10};
do
  tmux new-session -d -s "delta_complete$i" "python3 run_delta.py --model-type $MODEL_TYPE --algorithms $ALGORITHMS --seed $i"
done

MODEL_TYPE="grid"
ALGORITHMS="mf bp ijgp mbe wmbe mbr gbr"

for i in {1..10};
do
  tmux new-session -d -s "delta_grid$i" "python3 run_delta.py --model-type $MODEL_TYPE --algorithms $ALGORITHMS --seed $i"
done
