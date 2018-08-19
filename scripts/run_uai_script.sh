#!/bin/bash
MODEL_NAME="linkage_"
ALGORITHMS="bp ijgp mbe wmbe mbr gbr"

for i in {11..27};
do
  tmux new-session -d -s "uai_linkage$i" "python3 run_uai.py --model-name $MODEL_NAME$i --algorithms $ALGORITHMS"
done

MODEL_NAME="Promedus_"
ALGORITHMS="bp ijgp mbe wmbe mbr gbr"

for i in {11..38};
do
  tmux new-session -d -s "uai_Promedus$i" "python3 run_uai.py --model-name $MODEL_NAME$i --algorithms $ALGORITHMS"
done
