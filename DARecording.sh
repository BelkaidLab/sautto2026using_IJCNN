#!/bin/bash
X=100
RUNTYPE=("Rewarded" "Unrewarded")
#RUNTYPE=("Trials")
MODEL=("FULLINH" "FULLDIR")
VALUE=("Hi" "Lo")
#VALUE=("Mi")

for runtype in "${RUNTYPE[@]}"; do
  for model in "${MODEL[@]}"; do
    for value in "${VALUE[@]}"; do
      for i in $(seq 0 $X); do  
        echo "Running: $reward $model $value $i"
        /home/roberto-sautto/Desktop/Project3/.conda/bin/python /home/roberto-sautto/Desktop/Project3/simulation.py "$i" "$runtype" "$model" "$value"
      done
    done
  done
done
