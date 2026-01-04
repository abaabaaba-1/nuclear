#!/bin/bash

# Define algorithms and seeds
ALGOS=("random" "sa" "ga")
SEEDS=(43 44 45 46)
CONFIG="problem/stellarator_coil_gsco_lite/config.yaml"

# Create logs directory
mkdir -p logs

echo "Starting extra baseline experiments..."

for seed in "${SEEDS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "Running $algo with seed $seed..."
        python3 run_gsco_baselines.py --algo "$algo" --seed "$seed" --config "$CONFIG" > "logs/${algo}_${seed}.log" 2>&1
        if [ $? -eq 0 ]; then
            echo "Finished $algo with seed $seed."
        else
            echo "Error running $algo with seed $seed. Check logs/${algo}_${seed}.log"
        fi
    done
done

echo "All experiments completed."
