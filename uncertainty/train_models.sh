#!/bin/bash
# Script to train models for an increasing number of time steps.

let num_times_line=2

let head_num=$num_times_line-1
let tail_num=$num_times_line+1

for num_times in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    echo $num_times
    head -n $head_num configs/train.config > train_temp.config
    echo "time_size = $num_times" >> train_temp.config
    tail -n +$tail_num configs/train.config >> train_temp.config

    python train.py --data_dir data/ --output_dir models/uncertain_$num_times --train_config train_temp.config
done
rm train_temp.config