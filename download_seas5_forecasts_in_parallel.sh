#!/bin/bash

mkdir -p logs/seas5_download_logs/

for ((i=1; i<7; i++)); do

        python3 icenet/download_seas5_forecasts.py --leadtime $i > logs/seas5_download_logs/"$i.txt" 2>&1 &

        echo -e "Running $(jobs -p | wc -w) jobs after submitting $init_date"

        sleep 1
done
