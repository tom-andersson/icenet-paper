#!/bin/bash

mkdir -p logs/seas5_download_logs/

for ((leadtime=1; leadtime<7; leadtime++)); do

        python3 icenet/download_seas5_forecasts.py --leadtime $leadtime > logs/seas5_download_logs/"$leadtime.txt" 2>&1 &

        echo -e "Running $(jobs -p | wc -w) jobs after submitting lead time: $leadtime month/s"

        sleep 2
done
