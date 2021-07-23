#!/bin/bash

mkdir -p logs/era5_download_logs/

python3 icenet/download_era5_data.py --var tas > logs/era5_download_logs/tas.txt 2>&1 &
python3 icenet/download_era5_data.py --var tos > logs/era5_download_logs/tos.txt 2>&1 &
python3 icenet/download_era5_data.py --var ta500 > logs/era5_download_logs/ta500.txt 2>&1 &
python3 icenet/download_era5_data.py --var rsds_and_rsus > logs/era5_download_logs/rsds_and_rsus.txt 2>&1 &
python3 icenet/download_era5_data.py --var psl > logs/era5_download_logs/psl.txt 2>&1 &
python3 icenet/download_era5_data.py --var zg500 > logs/era5_download_logs/zg500.txt 2>&1 &
python3 icenet/download_era5_data.py --var zg250 > logs/era5_download_logs/zg250.txt 2>&1 &
python3 icenet/download_era5_data.py --var ua10 > logs/era5_download_logs/ua10.txt 2>&1 &
python3 icenet/download_era5_data.py --var uas > logs/era5_download_logs/uas.txt 2>&1 &
python3 icenet/download_era5_data.py --var vas > logs/era5_download_logs/vas.txt 2>&1 &
