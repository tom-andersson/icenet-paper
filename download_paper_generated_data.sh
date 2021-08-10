#!/bin/bash

# Results
mkdir -p results/uncertainty_results_temp/
mkdir -p results/forecast_results_temp/
mkdir -p results/permute_and_predict_results_temp/
wget -O results/uncertainty_results_temp/sip_bounding_results.csv 'https://ramadda.data.bas.ac.uk/repository/entry/get/sip_bounding_results.csv?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL3Jlc3VsdHMvdW5jZXJ0YWludHlfcmVzdWx0cy9zaXBfYm91bmRpbmdfcmVzdWx0cy5jc3Y%3D'
wget -O results/uncertainty_results_temp/ice_edge_region_results.csv 'https://ramadda.data.bas.ac.uk/repository/entry/get/ice_edge_region_results.csv?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL3Jlc3VsdHMvdW5jZXJ0YWludHlfcmVzdWx0cy9pY2VfZWRnZV9yZWdpb25fcmVzdWx0cy5jc3Y%3D'
wget -O results/uncertainty_results_temp/uncertainty_results.csv 'https://ramadda.data.bas.ac.uk/repository/entry/get/uncertainty_results.csv?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL3Jlc3VsdHMvdW5jZXJ0YWludHlfcmVzdWx0cy91bmNlcnRhaW50eV9yZXN1bHRzLmNzdg%3D%3D'
wget -O results/forecast_results_temp/2021_07_01_183913_forecast_results.csv 'https://ramadda.data.bas.ac.uk/repository/entry/get/2021_07_01_183913_forecast_results.csv?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL3Jlc3VsdHMvZm9yZWNhc3RfcmVzdWx0cy8yMDIxXzA3XzAxXzE4MzkxM19mb3JlY2FzdF9yZXN1bHRzLmNzdg%3D%3D'
wget -O results/permute_and_predict_results_temp/permute_and_predict_results.csv 'https://ramadda.data.bas.ac.uk/repository/entry/get/permute_and_predict_results.csv?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL3Jlc3VsdHMvcGVybXV0ZV9hbmRfcHJlZGljdF9yZXN1bHRzL3Blcm11dGVfYW5kX3ByZWRpY3RfcmVzdWx0cy5jc3Y%3D'

# Sea Ice Outlook historical Sea Ixe Extent error data
mkdir -p data/
wget -O data/sea_ice_outlook_errors.csv 'https://ramadda.data.bas.ac.uk/repository/entry/get/sea_ice_outlook_errors.csv?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL3Jlc3VsdHMvb3V0bG9va19lcnJvcnMvc2VhX2ljZV9vdXRsb29rX2Vycm9ycy5jc3Y%3D'

# IceNet Sea Ice Probability forecasts (2012-2020)
folder=data/forecasts/icenet/2021_06_15_1854_icenet_nature_communications/unet_tempscale/
mkdir -P $folder
wget -O $folder/icenet_sip_forecasts_tempscaled.nc 'https://ramadda.data.bas.ac.uk/repository/entry/get/icenet_sip_forecasts_tempscaled.nc?entryid=synth%3A71820e7d-c628-4e32-969f-464b7efb187c%3AL2ZvcmVjYXN0X25ldGNkZi9pY2VuZXRfc2lwX2ZvcmVjYXN0c190ZW1wc2NhbGVkLm5j'
