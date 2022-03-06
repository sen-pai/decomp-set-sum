#!/bin/bash

YEARS=('2002' '2003' '2004' '2005' '2006' '2007' '2008' '2009' '2010' '2011' '2012' '2013' '2014' '2015' '2016' '2017' '2018')


for year in "${YEARS[@]}"
do
    mkdir ./ndvi_data/$year
    cd ./ndvi_data/$year
    aws s3 cp s3://mantlelabs-eu-modis-boku/modis_boku-preprocess/v1.0/ ./ --recursive --exclude "*" --include "*$year*10*OF/NDVI/ndvi.tif"
    cd ../../
done