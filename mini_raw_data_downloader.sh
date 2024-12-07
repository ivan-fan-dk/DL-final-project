#!/bin/bash

# Create the local data directory
mkdir -p ./DL_Mini_Data

files=(2011_09_26_calib.zip
       2011_09_26_drive_0001
       2011_09_28_calib.zip
       2011_09_28_drive_0001
       2011_09_29_calib.zip
       2011_09_29_drive_0004
       2011_09_30_calib.zip
       2011_09_30_drive_0016
       2011_10_03_calib.zip
       2011_10_03_drive_0058)

current_foldername="2011_09_26"

for i in ${files[@]}; do
    if [ ${i:(-3)} != "zip" ]; then
        shortname=$i'_sync.zip'
        fullname=$i'/'$i'_sync.zip'
    else
        shortname=$i
        fullname=$i
    fi
    
    echo "Downloading: "$shortname
    wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
    unzip -o -q $shortname
    rm $shortname
    
    # Clean up unnecessary files
    find . -type d \( -name "image_00" -o -name "image_01" \) -exec rm -r {} +
    find . -type f -name "*.png" | awk -F'/' '{print $NF, $0}' | awk '$1 ~ /^[0-9]+\.png$/ && $1 > "0000000005.png" {print $2}' | xargs rm
    find . -type f -name "*.bin" | awk -F'/' '{print $NF, $0}' | awk '$1 ~ /^[0-9]+\.bin$/ && $1 > "0000000005.bin" {print $2}' | xargs rm
    find . -type f -name "*.txt" | awk -F'/' '{print $NF, $0}' | awk '$1 ~ /^[0-9]+\.txt$/ && $1 > "0000000005.txt" {print $2}' | xargs rm
    
    if [[ ${i:0:10} != ${current_foldername} ]]; then
        mv $current_foldername ./DL_Mini_Data/$current_foldername
        current_foldername=${i:0:10}
    fi
done

# Move the last folder
mv $current_foldername ./DL_Mini_Data/$current_foldername

# Copy to Google Drive if needed
if [ -d "/content/gdrive/MyDrive" ]; then
    mkdir -p /content/gdrive/MyDrive/DL_Mini_Data
    cp -r ./DL_Mini_Data/* /content/gdrive/MyDrive/DL_Mini_Data/
fi
