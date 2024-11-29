#!/bin/bash

files=(2011_09_26_calib.zip
2011_09_26_drive_0001
2011_09_26_drive_0002
2011_09_26_drive_0005
2011_09_28_calib.zip
2011_09_28_drive_0001
2011_09_28_drive_0002
2011_09_28_drive_0016
2011_09_29_calib.zip
2011_09_29_drive_0004
2011_09_29_drive_0026
2011_09_29_drive_0071
2011_09_30_calib.zip
2011_09_30_drive_0016
2011_09_30_drive_0018
2011_09_30_drive_0020
2011_10_03_calib.zip
2011_10_03_drive_0027
2011_10_03_drive_0034
2011_10_03_drive_0042)

current_foldername="2011_09_26"
for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
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
        find . -type d \( -name "image_00" -o -name "image_01" \) -exec rm -r {} +      # remove folders with grayscale images
        
        if [[ ${i:0:10} != ${current_foldername} ]]; then
                # zip -r $current_foldername'.zip' $current_foldername
                mv $current_foldername /content/gdrive/MyDrive/DL_Data/$current_foldername
                current_foldername=${i:0:10}
        fi
done
mv $current_foldername /content/gdrive/MyDrive/DL_Data/$current_foldername
