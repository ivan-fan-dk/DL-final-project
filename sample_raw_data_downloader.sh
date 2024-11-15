#!/bin/bash

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
                mv $current_foldername /content/gdrive/MyDrive/DL_Final_Project/$current_foldername
                current_foldername=${i:0:10}
        fi
done
mv $current_foldername /content/gdrive/MyDrive/DL_Final_Project/$current_foldername

# from google.colab import drive
# drive.mount('/content/gdrive')
# !mkdir /content/gdrive/MyDrive/DL_Final_Project/
# !chmod +x ./raw_data_downloader.sh
# !./raw_data_downloader.sh
# !git clone https://github.com/ivan-fan-dk/DL-final-project.git
# !mv DL-final-project/* ./
# !rm -rf DL-final-project
# !pip3 install -r requirements.txt
# # !mkdir formatted
# !python3 data/prepare_train_data.py ./ --dataset-format 'kitti_raw' --dump-root ./formatted/data/ --width 416 --height 128 --num-threads 4