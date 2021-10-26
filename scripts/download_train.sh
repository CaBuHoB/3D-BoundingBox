#!/bin/bash -x

echo "Downloading dataset ..."
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -O Kitti/DOI.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip -O Kitti/DOC.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip -O Kitti/DOL.zip

echo "Uncompressing dataset ..."
unzip -u Kitti/DOI.zip -d Kitti/
unzip -u Kitti/DOC.zip -d Kitti/
unzip -u Kitti/DOL.zip -d Kitti/

echo "Remove dataset zips ..."
rm Kitti/DOI.zip 
rm Kitti/DOC.zip 
rm Kitti/DOL.zip 