#! /bin/bash -x

echo "Downloading 3d bbox weights ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA" -O weights/epoch_10.pkl && rm -rf /tmp/cookies.txt
echo "Downloading yolo weights ..."
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

mkdir -p eval/video/
echo "Downloading video (images) for test ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QBvwPBWKHV0Y6zle6Y2iI-vvE5pMrs8x' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QBvwPBWKHV0Y6zle6Y2iI-vvE5pMrs8x" -O eval/video/2011_09_26.tar.gz && rm -rf /tmp/cookies.txt
echo "Uncompressing images ..."
tar -xvzf eval/video/2011_09_26.tar.gz -C eval/video/
rm eval/video/2011_09_26.tar.gz
