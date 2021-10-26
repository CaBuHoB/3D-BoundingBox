# 3D Bounding Box Estimation Using Deep Learning and Geometry

This repository is not an original official implementation of the work, but a refactored codebase. Performed within the FSE coursework at Skoltech.

## Description
The code is a PyTorch implementation for this [paper](https://arxiv.org/abs/1612.00496).

![example-image](http://soroushkhadem.com/img/2d-top-3d-bottom1.png)

At the moment, it takes approximately 0.4s per frame, depending on the number of objects
detected. The speed will be improved soon. Here is the current fastest possible:

![example-video](eval/example/3d-bbox-vid.gif)

# Quickstart

Use quickstart to test the model on default data using pre-trained weights.

## Requirements
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (optional)

## Launch
Clone this repository and enter the directory using the commands below:
```
git clone https://github.com/CaBuHoB/3D-BoundingBox.git
cd 3D-BoundingBox/
```

Build docker or pull from Docker Hub:

a. ```docker build . -t 3dboundingbox```

b. ```docker pull cabuhob/3dboundingbox```

Dowload weights and test data by executing the following file
```bash
./scripts/download_test.sh
```

Run docker with CUDA:
```bash
docker run --rm -it \
    -v ${PWD}/weights/:/weights/ \
    -v ${PWD}/output/:/output/ \
    -e MODE=eval \
    -e IMWRITE=1 \
    -e OUTPUT_DIR=/output/ \
    -e DEVICE=cuda \
    -e WEIGHTS_PATH=/weights/ \
    --gpus=all \
    -e DATASET_PATH=/eval/ \
    -v ${PWD}/eval/video/2011_09_26/image_2/:/eval/ \
    3dboundinbox
```

Run docker with CPU:
```bash
docker run --rm -it \
    -v ${PWD}/weights/:/weights/ \
    -v ${PWD}/output/:/output/ \
    -e MODE=eval \
    -e IMWRITE=1 \
    -e OUTPUT_DIR=/output/ \
    -e DEVICE=cpu \
    -e WEIGHTS_PATH=/weights/ \
    -e DATASET_PATH=/eval/ \
    -v ${PWD}/eval/video/2011_09_26/image_2/:/eval/ \
    3dboundinbox
```

Generated images will appear in the folder ```${PWD}/output/```

# Development

## Train

To train the model Dowload train data by executing the following file
```bash
./scripts/download_train.sh
```

Run docker with CUDA:
```bash
docker run --rm -it \
    -v ${PWD}/weights/:/weights/ \
    -v ${PWD}/Kitti/training/:/data/ \
    -e MODE=train \
    -e DEVICE=cuda \
    -e WEIGHTS_PATH=/weights/ \
    --gpus=all \
    -e DATASET_PATH=/data/ \
    3dboundinbox
```

Run docker with CPU:
```bash
docker run --rm -it \
    -v ${PWD}/weights/:/weights/ \
    -v ${PWD}/Kitti/training/:/data/ \
    -e MODE=train \
    -e DEVICE=cpu \
    -e WEIGHTS_PATH=/weights/ \
    -e DATASET_PATH=/data/ \
    3dboundinbox
```

## Using without docker

You can use [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for fast environment creation with all dependencies.
For that run
```bash
conda env create --prefix ./env --file environment.yml
conda activate 3dboundingbox
```

To see all the options run
```bash
python Run.py --help
```



### All arguments for running the model

|                 Running files                |    Argument    | Docker argument |                                                                                     Description                                                                                    |
|:------------------------------:|:--------------------:|:---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Run.py<br>Run_no_yolo.py<br>Train.py | \-\-dataset-path |   DATASET_PATH  |                                                                          Path to directory with dataset                                                                         |
| Run.py<br>Run_no_yolo.py<br>Train.py |  \-\-calib-path  |    CALIB_PATH   |                                                                    Path file with calibrating data for camera                                                                   |
| Run.py<br>Run_no_yolo.py<br>Train.py | \-\-weights-path |   WEIGHTS_PATH  |                                                    Path to folder, where weights will be saved. By default, this is weights/                                                    |
| Run.py<br>Run_no_yolo.py<br>Train.py |    \-\-device    |      DEVICE     |                                                                             PyTorch device: cuda(default)/cpu                                                                            |
|         Run.py<br>Run_no_yolo.py        |  \-\-output-dir  |    OUTPUT_DIR   |                                     If the imwrite flag is True, the images will be saved to this directory. By default, this is output_dir/                                    |
|         Run.py<br>Run_no_yolo.py        |    \-\-imwrite   |     IMWRITE     | Flag for running the code in the mode of saving images to a folder. If this flag is used, the files are saved in output_dir. By default, images are displayed using cv2.imshow. |
|                Run.py                |  \-\-hide-debug  |    HIDE_DEBUG   |                                                              Show the 2D BoundingBox detecions on a separate image                                                              |
|                Run.py                |   \-\-show-yolo  |    SHOW_YOLO    |                                                                     Supress the printing of each 3d location                                                                    |
|                Run.py                |     \-\-video    |      VIDEO      |                                Weather or not to advance frame-by-frame as fast as possible. By default, this will pull images from ./eval/video                                |

## Examples 
Train the model
```bash
python Train.py --device=cpu
```
Run through all images in default directory (eval/image_2/), optionally with the 2D
bounding boxes also drawn. Press SPACE to proceed to next image, and any other key to exit.
```bash
python Run.py --device=cpu [--show-yolo]
```

There is also a script provided to download the default video from Kitti in ./eval/video. Or,
download any Kitti video and corresponding calibration and use `--image-dir` and `--cal-dir` to
specify where to get the frames from.
```bash
python Run.py --video --device=cpu [--hide-debug]
```


## How it works
The PyTorch neural net takes in images of size 224x224 and predicts the orientation and
relative dimension of that object to the class average. Thus, another neural net must give
the 2D bounding box and object class. I chose to use YOLOv3 through OpenCV.
Using the orientation, dimension, and 2D bounding box, the 3D location is calculated, and then
back projected onto the image.

There are 2 key assumptions made:
1. The 2D bounding box fits very tightly around the object;
2. The object has ~0 pitch and ~0 roll (valid for cars on the road).
