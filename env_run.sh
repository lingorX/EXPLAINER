#!/bin/bash

# uncompress the packed enviroment into the local dir
function prepare_env() {
  tar -xf cellp.tar
  
  export PATH="$(pwd)/cellp/bin:$PATH"
  export LD_LIBRARY_PATH="$(pwd)/cellp/lib:$LD_LIBRARY_PATH"
  
  apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
}


# unify ui
prepare_env


# This is a Demo script which only train for one epoch
./tools/dist_train.sh configs/_pattern_/resnet50_EXPLAINER_4xb32_cifar100_Demo.py 1


