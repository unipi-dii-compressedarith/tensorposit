#!/bin/bash
docker run --rm -it  -e BUILD_BUDDY_API_KEY=$BUILD_BUDDY_API_KEY -v/home/federico/cppWorkspace/cppposit_private:/cppposit_private -v $PWD/tensorflow:/tensorposit -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:devel bash




