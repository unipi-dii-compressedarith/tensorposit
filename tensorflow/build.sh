#!bin/sh
bazel build --jobs=2 --local_ram_resources=4000 --local_cpu_resources=4 //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt

