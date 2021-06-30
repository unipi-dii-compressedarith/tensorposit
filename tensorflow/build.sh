#!bin/sh
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt

