bazel build //tensorflow/tools/pip_package:build_pip_package \
    --config=rbe_cpu_linux \
    --remote_executor=cloud.buildbuddy.io \
    --bes_backend=cloud.buildbuddy.io \
    --bes_results_url=https://app.buildbuddy.io/invocation/ \
    --config=monolithic \
    --verbose_failures \
    --jobs=200 \
    --nogoogle_default_credentials \
    --remote_header=x-buildbuddy-api-key=$BUILD_BUDDY_API_KEY


./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt
pip install /mnt/tensorflow-2.6.0-cp36-cp36m-linux_x86_64.whl  
pip uninstall keras-nightly
