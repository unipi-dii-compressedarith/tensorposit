bazel build //tensorflow/tools/pip_package:build_pip_package \
    --config=rbe_cpu_linux \
    --remote_executor=cloud.buildbuddy.io \
    --bes_backend=cloud.buildbuddy.io \
    --bes_results_url=https://app.buildbuddy.io/invocation/ \
    --config=monolithic \
    --verbose_failures \
    --jobs=200 \
    --nogoogle_default_credentials \
    --remote_header=x-buildbuddy-api-key=PCiJWYpP8gN8QCivcMsC