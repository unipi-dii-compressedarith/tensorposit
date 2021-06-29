# load(
#     "//tensorflow/core/platform:rules_cc.bzl",
#     "cc_library",
# )

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

# cc_library(
#     name = "posit",
#     hdrs = ["posit.h"],
#     deps = [
#          "//third_party/cppposit_private:posit"
#         "//tensorflow/core/platform:byte_order",
#     ],
# )

# # Export source files needed for mobile builds, which do not use granular targets.
# filegroup(
#     name = "mobile_srcs_no_runtime",
#     srcs = [
#         "posit.h",
#     ],
# )

exports_files(
    ["posit.h"]
)
