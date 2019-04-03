cc_library(
    name = "hellotime",
    srcs = ["hellotime.cc"],
    hdrs = ["hellotime.h"],
)

cc_binary(
    name = "helloworld",
    srcs = ["helloworld.cc"],
    deps = [
        ":hellotime",
        "//lib:hellogreet",
    ],
)