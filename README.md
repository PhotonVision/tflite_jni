# Shared Libraries

Last build of tensorflow libraries was on version 2.19.0
To build the necessary tensorflow libraries, the following commands can be used

```
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/delegates/external:external_delegate
```

# Building On a Rubik Pi

```
sudo apt-get update
sudo apt-get install -y build-essential cmake openjdk-17-jdk default-jdk
git clone https://github.com/PhotonVision/tflite_jni
cmake -B cmake_build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=cmake_build -DOPENCV_ARCH=linuxarm64 ; cmake --build cmake_build --target install -- -j 4
# Sudo is needed because of some weird permission issues with gradle and native libraries
sudo ./gradlew build -x spotlessCheck
```

## MemLeak test

Build and run repeated iterations of creating and destroying a detector to find memory leaks. It's necessary to watch memory by hand, as the test only runs repeated create and destroy cycles, it doesn't monitor memory.

```
./gradlew build -PmemLeakTestIterations=1000
```

## Benchmark

Run the detect function repeatedly to benchmark performance.

```
./gradlew build -PbenchmarkIterations=1000
```
