# Building On a Rubik Pi

```
sudo apt-get update
sudo apt-get install -y build-essential cmake openjdk-25-jdk default-jdk
git clone https://github.com/PhotonVision/rubik_jni
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
