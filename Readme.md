<h1 align="center">GPURepair Documentation</h1>

<h1>Table of Contents</h1>

- [A. Introduction](#a-introduction)
- [B. Developer Guide](#b-developer-guide)
    + [Building from Source on Ubuntu 18.04](#building-from-source-on-ubuntu-1804)
- [C. Getting Started with GPURepair](#c-getting-started-with-gpurepair)

# A. Introduction

GPURepair has been accepted in the 22nd International Conference on Verification, Model Checking, and Abstract Interpretation (VMCAI 2021) under the tool paper category. An extended version of the paper is available on arXiv at <a href="https://arxiv.org/abs/2011.08373">https://arxiv.org/abs/2011.08373</a>. The documentation at <a href="./docs/vmcai2021/Documentation.md">docs/vmcai2021/Documentation.md</a> provides a detailed guide on how to set up GPURepair (the artifact is available <a href="">here</a>) on the Virtual Machine provided for VMCAI 2021 artifact evaluation (available <a href="https://zenodo.org/record/4017293/#%5C.X4c%5C_mtAzaUk">here</a>).

If you are using our tool or the technique in your work, please cite our paper at <a href="https://link.springer.com/chapter/10.1007/978-3-030-67067-2_18">GPURepair: Automated Repair of GPU Kernels</a> <a href="#3">[3]</a>.

# B. Developer Guide

### Building from Source on Ubuntu 18.04

1. Install [Python 3](https://www.python.org/) and the [psutil](https://pypi.org/project/psutil/) module

```bash
sudo apt-get update
sudo apt install python3
sudo apt install python3-pip

pip3 install psutil
```

2. Install [Mono](https://www.mono-project.com/). Latest instructions can be found at [https://www.mono-project.com/download/stable/#download-lin](https://www.mono-project.com/download/stable/#download-lin)

```bash
sudo apt install gnupg ca-certificates
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
echo "deb https://download.mono-project.com/repo/ubuntu stable-bionic main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
sudo apt update

sudo apt install mono-devel
```

3. Install the [Git](https://git-scm.com/) client to download the source code and install [cmake](https://cmake.org/) for building the tools from source

```bash
sudo apt install git
sudo apt install cmake
```

4. Download the source code of LLVM and Clang from the [llvm-project 6.x](https://github.com/llvm/llvm-project/tree/release/6.x) repository and compile it

```bash
export BUILD_ROOT=/path/to/build

# download the source code
export LLVM_RELEASE=release/6.x
mkdir -p ${BUILD_ROOT}/llvm_and_clang
cd ${BUILD_ROOT}/llvm_and_clang
git clone https://github.com/llvm/llvm-project.git src/6.x

cd src/6.x
git checkout ${LLVM_RELEASE}
cp -r ${BUILD_ROOT}/llvm_and_clang/src/6.x/clang ${BUILD_ROOT}/llvm_and_clang/src/6.x/llvm/tools/clang

# generate the build files
mkdir -p ${BUILD_ROOT}/llvm_and_clang/build
cd ${BUILD_ROOT}/llvm_and_clang/build
cmake -D CMAKE_BUILD_TYPE=Release -D LLVM_TARGETS_TO_BUILD=NVPTX ../src/6.x/llvm

# start the build (recommended value of N is the number of cores)
make -jN
```

5. Download the source code of libclc from the [llvm-project 8.x](https://github.com/llvm/llvm-project/tree/release/8.x) repository and compile it

```bash
# copy the libclc source code into a different folder
mkdir -p ${BUILD_ROOT}/libclc
cd ${BUILD_ROOT}/libclc
cp -r ${BUILD_ROOT}/llvm_and_clang/src/libclc ${BUILD_ROOT}/libclc/src

# build the source code
cd ${BUILD_ROOT}/libclc/src
python3 ./configure.py --with-llvm-config=${BUILD_ROOT}/llvm_and_clang/build/bin/llvm-config \
	--with-cxx-compiler=c++ \
	--prefix=${BUILD_ROOT}/libclc/install \
	nvptx-- nvptx64--
make

make install# download the source code
cd ${BUILD_ROOT}/llvm_and_clang
git clone https://github.com/llvm/llvm-project.git src/8.x

cd src/8.x
git checkout release/8.x

# copy the libclc source code into a different folder
mkdir -p ${BUILD_ROOT}/libclc
cd ${BUILD_ROOT}/libclc
cp -r ${BUILD_ROOT}/llvm_and_clang/src/8.x/libclc ${BUILD_ROOT}/libclc/src

# build the source code
cd ${BUILD_ROOT}/libclc/src
python3 ./configure.py --with-llvm-config=${BUILD_ROOT}/llvm_and_clang/build/bin/llvm-config \
	--with-cxx-compiler=c++ \
	--prefix=${BUILD_ROOT}/libclc/install \
	nvptx-- nvptx64--
make

make install
```

6. Download the source code of the [Z3 SMT Solver](https://github.com/Z3Prover/z3/tree/z3-4.6.0) and compile it

```bash
# download the source code
export Z3_RELEASE=z3-4.6.0
cd ${BUILD_ROOT}
git clone https://github.com/Z3Prover/z3.git

# generate the build files
cd ${BUILD_ROOT}/z3
git checkout -b ${Z3_RELEASE} ${Z3_RELEASE}
python3 scripts/mk_make.py

# start the build (recommended value of N is the number of cores)
cd build
make -jN

# install the library and create a symbolic link
sudo make install
ln -s z3 z3.exe
```

7. Download the source code of the [Bugle](https://github.com/cs17resch01003/bugle) and compile it. This repository is a fork of [https://github.com/mc-imperial/bugle](https://github.com/mc-imperial/bugle) with additional code to support the [CUDA Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)

```bash
# download the source code
cd ${BUILD_ROOT}
git clone https://github.com/cs17resch01003/bugle.git ${BUILD_ROOT}/bugle/src

# generate the build files
mkdir ${BUILD_ROOT}/bugle/build
cd ${BUILD_ROOT}/bugle/build
cmake -D LLVM_CONFIG_EXECUTABLE=${BUILD_ROOT}/llvm_and_clang/build/bin/llvm-config \
	-D CMAKE_BUILD_TYPE=Release \
	-D LIBCLC_DIR=${BUILD_ROOT}/libclc/install \
	../src

# start the build (recommended value of N is the number of cores)
make -jN
```

8. Download the source code of the [GPUVerify](https://github.com/cs17resch01003/gpuverify) and compile it. This repository is a fork of [https://github.com/mc-imperial/gpuverify](https://github.com/mc-imperial/gpuverify) with additional code to support the [CUDA Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)

```bash
# download the source code
cd ${BUILD_ROOT}
git clone https://github.com/cs17resch01003/gpuverify.git

# compile the code
cd ${BUILD_ROOT}/gpuverify
msbuild /p:Configuration=Release GPUVerify.sln

# copy the config file and change the "rootDir" variable to ${BUILD_ROOT}
cd ${BUILD_ROOT}/gpuverify
cp gvfindtools.templates/gvfindtools.dev.py gvfindtools.py
vim gvfindtools.py

# test the build
cd ${BUILD_ROOT}/gpuverify
python3 ./gvtester.py --write-pickle run.pickle testsuite
```

9. Download the source code of the [GPURepair](https://github.com/cs17resch01003/gpurepair) and compile it

```bash
# download the source code
cd ${BUILD_ROOT}
git clone https://github.com/cs17resch01003/gpurepair.git

# compile the code
cd ${BUILD_ROOT}/gpurepair/src
msbuild /p:Configuration=Release /p:Platform=x86 GPURepair.sln
msbuild /p:Configuration=Release /p:Platform=x86 GPURepair.ReportGenerator.sln

# edit the config file and change the "rootDir" variable to ${BUILD_ROOT}
cd ${BUILD_ROOT}/gpurepair/src/Toolchain
vim gvfindtools.py

# test the build
cd ${BUILD_ROOT}/gpurepair/src/Toolchain
python3 ./grtester.py ../../tests/testsuite
```

# C. Getting Started with GPURepair

GPURepair follows the same execution syntax as GPUVerify. Below is a CUDA kernel that has a data race. This kernel is located at *${BUILD_ROOT}/gpurepair/tests/testsuite/Bugged/race/kernel.cu*

```cpp
#include <cuda.h>
__global__ void race (int* A)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = A[idx + 1];
    A[idx] = temp;
}
```

To repair this kernel, run the following command.

```bash
cd ${BUILD_ROOT}/gpurepair/src/Toolchain
python3 ./GPURepair.py ../../tests/testsuite/Bugged/race/kernel.cu --blockDim=32 --gridDim=1
```

This command repairs the kernel and prints the number of changes that are required to correct this kernel. The changes that are needed are written to a summary file at the same location where the kernel is.

```bash
cat ../../tests/testsuite/Bugged/race/kernel.summary
# prints the line number where the barrier needs to be inserted to fix the data race
```

The changes in the summary file specify the file and line where a barrier needs to be added or removed. The fixed program is also available in its Boogie Intermediate Representation (IR) at the same location with the extension **.fixed.cbpl**

The repaired kernel will look like below.

```cpp
#include <cuda.h>
__global__ void race (int* A)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = A[idx + 1];
    __syncthreads();
    A[idx] = temp;
}
```

All the command-line options available for GPUVerify at [https://github.com/mc-imperial/gpuverify/blob/master/Documentation/basic_usage.rst](https://github.com/mc-imperial/gpuverify/blob/master/Documentation/basic_usage.rst) can be used with GPURepair as well. GPURepair passes these command-line options, if specified during its invocation, to GPUVerify for the verification process.

Besides the verification options provided by GPUVerify, GPURepair provides command-line options to change the solver type used in the repair process, disable instrumentation of grid-level barriers and inspection of programmer inserted barriers. It also provides additional logging options that log several metrics during the repair process, including the clauses that were generated. A reference of these options can be accessed using the command

```bash
python3 ./GPURepair.py --help
```

By default, GPURepair uses the mhs solver, enables the instrumentation of grid-level barriers and inspection of pre-existing barriers. The other command-line options available are

| Option | Description |
| :------------- | :------------- |
| --detailed-logging | Enables detailed logging for instrumentation and repair |
| --log-clauses | Enables logging of clauses during the repair process |
| --disable-inspect | Disables inspection of programmer inserted barriers |
| --disable-grid | Disables grid-level barriers during instrumentation |
| --use-axioms | Use axioms for instrumentation instead of variable assignments |
| --loop-depth-weight | The weight of a barrier when it is inside a loop. This value will be raised exponentially based on the loop depth |
| --mhs | Use the mhs solver in the repair process |
| --maxsat | Use the MaxSAT solver in the repair process |


# Citation
If you use GPURepair in any fashion, we would appreciate if you cite the following corresponding paper:


 * Saurabh Joshi and Gautam Muduganti, _"GPURepair: Automated Repair of GPU Kernels"_.
   In VMCAI 2021, pages 401-414. Springer, 2021.
 
 You may download the BibTeX file from here: [BibTex](https://citation-needed.springer.com/v2/references/10.1007/978-3-030-67067-2_18?format=bibtex&flavour=citation)




# References

<a id="1">[1]</a>
Sourav Anand and Nadia Polikarpova.
Automatic Synchronization for GPU Kernels.
In FMCAD 2018, pages 1-9. IEEE, 2018.

<a id="2">[2]</a>
Adam Betts, Nathan Chong, Alastair F. Donaldson, Shaz Qadeer and Paul Thomson.
GPUVerify: A Verifier for GPU Kernels.
In OOPSLA 2012, pages 113-132. ACM, 2012.

<a id="3">[3]</a>
Saurabh Joshi and Gautam Muduganti.
GPURepair: Automated Repair of GPU Kernels.
In VMCAI 2021, pages 401-414. Springer, 2021.
