""" This module defines the paths that GPUVerify will use
    to run the various tools that GPUVerify Depends on.

    These paths must be absolute paths.
"""
import os
import sys

# THIS IS A TEMPLATE FOR DEVELOPMENT. MODIFY THE PATHS TO SUIT YOUR BUILD
# ENVIRONMENT. THEN COPY THIS FILE INTO THE ROOT GPUVERIFY DIRECTORY (where
# GPUVerify.py lives) AND RENAME IT TO "gvfindtools.py". "gvfindtools.py" WILL
# BE IGNORED BY MERCURIAL SO IT WILL NOT BE UNDER VERSION CONTROL SO THAT YOU
# CAN MAINTAIN YOUR OWN PERSONAL COPY OF "gvfindtools.py" WITHOUT AFFECTING
# OTHER DEVELOPERS.
#
# Please note Windows users should use the following style:
# rootDir = r"c:\projects\gpuverify"
# bugleSrcDir = rootDir + r"\bugle\src"

rootDir = r"C:\SRC\GPURepair"

# The path to the Bugle Source directory.
# The include-blang/ folder should be there
bugleSrcDir = rootDir + r"\bugle\src"

# The Path to the directory where the "bugle" executable can be found.
bugleBinDir = rootDir + r"\bugle\build\Release"

# The path to the libclc Source directory.
libclcSrcDir = rootDir + r"\libclc\src"

# The path to the libclc install directory.
# The include/ and lib/clc/ folders should be there
libclcInstallDir = rootDir + r"\libclc\install"

# The path to the llvm Source directory.
llvmSrcDir = rootDir + r"\llvm_and_clang\src"

# The path to the directory containing the llvm binaries.
# llvm-nm, clang and opt should be there
llvmBinDir = rootDir + r"\llvm_and_clang\build\Release\bin"

# The path containing the llvm libraries
llvmLibDir = rootDir + r"\llvm_and_clang\build\Release\lib"

# The path to the directory containing the GPUVerify binaries.
# GPUVerifyVCGen.exe, GPUVerifyCruncher.exe and GPUVerifyBoogieDriver.exe should be there
gpuVerifyBinDir = rootDir + r"\gpuverify\Binaries"

# The path to the directory containing the GPURepair binaries.
# GPURepair.Instrumentation.exe and GPURepair.Repair.exe should be there
gpuRepairBinDir = rootDir + r"\GPURepair\Binaries"

# The path to the z3 Source directory.
z3SrcDir = rootDir + r"\z3"

# The path to the directory containing z3.exe
z3BinDir = rootDir + r"\z3\build"

# The path to the cvc4 Source directory.
cvc4SrcDir = rootDir + r"\CVC4\src"

# The path to the directory containing cvc4.exe
cvc4BinDir = rootDir + r"\CVC4\install\bin"

# Default solver should be one of ['z3','cvc4']
defaultSolver = 'z3'

# If true mono will prepended to every command involving CIL executables
useMono = True if os.name == 'posix' else False

def init(prefixPath):
  """This method does nothing"""
  pass
