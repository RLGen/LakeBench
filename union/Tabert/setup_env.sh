#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "----------------- Check Env -----------------"
nvidia-smi
nvcc -V
python -V
python -c 'import torch; print(torch.__version__)'

echo "----------------- Check File System -----------------"
echo "I am " $(whoami)
echo -n "CURRENT_DIRECTORY"
pwd

echo "---------------- Update Conda Environment ---------------"
echo "Updating conda environment"
conda env create --file scripts/env.yml
# or use `conda env update to update the current environment`
conda activate tabert

PWD_DIR=$(pwd)

echo "----------------- Install Apex -----------------"
mkdir -p third_party
git clone -q https://github.com/NVIDIA/apex.git third_party/apex
cd third_party/apex
# per https://github.com/NVIDIA/apex/issues/605
# remember to set to the correct target!
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" ./

cd ${PWD_DIR}