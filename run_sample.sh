# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:25:53 on Sun, May 29, 2022
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH
export LD_LIBRARY_PATH=$WORK_PATH/output/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu

rm -rf log && mkdir -p log/sample/cuda log/sample/nvml

# cuda/cudart
nohup $WORK_PATH/output/sample/cuda/bandwidth_test > log/sample/cuda/bandwidth_test.log 2>&1 &
nohup $WORK_PATH/output/sample/cuda/matrix_mul > log/sample/cuda/matrix_mul.log 2>&1 &
nohup $WORK_PATH/output/sample/cuda/vector_add > log/sample/cuda/vector_add.log 2>&1 &

# nvml
nohup $WORK_PATH/output/sample/nvml/nvml_example > log/sample/nvml/nvml_example.log 2>&1 &
nohup $WORK_PATH/output/sample/nvml/supported_vgpus > log/sample/nvml/supported_vgpus.log 2>&1 &
