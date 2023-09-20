# Hierarchical Neural Memory Network
# 
# Copyright (C) 2023 National Institute of Advanced Industrial Science and Technology
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of {{ project }} nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# sh ./scripts/run_eval.sh ./config/hmnet_B3_yolox_tbptt.py pretrained
# sh ./scripts/run_eval.sh ./config/hmnet_B3_yolox_regular_batch_relative.py 3
# sh ./scripts/run_eval.sh ./config/hmnet_B3_yolox_regular_batch_absolute.py 3

if [ $# -le 0 ];then
    echo "Usage: $0 [1]"
    echo "    [1]: config file"
    exit
fi

NAME=${1##*/}
NAME=${NAME%.py}

test_dt_dir=$(ls -d ./workspace/${NAME}/result/pred_test_$2)
log_out=${test_dt_dir}/logs
mkdir -p ${log_out}


if [ -d "/home/tkyen/opencv_practice/data_1/Gen4_Automotive/HMNet" ]
then
    Gen4_Automotive_HMNet_dir="/home/tkyen/opencv_practice/data_1/Gen4_Automotive/HMNet"
elif [ -d "/tmp2/tkyen/Gen1_Automotive/HMNet" ]
then
    Gen4_Automotive_HMNet_dir="/tmp2/tkyen/Gen1_Automotive/HMNet"
elif [ -d "/tmp3/tkyen/Gen1_Automotive/HMNet" ]
then
    Gen4_Automotive_HMNet_dir="/tmp3/tkyen/Gen1_Automotive/HMNet"
else
    Gen4_Automotive_HMNet_dir=""
fi

if [ -z "$Gen4_Automotive_HMNet_dir" ]
then
    echo "\$Gen4_Automotive_HMNet_dir is NULL"
    exit
fi
test_gt_dir="$Gen4_Automotive_HMNet_dir/test_lbl"

python ./scripts/psee_evaluator.py \
        ${test_gt_dir} \
        ${test_dt_dir} \
        --camera GEN1 > ${log_out}/result.txt \
        --discard_small_obj
        # --event_folder /tmp2/tkyen/Gen1_Automotive/detection_dataset_duration_60s_ratio_1.0/test \
        
