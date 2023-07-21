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

while getopts ah OPT
do
    case $OPT in
        "a" ) FLAG_M=TRUE;;
        "h" ) FLAG_H=TRUE;;
          * ) echo "Usage: $CMDNAME [-a] [-h]" 1>&2
              exit 1;;
    esac
done

if [ "$FLAG_H" = "TRUE" ];then
    echo "Usage: ${0##*/} [-m] [-h]"
    echo "    -m  : Generate meta data. This may take long time."
    echo "    -h  : Show this message"
    echo ""
    exit 0
fi


echo "=========================================="
echo " Start preprocessing"
echo "=========================================="
Gen1_Automotive_dir="/home/tkyen/opencv_practice/data/Gen1_Automotive"
train_dir="$Gen1_Automotive_dir/detection_dataset_duration_60s_ratio_1.0/train"
val_dir="$Gen1_Automotive_dir/detection_dataset_duration_60s_ratio_1.0/val"
test_dir="$Gen1_Automotive_dir/detection_dataset_duration_60s_ratio_1.0/test"

Gen1_Automotive_HMNet_dir="/home/tkyen/opencv_practice/data/Gen1_Automotive/HMNet"
train_evt_dir="$Gen1_Automotive_HMNet_dir/train_evt"
val_evt_dir="$Gen1_Automotive_HMNet_dir/val_evt"
test_evt_dir="$Gen1_Automotive_HMNet_dir/test_evt"

train_lbl_dir="$Gen1_Automotive_HMNet_dir/train_lbl"
val_lbl_dir="$Gen1_Automotive_HMNet_dir/val_lbl"
test_lbl_dir="$Gen1_Automotive_HMNet_dir/test_lbl"

mkdir -p $train_evt_dir
mkdir -p $val_evt_dir
mkdir -p $test_evt_dir

mkdir -p $train_lbl_dir
mkdir -p $val_lbl_dir
mkdir -p $test_lbl_dir

python ./scripts/preproc_events.py $train_dir $train_evt_dir
python ./scripts/preproc_events.py $val_dir $val_evt_dir
python ./scripts/preproc_events.py $test_dir $test_evt_dir

python ./scripts/modify_lbl_field_name.py $train_dir $train_lbl_dir
python ./scripts/modify_lbl_field_name.py $val_dir $val_lbl_dir
python ./scripts/modify_lbl_field_name.py $test_dir $test_lbl_dir

python ./scripts/validate_bbox.py $train_lbl_dir $train_lbl_dir
python ./scripts/validate_bbox.py $val_lbl_dir $val_lbl_dir
python ./scripts/validate_bbox.py $test_lbl_dir $test_lbl_dir

if [ "$FLAG_M" = "TRUE" ];then
    echo "=========================================="
    echo " Generating meta data"
    echo "=========================================="
    train_list_dir="$Gen1_Automotive_HMNet_dir/list/train"
    val_list_dir="$Gen1_Automotive_HMNet_dir/list/val"
    test_list_dir="$Gen1_Automotive_HMNet_dir/list/test"

    mkdir -p $train_list_dir
    mkdir -p $val_list_dir
    mkdir -p $test_list_dir

    ls $train_evt_dir/*.npy > $train_list_dir/events.txt
    ls $train_lbl_dir/*.npy > $train_list_dir/labels.txt
    ls $val_evt_dir/*.npy > $val_list_dir/events.txt
    ls $val_lbl_dir/*.npy > $val_list_dir/labels.txt
    ls $test_evt_dir/*.npy > $test_list_dir/events.txt
    ls $test_lbl_dir/*.npy > $test_list_dir/labels.txt

    train_meta_dir="$Gen1_Automotive_HMNet_dir/train_meta"
    val_meta_dir="$Gen1_Automotive_HMNet_dir/val_meta"
    test_meta_dir="$Gen1_Automotive_HMNet_dir/test_meta"

    python ./scripts/make_event_meta.py $train_evt_dir $train_meta_dir
    python ./scripts/make_event_meta.py $val_evt_dir $val_meta_dir
    python ./scripts/make_event_meta.py $test_evt_dir $test_meta_dir

    python ./scripts/merge_meta.py $train_meta_dir $train_list_dir
    python ./scripts/merge_meta.py $val_meta_dir $val_list_dir
    python ./scripts/merge_meta.py $test_meta_dir $test_list_dir

    python ./scripts/get_gt_interval.py $train_lbl_dir $train_list_dir
    python ./scripts/get_gt_interval.py $val_lbl_dir $val_list_dir
    python ./scripts/get_gt_interval.py $test_lbl_dir $test_list_dir
fi
