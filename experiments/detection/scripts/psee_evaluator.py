# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file is modified from the original code at
# https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/psee_evaluator.py
# The list of modifications are as follows:
# (1) "min_box_side" for box filtering is modified following the previous work:
#     Perot, Etienne, et al. "Learning to detect objects with a 1 megapixel event camera." Advances in Neural Information Processing Systems 33 (2020): 16639-16652.
# (2) Configs for GEN1 and GEN4 are added and passed to "evaluate_detection"

import glob
import numpy as np
import os
import argparse
import pickle as pkl
from numpy.lib import recfunctions as rfn

from coco_eval import evaluate_detection, evaluate_detection_RED
from hmnet.utils.psee_toolbox.io.box_filtering import filter_boxes
from hmnet.utils.psee_toolbox.io.box_loading import reformat_boxes
from hmnet.utils.common import get_list, mkdir

EVAL_CONF_GEN1 = dict(
    classes = ('car', 'pedestrian'),
    width = 304,
    height = 240,
    time_tol = 25000, # +/- 25 msec (50 msec)
)

EVAL_CONF_GEN4 = dict(
    classes = ('pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light'),
    width = 1280,
    height = 720,
    time_tol = 25000, # +/- 25 msec (50 msec)
)

# EVAL_CONF_GEN4 = dict(
#     classes = ('background', 'pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light'),
#     width = 1280//2,
#     height = 720//2,
#     time_tol = 25000, # +/- 25 msec (50 msec)
# )

def evaluate_folders(dt_folder, gt_lst, discard_small_obj, event_folder, camera):
    dt_file_paths = get_list(dt_folder, ext='npy')
    gt_file_paths = get_list(gt_lst, ext='npy')
    assert len(dt_file_paths) == len(gt_file_paths)
    print("There are {} GT bboxes and {} PRED bboxes".format(len(gt_file_paths), len(dt_file_paths)))
    npy_file_list = list()
    for dt_file_path, gt_file_path in zip(dt_file_paths, gt_file_paths):
        npy_file_list.append(os.path.basename(gt_file_path))
        assert os.path.basename(dt_file_path)==os.path.basename(gt_file_path)

    result_boxes_list = [np.load(p) for p in dt_file_paths]
    result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
    # result_boxes_list[0].dtype.names = ('t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence')

    gt_boxes_list = [np.load(p) for p in gt_file_paths]
    # gt_boxes_list[0].dtype.names     = ('t', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id', 'invalid')
    if 'invalid' in gt_boxes_list[0].dtype.names:
        for i, gt_boxes in enumerate(gt_boxes_list):
            invalids = gt_boxes['invalid']
            if np.sum(invalids) > 0:
                gt_boxes = gt_boxes[np.logical_not(invalids)]
            gt_boxes = rfn.drop_fields(gt_boxes, 'invalid')
            gt_boxes_list[i] = gt_boxes
    gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]
    # gt_boxes_list[0].dtype.names     = ('t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence')

    eval_conf = EVAL_CONF_GEN4 if camera == 'GEN4' else EVAL_CONF_GEN1
    if discard_small_obj:
        min_box_diag = 60 if camera == 'GEN4' else 30
        min_box_side = 20 if camera == 'GEN4' else 10
        filter_boxes_fn = lambda x:filter_boxes(x, int(5e5), min_box_diag, min_box_side)
        gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
        result_boxes_list = map(filter_boxes_fn, result_boxes_list)
    # evaluate_detection(gt_boxes_list, result_boxes_list, npy_file_list, dt_folder, event_folder, **eval_conf)
    evaluate_detection_RED(gt_boxes_list, result_boxes_list, npy_file_list, dt_folder, event_folder, **eval_conf)

def main():
    parser = argparse.ArgumentParser(prog='psee_evaluator.py')
    parser.add_argument('gt_lst', type=str, help='Text file contaiing list of GT .npy files')
    parser.add_argument('dt_folder', type=str, help='RESULT folder containing .npy files')
    parser.add_argument('--discard_small_obj', action='store_true', default=False)
    parser.add_argument('--event_folder', type=str, help='Event folder containing .dat files')
    parser.add_argument('--camera', type=str, default='GEN4', help='GEN1 (QVGA) or GEN4 (720p)')
    opt = parser.parse_args()
    evaluate_folders(opt.dt_folder, opt.gt_lst, opt.discard_small_obj, opt.event_folder, opt.camera)

if __name__ == '__main__':
    '''
    python ./scripts/psee_evaluator.py \
      /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet/test_lbl \
      ./workspace/hmnet_B3_yolox_tbptt/result/pred_test_pretrained \
      --event_folder /home/tkyen/opencv_practice/data_1/Gen1_Automotive/detection_dataset_duration_60s_ratio_1.0/test \
      --camera GEN1

    python ./scripts/psee_evaluator.py \
      /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet/test_lbl \
      ./workspace/hmnet_B3_yolox_regular_batch_relative/result/pred_test_10 \
      --event_folder /home/tkyen/opencv_practice/data_1/Gen1_Automotive/detection_dataset_duration_60s_ratio_1.0/test \
      --camera GEN1

    python ./scripts/psee_evaluator.py \
      /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet/test_lbl \
      ./workspace/hmnet_B3_yolox_regular_batch_absolute/result/pred_test_10 \
      --event_folder /home/tkyen/opencv_practice/data_1/Gen1_Automotive/detection_dataset_duration_60s_ratio_1.0/test \
      --camera GEN1

    python ./scripts/psee_evaluator.py \
      /home/tkyen/opencv_practice/data_3/Gen4_Automotive_event_cube_paper/result_vanilla_ssd_level_5/evaluation_epoch_31/gt \
      /home/tkyen/opencv_practice/data_3/Gen4_Automotive_event_cube_paper/result_vanilla_ssd_level_5/evaluation_epoch_31/dt \
      --camera GEN4
    '''
    main()
