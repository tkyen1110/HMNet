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

import os
import numpy as np
import random
import collections
import torch
import torchvision
import pandas as pd
import math
import json
import copy
import pickle as pkl
import numpy.lib.recfunctions as rfn
from torch.utils import data

from hmnet.utils.common import get_list
from hmnet.models.base.event_repr.builder import build_eventrepr

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

IGNORE_LABEL_IDX = 255    # must be uint8
DUMMY_LABEL_IDX = -2
WIDTH = 304
HEIGHT = 240

PSEE_ID2NAMES = {
    0 : 'car',
    1 : 'pedestrian',
}

# HMNet_dataset = '/home/tkyen/opencv_practice/data/Gen1_Automotive/HMNet'
# train_dataset = EventPacketStream(
#     fpath_evt_lst      = os.path.join(HMNet_dataset, 'list/train/events.txt'),
#     fpath_lbl_lst      = os.path.join(HMNet_dataset, 'list/train/labels.txt'),
#     base_path          = '',
#     fpath_meta         = os.path.join(HMNet_dataset, 'list/train/meta.pkl'),
#     fpath_gt_duration  = os.path.join(HMNet_dataset, 'list/train/gt_interval.csv'),
#     video_duration     = 60e6,
#     train_duration     = TRAIN_DURATION, # 200e3
#     delta_t            = DELTA_T, # 5e3
#     skip_ts            = 0,
#     use_nearest_label  = False,
#     sampling           = 'label',
#     min_box_diag       = 30,
#     min_box_side       = 10,
#     random_time_scaling = False,
#     start_index_aug_method = 'end',
#     start_index_aug_ratio = 0.25,
#     event_transform    = train_transform,
# )
class EventPacket(data.Dataset):
    def __init__(self, fpath_evt_lst=None, fpath_lbl_lst=None, base_path='', fpath_meta=None, fpath_gt_duration=None, video_duration=6e7, train_duration=5e4,
                 fpath_sampling=None, sampling='random', start_index_aug_method='center', start_index_aug_ratio=1., sampling_stride=-1,
                 min_box_diag=30, min_box_side=10,
                 random_time_scaling=False, min_time_scale=0.5, max_time_scale=2.0,
                 event_transform=None, output_type=None,
                 max_events_per_packet=-1, downsample_packet_length=None, batch_size=1):

        assert sampling in ('random', 'file', 'label', 'regular', 'regular_batch')

        self.base_path = base_path                           # ''
        self.sampling = sampling                             # 'label'
        self.ev_meta = pkl.load(open(fpath_meta, 'rb'))
        self.fpath_gt_duration = fpath_gt_duration
        self.event_transform = event_transform
        self.output_type = output_type                       # None
        self.max_events_per_packet = max_events_per_packet   # -1
        self.downsample_packet_length = downsample_packet_length or train_duration # 200e3
        self.start_index_aug_method = start_index_aug_method # 'end'
        self.start_index_aug_ratio = start_index_aug_ratio   # 0.25
        self.video_duration = int(video_duration)            # 60e6  us
        self.train_duration = int(train_duration)            # 200e3 us
        self.random_time_scaling = random_time_scaling       # False
        self.min_time_scale = min_time_scale                 # 0.5
        self.max_time_scale = max_time_scale                 # 2.0
        self.min_box_diag = min_box_diag                     # 30
        self.min_box_side = min_box_side                     # 10

        self.list_fpath_evt = get_list(fpath_evt_lst, ext=None)
        self.list_fpath_lbl = get_list(fpath_lbl_lst, ext=None)
        assert len(self.list_fpath_evt) == len(self.list_fpath_lbl)

        if sampling == 'file':
            sampling_schedule = pkl.load(open(fpath_sampling, 'rb'))
            self.sampling_timings = sampling_schedule['timings']
            self.list_fpath_evt   = sampling_schedule['fpath_evt']
            self.list_fpath_lbl   = sampling_schedule['fpath_label']
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'label':
            self.sampling_timings = []
            for ifile, fname_lbl in enumerate(self.list_fpath_lbl):
                seg_indices = np.unique(np.load(self._get_path(fname_lbl))['t'] // 1000).tolist()
                self.sampling_timings += [ (ifile, seg_index) for seg_index in seg_indices ]
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'regular' or sampling == 'regular_batch':
            self.sampling_timings = []
            sampling_stride = sampling_stride if sampling_stride > 0 else train_duration
            seg_stride = int(sampling_stride // 1000)       # 200 ms
            seg_duration = int(self.video_duration // 1000) # 60000 ms
            for ifile in range(len(self.list_fpath_evt)):
                self.sampling_timings += [ (ifile, seg_index) for seg_index in range(0,seg_duration,seg_stride) ]

            if sampling == 'regular_batch' and len(self.list_fpath_evt) % batch_size != 0:
                append_num = batch_size - len(self.list_fpath_evt) % batch_size
                append_ifiles = np.random.randint(len(self.list_fpath_evt), size=append_num)
                for ifile in append_ifiles:
                    self.sampling_timings += [ (ifile, seg_index) for seg_index in range(0,seg_duration,seg_stride) ]
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'random':
            sampling_stride = sampling_stride if sampling_stride > 0 else train_duration
            self.total_seq = len(self.list_fpath_evt) * int(self.video_duration // sampling_stride)

        self._image_meta = {
            'width': WIDTH,
            'height': HEIGHT,
        }

    def _get_path(self, filename):
        if self.base_path == '':
            return filename
        if filename[:2] == './':
            filename = filename[2:]
        return self.base_path + '/' + filename

    def __len__(self):
        return self.total_seq

    def __getitem__(self, index):
        event_dict, bbox_dict, meta_data = self.getdata(index)
        bbox_dict = self._filter_dummy_label(bbox_dict)
        bbox_dict.pop('times')
        return event_dict, bbox_dict, meta_data

    def getdata(self, index, keep_latest_labels=True, skip_ts=0, time_method='relative_time'):
        if self.random_time_scaling:
            time_scaling = random.uniform(self.min_time_scale, self.max_time_scale)
        else:
            time_scaling = 1

        train_duration = int(self.train_duration * time_scaling)  # 200e3 us

        fpath_evt, fpath_lbl, base_time, ev_range, gt_duration = self._choose_file_and_time(index, train_duration)
        events, labels = self._load(fpath_evt, fpath_lbl, base_time, train_duration, ev_range, gt_duration)

        curr_time_org = base_time + train_duration
        curr_time_crop = train_duration
        image_meta = self._update_image_meta(self._image_meta, fpath_evt, curr_time_org, curr_time_crop, train_duration, time_scaling)

        labels = self._filter_invalid_bboxes(labels)

        if skip_ts > 0:
            labels = self._filter_early_bboxes(labels, base_time, skip_ts)

        events, labels = self._bind(events, labels, base_time, time_method)

        if self.max_events_per_packet > 0:
            events = self._event_downsample(events, self.max_events_per_packet, self.downsample_packet_length)

        event_dict = { 'events': torch.from_numpy(events) }
        bbox_dict = self._labels2bboxdict(labels)
        # bbox_dict = {
        #     'times': torch.from_numpy(gt_times),
        #     'bboxes': torch.from_numpy(gt_bboxes),
        #     'labels': torch.from_numpy(gt_labels),
        #     'ignore_mask': torch.from_numpy(ignore_mask).bool(),
        # }
        if self.event_transform is not None:
            event_dict, bbox_dict, image_meta = self.event_transform(event_dict, bbox_dict, image_meta, types=['event', 'bbox', 'meta']) # TODO: to read

        labels = self._bboxdict2labels(bbox_dict)
        labels = self._label_padding(labels, train_duration, gt_duration) # TODO: to read

        if keep_latest_labels:
            labels = self._keep_latest_labels(labels)
        bbox_dict = self._labels2bboxdict(labels)

        bbox_dict = self._filter_small_bboxes(bbox_dict, self.min_box_diag, self.min_box_side)

        meta_data = {
            'image_meta': image_meta,
            'label_meta': dict(ignore_index=IGNORE_LABEL_IDX),
        }

        if self.output_type is None:
            pass
        elif self.output_type == 'long':
            events = event_dict['events']
            events[:,-1] = ((events[:,-1] + 1) * 0.5)
            event_dict['events'] = events.long()
        elif self.output_type == 'float':
            event_dict['events'] = event_dict['events'].float()
        else:
            raise RuntimeError

        return event_dict, bbox_dict, meta_data


    def _load(self, fpath_evt, fpath_lbl, time, duration, ev_range, gt_duration):
        npy = np.load(fpath_evt, mmap_mode='r')
        st, ed = ev_range
        events = npy[st:ed]
        events = np.array(events)
        events['p'] = (events['p'] - 0.5) * 2

        labels = self._load_label_delta_t(fpath_lbl, time, duration+gt_duration) # TODO: Why duration+gt_duration?

        del npy

        return events, labels

    def _load_label_delta_t(self, fpath_lbl, time, delta_t):
        labels = np.load(fpath_lbl)
        labels = rfn.rename_fields(labels, {'ts': 't'})
        mask = (labels['t'] >= time) & (labels['t'] < time + delta_t)
        lbl = labels[mask]
        return lbl

    def _augment_index(self, seg_index, method, aug_ratio, nseg_per_packet, nseg_per_video):
        # method = 'end'
        # aug_ratio = 0.25
        # nseg_per_packet = 200
        # nseg_per_video = 60000
        whole_range = nseg_per_packet - 1
        aug_range = int(whole_range * aug_ratio)
        if method == 'none':
            return seg_index, True
        elif method == 'center':
            min_seg_idx = seg_index - int(whole_range * 0.5 + aug_range * 0.5)
            max_seg_idx = seg_index - int(whole_range * 0.5 - aug_range * 0.5)
        elif method == 'start':
            min_seg_idx = seg_index - aug_range
            max_seg_idx = seg_index
        elif method == 'end':
            min_seg_idx = seg_index - whole_range
            max_seg_idx = seg_index - whole_range + aug_range

        min_seg_idx = max(0, min_seg_idx)
        max_seg_idx = min(nseg_per_video - nseg_per_packet, max_seg_idx)

        if min_seg_idx > max_seg_idx:
            if min_seg_idx == 0:
                return 0, False
            else:
                return max_seg_idx, False

        seg_index = random.randint(min_seg_idx, max_seg_idx)
        return seg_index, True

    def _choose_file_and_time(self, index, train_duration):
        nseg_per_packet = int(train_duration / 1000) # 200
        nseg_per_video = int(self.video_duration / 1000) # 60000

        if self.sampling == 'random':
            ifile = random.randint(0, len(self.list_fpath_evt)-1)
            seg_index = random.randint(0, nseg_per_video - nseg_per_packet)
        elif self.sampling in ('file', 'label', 'regular', 'regular_batch'):
            ifile, seg_index = self.sampling_timings[index]
        # print(ifile, seg_index)
        seg_index, is_success = self._augment_index(seg_index, self.start_index_aug_method, self.start_index_aug_ratio, nseg_per_packet, nseg_per_video)
        if not is_success and self.sampling == 'file':
            # retry with other index
            print('Desired segment cannnot be cropped. Retrying with another index.')
            index = random.randrange(len(self.sampling_timings))
            return self._choose_file_and_time(index, train_duration)
        elif not is_success:
            print('Desired segment cannnot be cropped. Cropping nearest segment.')

        fpath_evt = self._get_path(self.list_fpath_evt[ifile])
        fpath_lbl = self._get_path(self.list_fpath_lbl[ifile])

        ev_index, ev_count = self.ev_meta[fpath_evt.split('/')[-1]]
        ev_i = ev_index[seg_index]
        ev_c = ev_count[seg_index: seg_index + nseg_per_packet].sum()
        ev_range = (ev_i, ev_i + ev_c)
        time = seg_index * 1000

        gt_durations = pd.read_csv(self.fpath_gt_duration, index_col=0)
        gt_duration = gt_durations.loc[fpath_lbl.split('/')[-1], 'interval']

        return fpath_evt, fpath_lbl, time, ev_range, gt_duration

    def _update_image_meta(self, image_meta, fpath_evt, curr_time_org, curr_time_crop, delta_t, time_scaling, init_states=False):
        image_meta = image_meta.copy()
        image_meta['filename'] = fpath_evt
        image_meta['ori_filename'] = fpath_evt.split('/')[-1]
        image_meta['curr_time_org'] = curr_time_org
        image_meta['curr_time_crop'] = curr_time_crop
        image_meta['delta_t'] = delta_t
        image_meta['stride_t'] = delta_t
        image_meta['time_scaling'] = time_scaling
        image_meta['init_states'] = init_states
        return image_meta

    def _keep_latest_labels(self, labels):
        if len(labels) == 0:
            return labels
        latest_time_lbl = labels[-1,0]
        return labels[labels[:,0]==latest_time_lbl]

    def _filter_invalid_bboxes(self, labels):
        if 'invalid' not in labels.dtype.fields:
            return labels
        mask = np.logical_not(labels['invalid'])
        labels = labels[mask]
        return labels

    def _filter_small_bboxes(self, bbox_dict, min_box_diag=60, min_box_side=20):
        bboxes = bbox_dict['bboxes']
        W = bboxes[:,2] - bboxes[:,0]
        H = bboxes[:,3] - bboxes[:,1]
        diag_square = W**2+H**2
        min_side = torch.minimum(W,H)
        mask = torch.logical_or(diag_square < min_box_diag**2, min_side < min_box_side).to(bool)
        bbox_dict['ignore_mask'] = mask
        return bbox_dict

    def _filter_early_bboxes(self, labels, base_time, skip_ts=0):
        times_lbl = labels['t'].astype(np.int) - base_time
        mask = (times_lbl > skip_ts)
        labels = labels[mask]
        return labels

    def _xywh2xyxy(self, labels):
        labels = labels.copy()
        labels[:,3] += labels[:,1]
        labels[:,4] += labels[:,2]
        return labels

    def _labels2bboxdict(self, labels):
        if len(labels) == 0:
            gt_times = np.empty([0], dtype=int)
            gt_bboxes = np.empty([0,4], dtype=int)
            gt_labels = np.empty([0], dtype=int)
            ignore_mask = np.empty([0], dtype=bool)
        else:
            gt_times = labels[:,0]
            gt_bboxes = labels[:,1:5]
            gt_labels = labels[:,5]
            if labels.shape[1] == 7:
                ignore_mask = labels[:,6]
            else:
                ignore_mask = np.zeros_like(gt_labels).astype(bool)

        bbox_dict = {
            'times': torch.from_numpy(gt_times),
            'bboxes': torch.from_numpy(gt_bboxes),
            'labels': torch.from_numpy(gt_labels),
            'ignore_mask': torch.from_numpy(ignore_mask).bool(),
        }

        return bbox_dict

    def _bboxdict2labels(self, bbox_dict):
        gt_times = bbox_dict['times']
        gt_bboxes = bbox_dict['bboxes']
        gt_labels = bbox_dict['labels']
        ignore_mask = bbox_dict['ignore_mask']
        labels = torch.cat([gt_times[:,None], gt_bboxes, gt_labels[:,None], ignore_mask[:,None]], dim=-1)
        return labels.numpy()

    def _filter_dummy_label(self, bbox_dict):
        if bbox_dict['labels'] is None:
            return bbox_dict
        mask = bbox_dict['labels'] != DUMMY_LABEL_IDX
        bbox_dict['times'] = bbox_dict['times'][mask]
        bbox_dict['bboxes'] = bbox_dict['bboxes'][mask]
        bbox_dict['labels'] = bbox_dict['labels'][mask]
        bbox_dict['ignore_mask'] = bbox_dict['ignore_mask'][mask]
        return bbox_dict

    def _bind(self, events, labels, base_time, time_method):
        t_evt, x_evt, y_evt, p_evt = events['t'], events['x'], events['y'], events['p']
        t_lbl, x_lbl, y_lbl, w_lbl, h_lbl, c_lbl = labels['t'], labels['x'], labels['y'], labels['w'], labels['h'], labels['class_id']
        if time_method == 'relative_time':
            t_evt = t_evt - base_time
            t_lbl = t_lbl - base_time
        base_time = 0
        events = np.stack([t_evt, x_evt, y_evt, p_evt], axis=-1).astype(np.int)
        labels = np.stack([t_lbl, x_lbl, y_lbl, w_lbl, h_lbl, c_lbl], axis=-1).astype(np.int)
        labels = self._xywh2xyxy(labels)
        return events, labels

    def _label_padding(self, labels, train_duration, gt_duration):
        train_duration = train_duration + gt_duration # add margin
        labels_dict = dict()
        num_gt_frames = int(train_duration / gt_duration) + 1

        if len(labels) > 0:
            times_lbl = labels[:,0].astype(np.int)
            offset = int(times_lbl[0] % gt_duration - gt_duration * 0.5)
            alinged_times = times_lbl - offset
            frame_indices = (alinged_times / gt_duration).astype(int)
            label_splits, frame_indices = self._split_by_indices(labels, frame_indices)

            for lbl, fidx in zip(label_splits, frame_indices):
                labels_dict[fidx] = lbl

            times_for_dummy_gt_frames = ((np.arange(num_gt_frames) - frame_indices[0]) * gt_duration + times_lbl[0]).astype(np.int) # TODO
            mask = np.logical_and(times_for_dummy_gt_frames >= 0, times_for_dummy_gt_frames < train_duration)
        else:
            times_for_dummy_gt_frames = (np.arange(num_gt_frames) * gt_duration).astype(np.int)
            mask = np.logical_and(times_for_dummy_gt_frames >= 0, times_for_dummy_gt_frames < train_duration)

        times_for_dummy_gt_frames = times_for_dummy_gt_frames[mask]
        num_gt_frames = len(times_for_dummy_gt_frames)

        labels_padded = []
        for i in range(num_gt_frames):
            if i in labels_dict:
                time = labels_dict[i][0,0]
                dummy_gt = np.array([[time , 0, 0, 1000, 1000, DUMMY_LABEL_IDX, False]], dtype=np.int)    # set dummy GT with for identifying gt timing
                labels_padded.append(dummy_gt)
                labels_padded.append(labels_dict[i])
            else:
                time = times_for_dummy_gt_frames[i]
                dummy_gt = np.array([[time , 0, 0, 1000, 1000, DUMMY_LABEL_IDX, False]], dtype=np.int)    # set dummy GT with for identifying gt timing
                labels_padded.append(dummy_gt)

        labels_padded = np.concatenate(labels_padded, axis=0)

        # discard margin
        labels_padded = labels_padded[labels_padded[:,0] < int(train_duration - gt_duration)]

        return labels_padded

    def _split_by_indices(self, data_array, indices):
        split_indices = np.flatnonzero(indices[1:] - indices[:-1]) + 1
        data_splits = np.split(data_array, split_indices)
        new_indices = np.unique(indices)
        return data_splits, new_indices

    def _blank_bbox_as_none(self, bbox_dict):
        if bbox_dict['times'] is None:
            return bbox_dict
        if len(bbox_dict['times']) == 0:
            bbox_dict['times'] = None
            bbox_dict['bboxes'] = None
            bbox_dict['labels'] = None
            bbox_dict['ignore_mask'] = None
        return bbox_dict

    def _event_downsample(self, events, num_samples_per_duration, duration):
        times_evt = events[:,0].astype(int)
        segment_indices_evt = times_evt // duration
        event_splits, segment_indices_evt = self._split_by_indices(events, segment_indices_evt)
        event_splits = [ self._randchoice(evt, num_samples_per_duration) for evt in event_splits ]
        output = np.concatenate(event_splits, axis=0)
        return output

    def _randchoice(self, data, num_sample):
        if len(data) <= num_sample:
            return data
        indices = np.random.choice(len(data), size=num_sample, replace=False)
        indices = np.sort(indices)
        return data[indices]


class EventFrame(EventPacket):
    def __init__(self, *args, event_repr=None, frame_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert event_repr is not None
        self.event_repr = build_eventrepr(event_repr)
        self.frame_transform = frame_transform

    def __getitem__(self, index):
        event_dict, bbox_dict, meta_data = self.getdata(index)
        bbox_dict = self._filter_dummy_label(bbox_dict)
        bbox_dict.pop('times')

        events = event_dict['events']
        image_meta = meta_data['image_meta']

        image, image_meta = self.event_repr(events, image_meta)
        if self.frame_transform is not None:
            image, bbox_dict, image_meta = self.frame_transform(image, bbox_dict, image_meta, types=['image', 'bbox', 'meta'])

        assert image.shape[1] == image_meta['height'] and image.shape[2] == image_meta['width']

        meta_data['image_meta'] = image_meta

        return image, bbox_dict, meta_data

# HMNet_dataset = '/home/tkyen/opencv_practice/data/Gen1_Automotive/HMNet'
# train_dataset = EventPacketStream(
#     fpath_evt_lst      = os.path.join(HMNet_dataset, 'list/train/events.txt'),
#     fpath_lbl_lst      = os.path.join(HMNet_dataset, 'list/train/labels.txt'),
#     base_path          = '',
#     fpath_meta         = os.path.join(HMNet_dataset, 'list/train/meta.pkl'),
#     fpath_gt_duration  = os.path.join(HMNet_dataset, 'list/train/gt_interval.csv'),
#     video_duration     = 60e6,
#     train_duration     = TRAIN_DURATION, # 200e3
#     delta_t            = DELTA_T, # 5e3
#     skip_ts            = 0,
#     use_nearest_label  = False,
#     sampling           = 'label',
#     min_box_diag       = 30,
#     min_box_side       = 10,
#     random_time_scaling = False,
#     start_index_aug_method = 'end',
#     start_index_aug_ratio = 0.25,
#     event_transform    = train_transform,
# )
class EventPacketStream(EventPacket):
    def __init__(self, *args, skip_ts=0, delta_t=1000, stream_stride=None, use_nearest_label=False,
                 time_method='relative_time', **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_t = delta_t                         # 5e3
        self.stride_t = stream_stride or delta_t       # 5e3
        self.skip_ts = skip_ts                         # 0
        self.use_nearest_label = use_nearest_label     # False
        self.time_method = time_method                 # 'relative_time' or 'absolute_time'

        self.sampling = kwargs['sampling']
        self.video_duration = kwargs['video_duration'] # 60e6 us
        self.train_duration = kwargs['train_duration'] # 200e3 us
        self.batch_size = kwargs['batch_size']

        if self.time_method == 'absolute_time':
            self.init_states = False
        else:
            self.init_states = True

    def __getitem__(self, index):
        if self.sampling == 'regular_batch':
            num_frames     = int(self.video_duration / self.train_duration)
            batch_index    = index // (self.batch_size * num_frames)
            batch_residual = index  % (self.batch_size * num_frames)
            seg_index =  batch_residual % self.batch_size
            time_index = (batch_residual - seg_index) // self.batch_size
            new_index = batch_index * (self.batch_size * num_frames) + seg_index * num_frames + time_index
            event_dict, bbox_dict, meta_data = self.getdata(new_index, keep_latest_labels=False, skip_ts=self.skip_ts,
                                                            time_method = self.time_method)
            if time_index == 0:
                self.init_states = True
            else:
                self.init_states = False
        else:
            event_dict, bbox_dict, meta_data = self.getdata(index, keep_latest_labels=False, skip_ts=self.skip_ts,
                                                            time_method = self.time_method)
        # event_dict = { 'events': torch.from_numpy(events) }
        # bbox_dict = {
        #     'times': torch.from_numpy(gt_times),
        #     'bboxes': torch.from_numpy(gt_bboxes),
        #     'labels': torch.from_numpy(gt_labels),
        #     'ignore_mask': torch.from_numpy(ignore_mask).bool(),
        # }
        # meta_data = { 
        #    'image_meta': {
        #                    'width': WIDTH,
        #                    'height': HEIGHT,
        #                    'filename'
        #                    'ori_filename'
        #                    'curr_time_org'
        #                    'curr_time_crop'
        #                    'delta_t'
        #                    'stride_t'
        #                    'time_scaling'
        #                  },
        #     'label_meta': dict(ignore_index=IGNORE_LABEL_IDX),
        # }

        events       = event_dict['events'].numpy()
        labels       = self._bboxdict2labels(bbox_dict)
        label_meta   = meta_data['label_meta']
        image_meta   = meta_data['image_meta']
        time_scaling = image_meta['time_scaling']
        fpath_evt    = image_meta['filename']

        train_duration = int(self.train_duration * time_scaling)
        delta_t        = int(self.delta_t * time_scaling)
        stride_t       = int(self.stride_t * time_scaling)
        num_frames     = int(math.ceil(train_duration / stride_t)) # 200e3 us / 5e3 us = 40

        # split data into sub-packets
        times_evt = events[:,0].astype(np.int)
        times_lbl = labels[:,0].astype(np.int)

        segment_indices_evt = times_evt // stride_t
        segment_indices_lbl = times_lbl // stride_t

        event_splits, segment_indices_evt = self._split_by_indices(events, segment_indices_evt)
        label_splits, segment_indices_lbl = self._split_by_indices(labels, segment_indices_lbl)

        # keep latest labels
        label_splits = [ self._keep_latest_labels(lbls) for lbls in label_splits ]

        backet_evt = DataBacket(num=num_frames)
        for event_data, seg_idx in zip(event_splits, segment_indices_evt):
            if self.time_method == 'absolute_time':
                seg_idx = seg_idx % num_frames
            if seg_idx < 0 or seg_idx >= num_frames:
                continue
            backet_evt.append(seg_idx, event_data)

        backet_lbl = DataBacket(num=num_frames)
        for label_data, seg_idx in zip(label_splits, segment_indices_lbl):
            if self.time_method == 'absolute_time':
                seg_idx = seg_idx % num_frames
            backet_lbl.append(seg_idx, label_data)

        if self.use_nearest_label:
            backet_lbl.pad_blank_backet()

        event_streams = backet_evt.concat(axis=0, dtype=np.float32)    # Tensor (L, 4)
        label_streams = backet_lbl.latest()
        meta_streams = []
        for i in range(len(backet_evt)):
            curr_time_org = (image_meta['curr_time_org'] - image_meta['curr_time_crop']) + stride_t * (i + 1)
            curr_time_crop = stride_t * (i + 1)
            image_meta = self._update_image_meta(image_meta, fpath_evt, curr_time_org, curr_time_crop, 
                                                 stride_t, time_scaling, self.init_states and i==0)
            meta = {
                'image_meta': image_meta,
                'label_meta': label_meta,
            }
            meta_streams.append(meta)

        if delta_t != stride_t:
            event_streams, label_streams, meta_streams = self.merge_streams(event_streams, label_streams, meta_streams, delta_t, stride_t)

        data_streams = [ {'events': torch.from_numpy(evt)} for evt in event_streams ]
        target_streams = [ self._labels2bboxdict(lbls) for lbls in label_streams ]
        target_streams = [ self._blank_bbox_as_none(bbox_dict) for bbox_dict in target_streams ]
        target_streams = [ self._filter_dummy_label(bbox_dict) for bbox_dict in target_streams ]
        for bbox_dict in target_streams:
            bbox_dict.pop('times')

        assert len(data_streams) == len(target_streams)
        assert len(data_streams) == len(meta_streams)
        # data_streams = [ { 'events': torch.from_numpy(events) }, {...}, {...}, ...]
        # target_streams = [ { 'bboxes': torch.from_numpy(gt_bboxes),
        #                      'labels': torch.from_numpy(gt_labels),
        #                      'ignore_mask': torch.from_numpy(ignore_mask).bool() },
        #                    {...},
        #                    {...},...
        #                  ]
        # meta_streams = [ { 'image_meta': { 'width': WIDTH,
        #                                    'height': HEIGHT,
        #                                    'filename'
        #                                    'ori_filename'
        #                                    'curr_time_org'
        #                                    'curr_time_crop'
        #                                    'delta_t'
        #                                    'stride_t'
        #                                    'time_scaling'},
        #                    'label_meta': dict(ignore_index=IGNORE_LABEL_IDX) },
        #                  {...},
        #                  {...},...
        #                ]
        return data_streams, target_streams, meta_streams

    def merge_streams(self, event_streams, label_streams, meta_streams, delta_t, stride_t):
        N = len(event_streams)
        l = int(delta_t // stride_t)
        delta_t = l * stride_t
        event_streams = [ self._merge_events(event_streams[i-l:i]) for i in range(l, N) ]
        label_streams = [ label_streams[i] for i in range(l, N)]
        meta_streams  = [  meta_streams[i] for i in range(l, N)]
        for meta in meta_streams:
            meta['image_meta']['delta_t'] = delta_t
        return event_streams, label_streams, meta_streams

    def _merge_events(self, list_events):
        list_events = [ events for events in list_events if len(events) > 0 ]
        if len(list_events) == 0:
            return np.array([], dtype=np.float32)
        else:
            return np.concatenate(list_events, axis=0)

class EventFrameStream(EventPacketStream):
    def __init__(self, *args, event_repr=None, frame_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert event_repr is not None
        assert frame_transform is not None
        self.event_repr = build_eventrepr(event_repr)
        self.frame_transform = frame_transform

    def __getitem__(self, index):
        data_streams, target_streams, meta_streams = super().__getitem__(index)

        out_image_streams, out_target_streams, out_meta_streams = [], [], []
        for idx, (event_dict, bbox_dict, meta_data) in enumerate(zip(data_streams, target_streams, meta_streams)):
            repeat = idx > 0
            events = event_dict['events']
            image_meta = meta_data['image_meta']

            image, image_meta = self.event_repr(events, image_meta)
            if self.frame_transform is not None:
                if bbox_dict['labels'] is not None:
                    image, bbox_dict, image_meta = self.frame_transform(image, bbox_dict, image_meta, types=['image', 'bbox', 'meta'], repeat=repeat)
                else:
                    image, image_meta = self.frame_transform(image, image_meta, types=['image', 'meta'], repeat=repeat)

            assert image.shape[1] == image_meta['height'] and image.shape[2] == image_meta['width']

            meta_data['image_meta'] = image_meta

            out_image_streams.append(image)
            out_target_streams.append(bbox_dict)
            out_meta_streams.append(meta_data)

        return out_image_streams, out_target_streams, out_meta_streams



class DataBacket(object):
    def __init__(self, num=1):
        self._backet = [ list() for _ in range(num) ]

    def append(self, idx, data):
        if idx >= len(self._backet):
            num_append = idx - len(self._backet) + 1
            blank_backets = [ list() for _ in range(num_append) ]
            self._backet += blank_backets
        self._backet[idx].append(data)

    def _backet_dtype(self):
        for contents in self._backet:
            if len(contents) > 0:
                return contents[0].dtype

    def concat(self, axis, dtype=None):
        if dtype is None:
            dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(np.concatenate(contents, axis=axis))
        return output

    def stack(self, axis=0):
        dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(np.stack(contents, axis=axis))
        return output

    def latest(self):
        dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(contents[-1])
        return output

    def expand(self, length):
        if length > len(self._backet):
            num_append = length - len(self._backet)
            blank_backets = [ list() for _ in range(num_append) ]
            self._backet += blank_backets

    def pad_blank_backet(self, direction='forward'):
        if direction == 'forward':
            self._pad_forward()
            self._pad_backward()
        elif direction == 'backward':
            self._pad_backward()
            self._pad_forward()

    def blank_backet_as_none(self):
        for i in range(len(self._backet)):
            if len(self._backet[i]) == 0:
                self._backet[i] = None

    def _pad_forward(self):
        pad = None
        for i, backet in enumerate(self._backet):
            if len(backet) > 0:
                pad = backet
            elif pad is not None:
                self._backet[i] = copy.deepcopy(pad)
            else:
                pass

    def _pad_backward(self):
        pad = None
        for i, backet in enumerate(reversed(self._backet)):
            if len(backet) > 0:
                pad = backet
            elif pad is not None:
                self._backet[len(self._backet)-1-i] = copy.deepcopy(pad)
            else:
                pass

    @property
    def data(self):
        return self._backet

    def __len__(self):
        return len(self._backet)


