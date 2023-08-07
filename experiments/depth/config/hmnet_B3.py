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

import sys
import os
import copy

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrsch

from hmnet.utils.lr_lambda import CombinationV2
from hmnet.utils.common import CDict as dict
from hmnet.utils.transform import Compose, RandomCrop, RandomFlip, RandomResize, Padding, Resize, ResizeInside
from hmnet.dataset import eventscape
from hmnet.dataset import mvsec
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.models.depth import HMDepth

TRAIN_DURATION = 200e3
DELTA_T = 5e3
INPUT_SIZE = (256, 512)
GRAYSCALE = False

backbone = dict(
    type         = 'HMNet',
    latent_sizes = [(64, 128), (32, 64), (16, 32)],
    latent_dims  = [128, 256, 256],
    output_dims  = [256, 256, 256],
    num_heads    = [4, 8, 8],
    depth        = [1, 3, 9],

    cfg_embed = dict(
        input_size    = INPUT_SIZE,
        out_dim       = [32, 32, 32],
        duration      = DELTA_T,
        discrete_time = True,
        time_bins     = 100,
        dynamic       = [True,True,True],
        dynamic_dim   = [32, 32, 32],
    ),

    cfg_memory1 = dict(
        freq = 1,
        vector_latent = True,

        cfg_update = dict(
            layer = 'res',
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        ),
    ),

    cfg_memory2 = dict(
        freq = 3,
        vector_latent = True,

        cfg_update = dict(
            layer = 'res',
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        ),

        cfg_write = dict(
            downsample  = 'conv',
            input_proj  = True,
            window_size = (7,7),
            grouping    = 'intra-window',
            act_layer   = nn.GELU,
            norm_layer  = nn.LayerNorm,
            drop = 0.,
            pos_dynamic = False,
            pos_log_scale = False,
        ),

        cfg_message = dict(
            input_resize  = 'merge',
            latent_resize = 'none',
            input_proj    = True,
            latent_proj   = False,
            out_proj      = True,
            norm_layer    = nn.LayerNorm,
            drop          = 0.,
            window_size   = (7,7),
            grouping    = 'intra-window',
            pos_dynamic   = False,
            pos_log_scale = False
        ),
    ),

    cfg_memory3 = dict(
        freq = 9,
        vector_latent = True,

        cfg_update = dict(
            layer = 'res',
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        ),

        cfg_write = dict(
            downsample  = 'conv',
            input_proj  = True,
            window_size = (7,7),
            grouping    = 'intra-window',
            act_layer   = nn.GELU,
            norm_layer  = nn.LayerNorm,
            drop = 0.,
            pos_dynamic = False,
            pos_log_scale = False,
        ),

        cfg_message = dict(
            input_resize  = 'merge',
            latent_resize = 'none',
            input_proj    = True,
            latent_proj   = False,
            out_proj      = True,
            norm_layer    = nn.LayerNorm,
            drop          = 0.,
            window_size   = (7,7),
            grouping    = 'intra-window',
            pos_dynamic   = False,
            pos_log_scale = False
        ),
    ),
)

neck = dict(
    type = 'Pyramid',
    input_proj     = False,
    in_channels    = [256, 256, 256],
    input_start_index = 0,
    dim            = 256,
    out_indices    = [0, 1, 2],
    cfg_pyramid_fuse = [
        dict(direction='bottom-up', pre_trans=False, fuse_method='add', post_trans=True, post_convs=False),
        dict(direction='top-down' , pre_trans=False, fuse_method='add', post_trans=True, post_convs=False),
    ],
)

head = dict(
    type = 'DepthRegHead',
    inc  = 256,
    outc = 1,
    num_extra_conv = 1,
    norm_layer     = nn.GroupNorm,
    act_layer      = nn.SiLU,
    drop           = 0.1,
    nlogdepth      = True,
    max_depth      = 1000,
    min_depth      = 3.346,
    cfg_loss       = [
        {'loss_type': 'SIGLoss',   'coef': 1.0},
        {'loss_type': 'GMLoss',    'coef': 0.25, 'grad_matching_scales': [1.0, 0.5, 0.25]},
    ],
)


class TrainSettings(object):
    # ======== train data settings ========
    def get_dataset(self):
        train_transform = Compose([
            RandomFlip(prob=0.5, direction='H'),
        ])

        train_dataset = eventscape.EventPacketStream(
            fpath_evt_lst        = './data/eventscape/list/train/events.txt',
            fpath_image_lst      = './data/eventscape/list/train/images.txt',
            fpath_label_lst      = './data/eventscape/list/train/labels.txt',
            base_path            = './data/eventscape',
            fpath_meta           = './data/eventscape/list/train/meta.pkl',
            fpath_video_duration = './data/eventscape/list/train/video_duration.csv',
            train_duration       = TRAIN_DURATION,
            delta_t              = DELTA_T,
            sampling             = 'random',
            skip_ts              = 0,
            use_nearest_label    = False,
            random_time_scaling  = False,
            start_index_aug_method = 'end',
            start_index_aug_ratio  = 0.25,
            event_transform      = train_transform,
            grayscale            = GRAYSCALE,
            skip_image_loading   = True,
        )

        return train_dataset

    loader_param = dict(
        batch_size  = 16,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_keep_dict,
    )

    # ======== model settings ========
    def get_model(self):
        d0 = torch.device('cuda:0')
        model = HMDepth(backbone, neck, head, devices=[d0,d0,d0,d0])
        model.init_weights()
        return model

    # ======== optimizer settings ========
    N_SAMPLES = 72371
    NUM_EPOCHS = 30
    bsize = loader_param['batch_size']
    iter_per_epoch = N_SAMPLES // bsize
    maxiter = iter_per_epoch * NUM_EPOCHS

    optimizer    = torch.optim.AdamW
    optim_params = dict(
        lr           = 2.0e-4,
        betas        = (0.9, 0.999),
        weight_decay = 0.01,
    )
    schedule = [
        {'method': 'cosine', 'range': (0,maxiter), 'start_lr': optim_params['lr'], 'end_lr': 1.0e-7}
    ]
    lr_scheduler = lrsch.LambdaLR
    lrsch_params = dict( lr_lambda = CombinationV2(schedule, optim_params['lr']) )

    # ======== DDP settings ========
    grad_as_bucket_view = True
    static_graph = False

    # ======== other settings ========
    resume      = ''
    print_freq  = 10
    fpath_script= sys.argv[0]

class FTSettings(object):
    # ======== train data settings ========
    def get_dataset(self):
        train_transform = Compose([
            Padding(size=(260,348), halign='center', valign='center', const_image=0, const_mask=-1, padding_mode='constant'),
            RandomFlip(prob=0.5, direction='H'),
        ])

        train_dataset = mvsec.EventPacketStream(
            fpath_data           = './data/mvsec/source/outdoor_day2_data.hdf5',
            fpath_gt             = './data/mvsec/source/outdoor_day2_gt.hdf5',
            fpath_meta           = './data/mvsec/source/outdoor_day2_meta.npy',
            image_mean_std       = 'eventscape',
            train_duration       = TRAIN_DURATION,
            delta_t              = DELTA_T,
            sampling             = 'label',
            use_nearest_label    = False,
            random_time_scaling  = False,
            start_index_aug_method = 'end',
            start_index_aug_ratio  = 0.25,
            event_transform      = train_transform,
            skip_image_loading   = True,
        )

        return train_dataset

    loader_param = dict(
        batch_size  = 16,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_keep_dict,
    )

    # ======== model settings ========
    def get_model(self):
        backbone['cfg_embed']['input_size'] = (260, 348)
        backbone['latent_sizes'] = [(65, 87), (33, 44), (17, 22)]
        head['max_depth'] = 80
        head['min_depth'] = 1.978

        d0 = torch.device('cuda:0')
        model = HMDepth(backbone, neck, head, devices=[d0,d0,d0,d0])
        model.init_weights()
        return model

    # ======== optimizer settings ========
    N_SAMPLES = 3000
    NUM_EPOCHS = 30
    bsize = loader_param['batch_size']
    iter_per_epoch = N_SAMPLES // bsize
    maxiter = iter_per_epoch * NUM_EPOCHS

    optimizer    = torch.optim.AdamW
    optim_params = dict(
        lr           = 2.0e-4,
        betas        = (0.9, 0.999),
        weight_decay = 0.01,
    )
    schedule = [
        {'method': 'cosine', 'range': (0,maxiter), 'start_lr': optim_params['lr'], 'end_lr': 1.0e-7}
    ]
    lr_scheduler = lrsch.LambdaLR
    lrsch_params = dict( lr_lambda = CombinationV2(schedule, optim_params['lr']) )

    # ======== DDP settings ========
    grad_as_bucket_view = True
    static_graph = False

    # ======== other settings ========
    resume      = ''
    load        = './workspace/hmnet_B3/checkpoint.pth.tar'
    print_freq  = 10
    fpath_script= sys.argv[0]

class TestEventscape(object):
    def get_model(self, devices=None, mode='single_process'):
        assert mode in ('single_process', 'multi_process', 'cuda_stream')
        if devices is None:
            d = torch.device('cuda:0')
            devices = [d, d, d, d]

        model = HMDepth(backbone, neck, head, devices=devices, test_aug=None)
        model.init_weights()

        if mode == 'multi_process':
            model.backbone.set_multi_process([False, True, True])
        elif mode == 'cuda_stream':
            model.backbone.set_cuda_stream([True, True, True])

        return model

    def get_dataset(self, fpath_evt, fpath_rgb, fpath_lbl, fpath_meta, fpath_video_duration, base_path, fast_mode=False, delta_t=None):
        delta_t = delta_t or DELTA_T

        test_dataset = eventscape.EventPacketStream(
            fpath_evt_lst      = [fpath_evt],
            fpath_image_lst    = [fpath_rgb],
            fpath_label_lst    = [fpath_lbl],
            base_path          = base_path,
            fpath_meta         = fpath_meta,
            fpath_video_duration = fpath_video_duration,
            train_duration     = 10e6,
            sampling_stride    = 10e6,
            delta_t            = delta_t,
            skip_ts            = 0,
            use_nearest_label  = False,
            sampling           = 'regular',
            start_index_aug_method = 'none',
            start_index_aug_ratio  = 0.,
            random_time_scaling = False,
            min_time_scale      = 0.5,
            max_time_scale      = 2.0,
            event_transform     = None,
            output_type         = 'long' if fast_mode else None,
            skip_image_loading  = True,
        )

        return test_dataset

    # ======== prediction settings ========
    checkpoint      = 'checkpoint.pth.tar'
    batch_size      = 1

    to_device_in_model = True



class TestMVSEC(object):
    def get_model(self, devices=None, mode='single_process'):
        assert mode in ('single_process', 'multi_process', 'cuda_stream')
        if devices is None:
            d = torch.device('cuda:0')
            devices = [d, d, d, d]

        backbone['cfg_embed']['input_size'] = (260, 348)
        backbone['latent_sizes'] = [(65, 87), (33, 44), (17, 22)]
        head['max_depth'] = 80
        head['min_depth'] = 1.978

        test_transform = Compose([
            Padding(size=(260,348), halign='center', valign='center', const_image=0, const_mask=-1, padding_mode='constant'),
        ])

        model = HMDepth(backbone, neck, head, devices=devices, test_aug=test_transform)
        model.init_weights()

        if mode == 'multi_process':
            model.backbone.set_multi_process([False, True, True])
        elif mode == 'cuda_stream':
            model.backbone.set_cuda_stream([True, True, True])

        return model

    def get_dataset(self, fpath_data, fpath_gt, fpath_meta, image_mean_std, fast_mode=False, delta_t=None, debug=False):
        delta_t = delta_t or DELTA_T

        test_dataset = mvsec.EventPacketStream(
            fpath_data           = fpath_data,
            fpath_gt             = fpath_gt,
            fpath_meta           = fpath_meta,
            image_mean_std       = image_mean_std,
            delta_t              = delta_t,
            train_duration       = 270e6 if not debug else 10e6,
            sampling_stride      = 270e6 if not debug else 10e6,
            use_nearest_label    = False,
            sampling             = 'regular',
            random_time_scaling  = False,
            start_index_aug_method = 'none',
            start_index_aug_ratio  = 0.,
            event_transform      = None,
            output_type          = 'long' if fast_mode else None,
            skip_image_loading   = True,
        )

        return test_dataset

    # ======== prediction settings ========
    checkpoint      = 'checkpoint_ft.pth.tar'
    batch_size      = 1

    to_device_in_model = True


