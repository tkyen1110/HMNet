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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_memory import LatentMemory, EventEmbedding
from ..blocks import BlockBase
from ..init import init_transformer

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]

from hmnet.utils.common import Timer
timer = Timer()

class HMNet(BlockBase):
    def __init__(self, latent_sizes, latent_dims, output_dims, num_heads, depth, warmup=20,
                 cfg_embed=None, cfg_memory1=None, cfg_memory2=None, cfg_memory3=None, relative_time=True) -> None:
        super().__init__()
        self.latent_dims = latent_dims   # [128, 256, 256]
        self.output_dims = output_dims   # [256, 256, 256]
        self.latent_sizes = latent_sizes # [(60, 76), (30, 38), (15, 19)]
        self.warmup = warmup    # output is valid after the warmup time steps

        D0 = sum(cfg_embed['out_dim']) # sum([32, 32, 32]) = 96
        D1, D2, D3 = latent_dims       # [128, 256, 256]
        O1, O2, O3 = output_dims       # [256, 256, 256]
        L1, L2, L3 = latent_sizes      # [(60, 76), (30, 38), (15, 19)]
        H1, H2, H3 = num_heads         # [4, 8,  8]
        N1, N2, N3 = depth             # [1, 3,  9]

        self.memory1 = LatentMemory(latent_size=L1, input_dim=D0, latent_dim=D1, output_dim=O1, num_heads=H1, update_depth=N1, \
                                    message_gen=False, event_write=True,  top_down=True, relative_time=relative_time, \
                                    **cfg_memory1, cfg_embed=cfg_embed)
        self.memory2 = LatentMemory(latent_size=L2, input_dim=D1, latent_dim=D2, output_dim=O2, num_heads=H2, update_depth=N2, \
                                    message_gen=True,  event_write=False, top_down=True, relative_time=relative_time,  \
                                    **cfg_memory2)
        self.memory3 = LatentMemory(latent_size=L3, input_dim=D2, latent_dim=D3, output_dim=O3, num_heads=H3, update_depth=N3, \
                                    message_gen=True,  event_write=False, top_down=False, relative_time=relative_time, \
                                    **cfg_memory3)
        self.set_module_names()

    def init_weights(self, pretrained=None):
        init_transformer(self.modules())

    def to_cuda(self, d0, d1, d2):
        self.devices = (d0, d1, d2)
        self.memory1 = self.memory1.to(d0)
        self.memory2 = self.memory2.to(d1)
        self.memory3 = self.memory3.to(d2)

    def set_devices(self, d0, d1, d2):
        self.devices = (d0, d1, d2)

    def set_multi_process(self, flags):
        f1, f2, f3 = flags
        self.memory1.use_multi_process = f1
        self.memory2.use_multi_process = f2
        self.memory3.use_multi_process = f3

    def set_cuda_stream(self, flags, priorities=[-1,0,0]):
        f1, f2, f3 = flags
        p1, p2, p3 = priorities
        self.memory1.use_cuda_stream = f1
        self.memory2.use_cuda_stream = f2
        self.memory3.use_cuda_stream = f3
        self.memory1.cuda_stream_priority = p1
        self.memory2.cuda_stream_priority = p2
        self.memory3.cuda_stream_priority = p3

    def prepair_for_inference(self, batch_size, image_size=None):
        memory1_dim, memory2_dim, memory3_dim = self.latent_dims     # [128, 256, 256]
        memory1_size, memory2_size, memory3_size = self.latent_sizes # [(60, 76), (30, 38), (15, 19)]
        d0, d1, d2 = self.devices

        self.memory1.prepair_for_inference(batch_size, device=d0)
        self.memory2.prepair_for_inference(batch_size, device=d1, input_dim=memory1_dim, input_size=memory1_size)
        self.memory3.prepair_for_inference(batch_size, device=d2, input_dim=memory2_dim, input_size=memory2_size, image_size=image_size)

    def forward(self, list_events, list_image_metas, gather_indices, list_images=None, init_states=True, detach=True, fast_training=True):
        # list_events       (list of list) = [Ts, B] = [20, 4]
        # list_image_metas  (list of list) = [Ts, B] = [20, 4]
        # list_images = None
        if list_images is None:
            list_images = [None] * len(list_events)

        if init_states:
            batch_size = len(list_events[0])
            height = list_image_metas[0][0]['height']
            width = list_image_metas[0][0]['width']
            self.memory1.init_states(batch_size, image_size=(height, width))
            self.memory2.init_states(batch_size, image_size=(height, width))
            self.memory3.init_states(batch_size, image_size=(height, width))

        # set place holders for outputs
        num_output = len(gather_indices['time'])
        outputs1 = self.memory1.place_holder(num_output)
        outputs2 = self.memory2.place_holder(num_output)
        outputs3 = self.memory3.place_holder(num_output)

        if fast_training:
            # extract key value in advance for fast trainig
            list_events = self.memory1.embed.forward_fast_train(self.memory1, list_events, list_image_metas)
            # list_events (list of list) = [Ts, 3]

        for time_idx, (events, images, image_metas) in enumerate(zip(list_events, list_images, list_image_metas)):
            # keys, values, ev_q = events
            # keys.shape = torch.Size([4927, 4, 32])
            # values.shape = torch.Size([4927, 4, 32])
            # ev_q.shape = torch.Size([4927])

            # forward one time step
            out1, out2, out3 = self._forward_one_step(events, image_metas, image_input=images, fast_training=fast_training)
            # out1.shape = torch.Size([B=7, 256, 60, 76])
            # out2.shape = torch.Size([B=7, 256, 30, 38])
            # out3.shape = torch.Size([B=7, 256, 15, 19])
            # time_idx =  0, out1 = new, out2 = None, out3 = None
            # time_idx =  1, out1 = new, out2 = None, out3 = None
            # time_idx =  2, out1 = new, out2 = None, out3 = None
            # time_idx =  3, out1 = new, out2 =  new, out3 = None
            # time_idx =  4, out1 = new, out2 =  old, out3 = None
            # time_idx =  5, out1 = new, out2 =  old, out3 = None
            # time_idx =  6, out1 = new, out2 =  new, out3 = None
            # time_idx =  7, out1 = new, out2 =  old, out3 = None
            # time_idx =  8, out1 = new, out2 =  old, out3 = None
            # time_idx =  9, out1 = new, out2 =  new, out3 =  new
            # time_idx = 10, out1 = new, out2 =  old, out3 =  old
            # time_idx = 11, out1 = new, out2 =  old, out3 =  old
            # time_idx = 12, out1 = new, out2 =  new, out3 =  old
            # time_idx = 13, out1 = new, out2 =  old, out3 =  old
            # time_idx = 14, out1 = new, out2 =  old, out3 =  old
            # time_idx = 15, out1 = new, out2 =  new, out3 =  old
            # time_idx = 16, out1 = new, out2 =  old, out3 =  old
            # time_idx = 17, out1 = new, out2 =  old, out3 =  old
            # time_idx = 18, out1 = new, out2 =  new, out3 =  new
            # time_idx = 19, out1 = new, out2 = None, out3 =  old
            '''
            if out1!=None:
                out1_shape = out1.shape
                if out1_old!=None:
                    out1_equal = torch.equal(out1, out1_old)
                out1_old = out1
            else:
                out1_shape = None

            if out2!=None:
                out2_shape = out2.shape
                if out2_old!=None:
                    out2_equal = torch.equal(out2, out2_old)
                out2_old = out2
            else:
                out2_shape = None

            if out3!=None:
                out3_shape = out3.shape
                if out3_old!=None:
                    out3_equal = torch.equal(out3, out3_old)
                out3_old = out3
            else:
                out3_shape = None
            print(time_idx)
            print(out1_shape, out1_equal)
            print(out2_shape, out2_equal)
            print(out3_shape, out3_equal)
            print()
            '''
            if out1 is None or out2 is None or out3 is None:
                continue

            # gather outputs of annotated timmings for loss calculation
            outputs1, outputs2, outputs3 = self._gather((outputs1, outputs2, outputs3), (out1, out2, out3), gather_indices, time_idx)

        # detach memory states (required for TBPTT)
        if detach:
            self.memory1.detach()
            self.memory2.detach()
            self.memory3.detach()

        return outputs1, outputs2, outputs3

    def inference(self, events, image_metas, image_input=None) -> Tensor:
        return self._forward_one_step(events, image_metas, image_input)

    def _forward_one_step(self, events, image_metas, image_input=None, fast_training=False) -> Tensor:
        # keys, values, ev_q = events
        # keys.shape = torch.Size([4927, 4, 32])
        # values.shape = torch.Size([4927, 4, 32])
        # ev_q.shape = torch.Size([4927])

        # get event meta data
        curr_time = [ meta['curr_time_crop'] for meta in image_metas ]
        duration = [ meta['delta_t'] for meta in image_metas ]

        # get current state
        z3, message3 = self.memory3.sync_and_get_state()
        # z3.meta = {'shape': [B, H=15, W=19]} ; z3.data.shape = torch.Size([B, HxW=285 , latent_dim=256])
        z2, message2 = self.memory2.sync_and_get_state()
        # z2.meta = {'shape': [B, H=30, W=38]} ; z2.data.shape = torch.Size([B, HxW=1140, latent_dim=256])
        z1, message1 = self.memory1.sync_and_get_state()
        # z1.meta = {'shape': (B, H=60, W=76)} ; z1.data.shape = torch.Size([B, HxW=4560, latent_dim=128])

        # forward one time step
        out3 = self.memory3(z2, None, image_input=image_input)
        out2 = self.memory2(z1, message3)
        out1 = self.memory1(events, message2, event_metas=(curr_time, duration), fast_training=fast_training)

        return out1, out2, out3

    def _gather(self, list_dst, list_src, gather_indices, time_idx):
        time_indices = gather_indices['time']
        batch_indices = gather_indices['batch']
        assert len(time_indices) == len(batch_indices)
        destination = torch.arange(len(time_indices))

        mask = time_indices == time_idx
        src_indices = batch_indices[mask]
        dst_indices = destination[mask]

        for dst, src in zip(list_dst, list_src):
            dst[dst_indices] = src[src_indices]

        return list_dst

    def termination(self):
        self.memory1.termination()
        self.memory2.termination()
        self.memory3.termination()

    def get_dummy_output(self):
        z1_out = self.memory1.out_buffer[:1]
        z2_out = self.memory2.out_buffer[:1]
        z3_out = self.memory3.out_buffer[:1]
        return z1_out, z2_out, z3_out



