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

import numpy as np
import glob

from hmnet.utils.common import get_list

VAL_IDS = [
    8,   9,  23,  24,  35,  36,  37,  38,  39,  43, 
   44,  51,  52,  65,  85,  86,  95,  96,  98, 102, 
  103, 112, 113, 118, 119, 121, 127, 128, 131, 133, 
  135, 136, 139, 141, 143, 145, 146, 149, 152, 156, 
  159, 160, 161, 163, 164, 165, 169, 174, 177, 178, 
  179, 180, 189, 199, 200, 204, 206, 207, 208, 209, 
  210, 211, 212, 213, 214, 215, 216, 219, 220, 221, 
  222, 228, 230, 231, 232, 233, 234, 235, 236, 237, 
  242, 243, 244, 245, 249, 250, 270, 272, 273, 280, 
  281, 289, 290, 291, 292, 294, 295, 296, 297, 300, 
]

def get_filter(phase):
    get_seqid = lambda fpath: int(fpath.split('/')[3].split('_')[1])

    if phase == 'train':
        return lambda fpath: 'Town01' in fpath or 'Town02' in fpath or 'Town03' in fpath
    elif phase == 'val':
        return lambda fpath: 'Town05' in fpath and get_seqid(fpath) in VAL_IDS
    elif phase == 'test':
        return lambda fpath: 'Town05' in fpath and get_seqid(fpath) not in VAL_IDS

def save(phase):
    header = 'name,start,end'

    filter = get_filter(phase)

    list_fpath = glob.glob('./source/Town*/sequence_*/events/data/boundary_timestamps.txt')
    list_fpath = [ fpath for fpath in list_fpath if filter(fpath) ]

    out = ''
    for fpath in list_fpath:
        print(fpath)
        name = fpath.replace('./source/', '').replace('/events/data/boundary_timestamps.txt', '').replace('/', '_') + '_depth.npy'

        with open(fpath, 'r') as fp:
            lines = fp.read().split('\n')
        if lines[-1] == '':
            lines = lines[:-1]

        start = float(lines[0].split(' ')[1]) * 1.0e6
        end = float(lines[-1].split(' ')[2]) * 1.0e6
        out += f'\n{name},{start},{end}'

    with open(f'./list/{phase}/video_duration.csv', 'w') as fp:
        fp.write(header + out)


save('train')
save('val')
save('test')
