# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-autotuned GEMM kernel
=============================================

Based on tutorial on tuning by Lianmin Zheng: https://docs.tvm.ai/tutorials/autotvm/tune_simple_template.html#sphx-glr-tutorials-autotvm-tune-simple-template-py 
"""

import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from topi.util import get_const_tuple
import topi
import topi.testing

import Passes



def conv_pack_conservative(batch, in_channel, in_size, num_filter, kernel):
    global B,adtype,wdtype,a_shape,w_shape
    in_height = in_width = in_size



    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A', dtype='uint8')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W', dtype='int8')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    adtype = A.dtype
    wdtype = W.dtype

    with tvm.target.create('llvm'):
        B = topi.nn.conv2d(A, W, 3, 1, 1, layout='NHWC', out_dtype="int32")
        s = topi.generic.schedule_conv2d_nhwc_pack([B])

    #### ADDED AUTO AUTOTUNING ####
#    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.CONSERVATIVE)
    ###############################

    dtype = A.dtype

    return s,[A,W,B]

def conv_pack_numpy(a_np,w_np):
    dw_np = topi.testing.dilate_python(w_np, (1, 1, 1, 1))
    b_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, 3, 1)
    return b_np

def conv_pack_input_generator(batch, in_channel, in_size, num_filter, kernel, dilation=1):
    a_np = np.random.uniform(size=a_shape).astype(adtype)
    w_np = np.random.uniform(size=w_shape).astype(wdtype)
    return a_np, w_np

def _a_shape(batch, in_channel, in_size):
    in_height = in_width = in_size
    return get_const_tuple((batch,in_height,in_width,in_channel))

def _w_shape(kernel,in_channel,num_filter):
    return get_const_tuple((kernel,kernel,in_channel,num_filter))

def conv_pack_default_args():
    batch = 2
    in_channel = 256
    in_size = 32
    num_filter = 256
    kernel = 2
    return (batch,in_channel,in_size,num_filter,kernel)

if __name__ == "__main__":
    
    args = conv_pack_default_args()



    with tvm.target.create("llvm"):
        s, arg_bufs = conv_pack_conservative(*conv_pack_default_args())
        func = tvm.build(s, arg_bufs)

    a_np,w_np = conv_pack_input_generator(*args)
    b_np = conv_pack_numpy(a_np,w_np)
    A,W,B = arg_bufs
    ctx = tvm.context("llvm", 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func(a,w,b)
    

    tvm.testing.assert_allclose(b_np, b.asnumpy(), rtol=1e-5)
