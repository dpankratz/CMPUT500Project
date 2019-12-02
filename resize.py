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



@autotvm.template
def resize_naive(batch, in_channel, in_height, in_width, out_height, out_width):
    global B
    out_shape = (batch, in_channel, out_height, out_width)
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.image.resize(A, (out_height, out_width), layout="NCHW", align_corners=True, method="bilinear")
    s = tvm.create_schedule(B.op)

    #### ADDED AUTO AUTOTUNING ####
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.NAIVE)
    ###############################

    dtype = A.dtype

    return s,[A,B]

@autotvm.template
def resize_moderate(batch, in_channel, in_height, in_width, out_height, out_width):
    global B
    out_shape = (batch, in_channel, out_height, out_width)
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.image.resize(A, (out_height, out_width), layout="NCHW", align_corners=True, method="bilinear")
    s = tvm.create_schedule(B.op)

    #### ADDED AUTO AUTOTUNING ####
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.NONNAIVE)
    ###############################

    dtype = A.dtype

    return s,[A,B]



@autotvm.template
def resize_conservative(batch, in_channel, in_height, in_width, out_height, out_width):
    global B
    out_shape = (batch, in_channel, out_height, out_width)
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.image.resize(A, (out_height, out_width), layout="NCHW", align_corners=True, method="bilinear")
    s = tvm.create_schedule(B.op)

    #### ADDED AUTO AUTOTUNING ####
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.CONSERVATIVE)
    ###############################

    dtype = A.dtype

    return s,[A,B]

def resize_numpy(a_np):
    _,_,_,_,out_height,out_width = resize_default_args()
    b_np = topi.testing.bilinear_resize_python(a_np, (out_height, out_width), 'NCHW', True)
    return b_np

def resize_input_generator(batch, in_channel, in_height, in_width, out_height, out_width):
    a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype('float32')
    return [a_np]

def resize_default_args():
    return (4, 16, 32, 32, 50, 50)

if __name__ == "__main__":
    
    args = resize_default_args()


    with tvm.target.create("llvm"):
        s, arg_bufs = resize_conservative(*resize_default_args())
        func = tvm.build(s, arg_bufs)

    a_np = resize_input_generator(*args)
    b_np = resize_numpy(a_np)
    
    ctx = tvm.context("llvm", 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func(a,b)
    

    tvm.testing.assert_allclose(b_np, b.asnumpy(), rtol=1e-2)
