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
def reorg_naive(batch, in_size, in_channel):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.vision.reorg(A, 2)

    a_shape = get_const_tuple(A.shape)
    dtype = A.dtype
    s=tvm.create_schedule(B.op)
    #### ADDED AUTO AUTOTUNING ####
    cfg = autotvm.get_config()
    cfg.add_flop(32)
    Passes.enable_autotune(s,[B],cfg,mode=Passes.NAIVE)
    ###############################

    dtype = A.dtype

    return s,[A,B]

@autotvm.template
def reorg_moderate(batch, in_size, in_channel):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.vision.reorg(A, 2)

    a_shape = get_const_tuple(A.shape)
    dtype = A.dtype
    s=tvm.create_schedule(B.op)
    #### ADDED AUTO AUTOTUNING ####
    cfg = autotvm.get_config()
    cfg.add_flop(32)
    Passes.enable_autotune(s,[B],cfg,mode=Passes.NONNAIVE)
    ###############################

    dtype = A.dtype

    return s,[A,B]


@autotvm.template
def reorg_conservative(batch, in_size, in_channel):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.vision.reorg(A, 2)

    a_shape = get_const_tuple(A.shape)
    dtype = A.dtype
    s=tvm.create_schedule(B.op)

    cfg = autotvm.get_config()
    cfg.add_flop(32)
    Passes.enable_autotune(s,[B],cfg,mode=Passes.CONSERVATIVE)
    ###############################

    dtype = A.dtype

    return s,[A,B]


    
    

def reorg_input_generator(batch, in_size, in_channel):
    in_height = in_width = in_size
    a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype('float32')
    return [a_np]

def reorg_numpy(a_np):
    b_np = topi.testing.reorg_python(a_np, 2)
    return b_np

def reorg_default_args():
    return (1, 20, 8)


if __name__ == "__main__":

    ctx = tvm.context('llvm', 0)


    with tvm.target.create('llvm'):
        s,args= reorg_naive(*reorg_default_args())
        f = tvm.build(s, args, 'llvm')
        a_np = reorg_input_generator(*reorg_default_args())
        A,B = args
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        f(a, b)
        b_np = reorg_numpy(a_np)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)
