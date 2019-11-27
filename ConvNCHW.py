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

#    input_shape = (batch_size, 3, 224, 224)
#    output_shape = (batch_size, 1000)
@autotvm.template
def conv_auto(batch, in_channel, in_size, num_filter, kernel):
    global B
    in_height = in_width = in_size

    A = tvm.placeholder(_a_shape(batch,in_channel,in_size), name='A')
    W = tvm.placeholder(_w_shape(kernel,in_channel,num_filter), name='W')
    B = topi.nn.conv2d_nhwc(A, W, 3, "SAME", 1)
    s = tvm.create_schedule(B.op)

    #### ADDED AUTO AUTOTUNING ####
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.NONNAIVE)
    ###############################

    dtype = A.dtype

    return s,[A,W,B]

@autotvm.template
def conv_naive(batch, in_channel, in_size, num_filter, kernel):
    global B
    in_height = in_width = in_size

    A = tvm.placeholder(_a_shape(batch,in_channel,in_size), name='A')
    W = tvm.placeholder(_w_shape(kernel,in_channel,num_filter), name='W')
    B = topi.nn.conv2d_nhwc(A, W, 3, "SAME", 1)
    s = tvm.create_schedule(B.op)

    #### ADDED AUTO AUTOTUNING ####
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.NAIVE)
    ###############################

    dtype = A.dtype

    return s,[A,W,B]


def conv_numpy(a_np,w_np):
    dw_np = topi.testing.dilate_python(w_np, (1, 1, 1, 1))
    b_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, 3, "SAME")
    return b_np

def conv_input_generator(batch, in_channel, in_size, num_filter, kernel, dilation=1):
    a_shape = get_const_tuple(_a_shape(batch, in_channel, in_size))
    w_shape = get_const_tuple(_w_shape(kernel,in_channel,num_filter))
    a_np = np.random.uniform(size=a_shape).astype('float32')
    w_np = np.random.uniform(size=w_shape).astype('float32')
    return a_np, w_np

def _a_shape(batch, in_channel, in_size):
    in_height = in_width = in_size
    return (batch,in_height,in_width,in_channel)

def _w_shape(kernel,in_channel,num_filter):
    return (kernel,kernel,in_channel,num_filter)

def conv_default_args():
    batch = 64
    in_channel = 128
    in_size = 16
    num_filter = 128
    kernel = 3
    return (batch,in_channel,in_size,num_filter,kernel)

if __name__ == "__main__":
    
    args = conv_default_args()


    task = autotvm.task.create(conv_auto, args=args, target='llvm')

    print("Creating task")

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=1))
    print("begin tuning")
    tuner = autotvm.tuner.RandomTuner(task)
    print(tuner.tune(n_trial=1,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file('conv.log')]))

    with autotvm.apply_history_best('conv.log'):
        with tvm.target.create("llvm"):
            s, arg_bufs = conv_auto(*conv_default_args())
            func = tvm.build(s, arg_bufs)

    a_np,w_np = conv_input_generator(*args)
    b_np = conv_numpy(a_np,w_np)
    
    ctx = tvm.context("llvm", 0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func(a,w,b)
    

    tvm.testing.assert_allclose(b_np, b.asnumpy(), rtol=1e-5)
