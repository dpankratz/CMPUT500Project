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
"""Test code for softmax"""
import os
import numpy as np
import tvm
import topi
import topi.testing
import logging
import Passes
from tvm import autotvm
from topi.util import get_const_tuple


@autotvm.template
def softmax_moderate(a,b,c,d):
    A = tvm.placeholder((a,b,c,d), dtype='float32', name='A')
    B = topi.nn.softmax(A,axis=1)
    s = tvm.create_schedule([B.op])
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.NONNAIVE)
    
    return s,[A,B]

@autotvm.template
def softmax_conservative(a,b,c,d):
    A = tvm.placeholder((a,b,c,d), dtype='float32', name='A')
    B = topi.nn.softmax(A,axis=1)
    s = tvm.create_schedule([B.op])
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.CONSERVATIVE)
    
    return s,[A,B]

@autotvm.template
def softmax_naive(a,b,c,d):
    A = tvm.placeholder((a,b,c,d), dtype='float32', name='A')
    B = topi.nn.softmax(A,axis=1)
    s = tvm.create_schedule([B.op])
    Passes.enable_autotune(s,[B],autotvm.get_config(),mode=Passes.NAIVE)
    
    return s,[A,B]


def softmax_default_args():
    return (1, 16, 256, 256)

def softmax_input_generator(a,b,c,d):
    a_np = np.random.uniform(size=get_const_tuple((a,b,c,d))).astype('float32')
    return [a_np]

def softmax_numpy(a_np):
    _, c, h, w = softmax_default_args()
    b_np = topi.testing.softmax_python(a_np.transpose(0, 2, 3, 1).reshape(h*w, c))
    b_np = b_np.reshape(1, h, w, c).transpose(0, 3, 1, 2)
    return b_np


if __name__ == "__main__":

    ctx = tvm.context('llvm', 0)

    batch,in_channel,in_height,in_width = softmax_default_args()

    with tvm.target.create('llvm'):
        s,c= softmax_auto(*softmax_default_args())
        A,B = c
        f = tvm.build(s, [A,B], 'llvm')
        a_np = softmax_input_generator(*softmax_default_args())[0]
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        f(a, b)
        b_np = softmax_numpy(a_np)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)
