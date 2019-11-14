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

from Passes import enable_autotune

@autotvm.template
def matmul_auto(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    #### ADDED AUTO AUTOTUNING ####
    enable_autotune(s,[C],autotvm.get_config())
    ###############################

    return s, [A, B, C]

@autotvm.template
def matmul_manual(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    a, b = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    
    cfg = autotvm.get_config()
    cfg.define_split("tile_0", a, num_outputs=2)
    cfg.define_split("tile_1", b, num_outputs=2)
    ##### define space end #####


    # schedule according to config
    yo, yi = cfg["tile_0"].apply(s, C, a)
    xo, xi = cfg["tile_1"].apply(s, C, b)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


################################################################
# Begin tuning
# ^^^^^^^^^^^^
# Here we continue our matrix multiplication example.
# First we should create a tuning task.
# We can also inspect the initialized search space.
# In this case, for a 512x512 square matrix multiplication

N, L, M = 512, 512, 512
task = autotvm.task.create(matmul_manual, args=(N, L, M, 'float32'), target='llvm')

################################################################
# Then we need to define how to measure the generated code and pick a tuner.
#
# We will log the tuning results into a log file. This file can be
# used to get the best config later.

# logging config (for printing tuning log to the screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure k times and take average to reduce variance.
measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# begin tuning, log records to file `GEMM.log`
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=20,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('GEMM.log')])

#########################################################################
# Finally we apply history best from the cache file and check its correctness.
# We can call the function :code:`matmul` directly under the
# :any:`autotvm.apply_history_best` context. When we call this function,
# it will query the dispatch context with its argument and get the best config
# with the same argument.

# apply history best from log file
with autotvm.apply_history_best('GEMM.log'):
    with tvm.target.create("llvm"):
        s, arg_bufs = matmul_manual(N, L, M, 'float32')
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
