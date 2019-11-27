
import logging
import sys

import numpy as np
import tvm
from tvm import autotvm

import Passes

@autotvm.template
def vectoradd_auto(K,dtype):
    A = tvm.placeholder((K,), name='A', dtype=dtype)
    B = tvm.placeholder((K,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule(C.op)

    #### ADDED AUTO AUTOTUNING ####
    Passes.enable_autotune(s,[C],autotvm.get_config(),mode=Passes.NONNAIVE)
    ###############################

    return s, [A, B, C]

@autotvm.template
def vectoradd_naive(K,dtype):
    A = tvm.placeholder((K,), name='A', dtype=dtype)
    B = tvm.placeholder((K,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule(C.op)

    #### ADDED AUTO AUTOTUNING ####
    Passes.enable_autotune(s,[C],autotvm.get_config(),mode=Passes.NAIVE)
    ###############################

    return s, [A, B, C]


@autotvm.template
def vectoradd_manual(K,dtype):
    A = tvm.placeholder((K,), name='A', dtype=dtype)
    B = tvm.placeholder((K,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule(C.op)

    a, = s[C].op.axis
    
    cfg = autotvm.get_config()
    cfg.define_split("tile_0", a, num_outputs=2)
    yo, yi = cfg["tile_0"].apply(s, C, a)
    cfg.define_reorder("reorder",[yo,yi],policy = 'all')

    return s, [A, B, C]

def vectoradd_numpy(a_np,b_np):
    c_np = a_np + b_np 
    return c_np

def vectoradd_input_generator(K,dtype):
    a_np = np.random.uniform(size=(K)).astype(dtype)
    b_np = np.random.uniform(size=(K)).astype(dtype)
    return [a_np,b_np]

def vectoradd_default_args():
    return (65536,'float32')

if __name__ == "__main__":

    K = 4096


    task = autotvm.task.create(vectoradd_manual, args=(K, 'float32'), target='llvm')

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=5))

    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=20,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file('VA.log')])

    with autotvm.apply_history_best('VA.log'):
        with tvm.target.create("llvm"):
            s, arg_bufs = vectoradd_manual(K, 'float32')
            func = tvm.build(s, arg_bufs)

    a_np = np.random.uniform(size=(K)).astype(np.float32)
    b_np = np.random.uniform(size=(K)).astype(np.float32)
    c_np = vectoradd_numpy(a_np,b_np)

    c_tvm = tvm.nd.empty(c_np.shape)
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

