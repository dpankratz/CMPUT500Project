import numpy as np
import tvm
import topi
import topi.testing
import math
import Passes
from tvm import autotvm
from topi.util import get_const_tuple


@autotvm.template
def relu_conservative(m,n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.nn.relu(A)
    with tvm.target.create('llvm'):
        s = tvm.create_schedule(B.op)


    cfg = autotvm.get_config()

    Passes.enable_autotune(s,[B],cfg,mode=Passes.CONSERVATIVE)


    return s,[A,B]

@autotvm.template
def relu_moderate(m,n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.nn.relu(A)
    with tvm.target.create('llvm'):
        s = tvm.create_schedule(B.op)


    cfg = autotvm.get_config()

    Passes.enable_autotune(s,[B],cfg,mode=Passes.NONNAIVE)


    return s,[A,B]

@autotvm.template
def relu_naive(m,n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.nn.relu(A)
    with tvm.target.create('llvm'):
        s = tvm.create_schedule(B.op)


    cfg = autotvm.get_config()

    Passes.enable_autotune(s,[B],cfg,mode=Passes.NAIVE)


    return s,[A,B]



    

def relu_input_generator(m,n):
    a_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple((m,n))).astype('float32')
    return [a_np]

def relu_numpy(a_np):
    b_np = a_np * (a_np > 0)
    return b_np

def relu_default_args():
    return (1024 * 100 , 512)

if __name__ == "__main__":

    ctx = tvm.context('llvm', 0)


    with tvm.target.create('llvm'):
        s,c= relu_auto(*relu_default_args())
        A,B = c
        a_np = relu_input_generator(*relu_default_args())[0]
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        foo = tvm.build(s, [A, B], 'llvm', name="relu")
        foo(a, b)
        b_np = relu_numpy(a_np)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)
