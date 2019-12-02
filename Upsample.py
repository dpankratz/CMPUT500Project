import numpy as np
import tvm
import topi
import topi.testing
import math
import Passes
from tvm import autotvm


@autotvm.template
def upsample_auto(batch,in_channel,in_height,in_width):
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    out_shape = (batch, in_channel, in_height*3, in_width*3)
    B = topi.nn.upsampling(A, 3, layout='NCHW', method='nearest_neighbor', align_corners=False)
    with tvm.target.create('llvm'):
        s = tvm.create_schedule(B.op)


    cfg = autotvm.get_config()
    Passes.enable_autotune(s,[B],cfg,mode=Passes.CONSERVATIVE)


    return s,[A,B]



    

def upsample_input_generator(batch,in_channel,in_height,in_width):
    a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype('float32')
    return a_np

def upsample_numpy(a_np):
    b_np = topi.testing.upsampling_python(a_np, (3, 3), 'NCHW')
    return b_np

def upsample_default_args():
    return (8,16,64,32)

if __name__ == "__main__":

    ctx = tvm.context('llvm', 0)

    batch,in_channel,in_height,in_width = upsample_default_args()

    with tvm.target.create('llvm'):
        s,a= upsample_auto(*upsample_default_args())
        f = tvm.build(s, a, 'llvm')
        a_np = upsample_input_generator(*upsample_default_args())
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros((batch, in_channel, in_height*3, in_width*3), dtype='float32'), ctx)
        f(a, b)
        b_np = upsample_numpy(a_np)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)
