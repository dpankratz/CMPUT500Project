
import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from VectorAdd import vectoradd_auto,vectoradd_numpy,vectoradd_input_generator,vectoradd_naive,vectoradd_default_args
from GEMM import matmul_auto,matmul_numpy,matmul_input_generator,matmul_naive,matmul_default_args
from ConvNCHW import conv_auto,conv_numpy,conv_input_generator,conv_naive,conv_default_args

class TestableKernel:

    def __init__(self,auto_autokernel, naive_autokernel,numpy_kernel,input_generator,default_args,name):
        self.auto_autokernel = auto_autokernel
        self.naive_autokernel = naive_autokernel
        self.numpy_kernel = numpy_kernel
        self.input_generator = input_generator
        self.default_args = default_args
        self.name = name

    def kernel_name(self,kernel):
        if(kernel == self.auto_autokernel):
            return self.name + "_auto"
        elif(kernel == self.naive_autokernel):
            return self.name + "_naive"

    def logfile_name(self,kernel):
        return self.kernel_name(kernel) + ".log"

    def get_tunable_kernels(self,):
        return [self.auto_autokernel,self.naive_autokernel]
        



testing_kernels = {'VectorAdd.py' : 
    TestableKernel(auto_autokernel = vectoradd_auto,
        naive_autokernel = vectoradd_naive,
        numpy_kernel = vectoradd_numpy,
        input_generator = vectoradd_input_generator,
        default_args = vectoradd_default_args,
        name ="vectoradd"),
        'GEMM.py':
    TestableKernel(auto_autokernel = matmul_auto,
        naive_autokernel = matmul_naive,
        numpy_kernel = matmul_numpy,
        input_generator = matmul_input_generator,
        default_args= matmul_default_args,
        name ="matmul"),
    'ConvNCHW.py':
    TestableKernel(auto_autokernel = conv_auto,
        naive_autokernel = conv_naive,
        numpy_kernel = conv_numpy,
        input_generator = conv_input_generator,
        default_args= conv_default_args,
        name ="conv")}

if __name__ == "__main__":
    t = TestableKernel(auto_autokernel = vectoradd_auto,
        numpy_kernel = vectoradd_numpy,
        input_generator = vectoradd_input_generator,
        name ="vectoradd")
    assert(t.logfile_name(t.auto_autokernel) == "vectoradd_auto.log")
    assert(t.logfile_name(t.manual_autokernel) == "vectoradd_manual.log")
    assert(t.get_tunable_kernels() == [vectoradd_auto,vectoradd_manual])
    