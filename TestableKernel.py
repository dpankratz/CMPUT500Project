
import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from VectorAdd import vectoradd_auto,vectoradd_manual,vectoradd_numpy,vectoradd_input_generator
from GEMM import matmul_auto,matmul_manual,matmul_numpy,matmul_input_generator

class TestableKernel:

    def __init__(self,auto_autokernel, manual_autokernel,numpy_kernel,input_generator,name):
        self.auto_autokernel = auto_autokernel
        self.manual_autokernel = manual_autokernel
        self.numpy_kernel = numpy_kernel
        self.input_generator = input_generator
        self.name = name

    def kernel_name(self,kernel):
        if(kernel == self.auto_autokernel):
            return self.name + "_auto"
        elif(kernel == self.manual_autokernel):
            return self.name + "_manual"

    def logfile_name(self,kernel):
        return self.kernel_name(kernel) + ".log"

    def get_tunable_kernels(self,):
        return [self.auto_autokernel,self.manual_autokernel]
        



testing_kernels = {'VectorAdd.py' : 
    TestableKernel(auto_autokernel = vectoradd_auto,
        manual_autokernel =vectoradd_manual,
        numpy_kernel = vectoradd_numpy,
        input_generator = vectoradd_input_generator,
        name ="vectoradd"),
        'GEMM.py':
    TestableKernel(auto_autokernel = matmul_auto,
        manual_autokernel =matmul_manual,
        numpy_kernel = matmul_numpy,
        input_generator = matmul_input_generator,
        name ="matmul")}

if __name__ == "__main__":
    t = TestableKernel(auto_autokernel = vectoradd_auto,
        manual_autokernel =vectoradd_manual,
        numpy_kernel = vectoradd_numpy,
        input_generator = vectoradd_input_generator,
        name ="vectoradd")
    assert(t.logfile_name(t.auto_autokernel) == "vectoradd_auto.log")
    assert(t.logfile_name(t.manual_autokernel) == "vectoradd_manual.log")
    assert(t.get_tunable_kernels() == [vectoradd_auto,vectoradd_manual])
    