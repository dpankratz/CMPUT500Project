
import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from VectorAdd import vectoradd_auto,vectoradd_manual,vectoradd_numpy,vectoradd_input_generator

class TestableKernel:

    def __init__(self,auto_autokernel, manual_autokernel,numpy_kernel,input_generator,name):
        self.auto_autokernel = auto_autokernel
        self.manual_autokernel = manual_autokernel
        self.numpy_kernel = numpy_kernel
        self.input_generator = input_generator
        self.name = name

    def logfile_name(self,kernel):
        if(kernel == self.auto_autokernel):
            return self.name + "_auto.log"
        elif(kernel == self.manual_autokernel):
            return self.name + "_manual.log"

    def get_tunable_kernels():
        return [self.auto_autokernel,self.manual_autokernel]
        

if __name__ == "__main__":
    t = TestableKernel(auto_autokernel = vectoradd_auto,
        manual_autokernel =vectoradd_manual,
        numpy_kernel = vectoradd_numpy,
        input_generator = vectoradd_input_generator
        name ="vectoradd")
    print(t.logfile_name(t.auto_autokernel))
    print(t.logfile_name(t.manual_autokernel))
    
    