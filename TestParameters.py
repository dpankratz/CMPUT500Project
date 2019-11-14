
import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm

class TestParameters:

    def __init__(self, variance_resistance_runs,trial_runs, dims, dtype):
        self.variance_resistance_runs = variance_resistance_runs
        self.trial_runs = trial_runs
        self.dims = dims
        self.dtype = dtype

    def get_measure_option(self):   
        return autotvm.LocalRunner(number = self.variance_resistance_runs)

    def numpy_dtype(self):
        if(self.dtype == "float32"):
            return np.float32

    def get_tvm_args(self):
        return tuple(self.dims) + (self.dtype,)

if(__name__ == "__main__"):
    t = TestParameters(variance_resistance_runs = 5, trial_runs = 20,dims = [512,512,512],dtype = 'float32')
    assert(type(t.get_measure_option()) == tvm.autotvm.measure.measure_methods.LocalRunner)
    assert(t.numpy_dtype() == np.float32)
    assert(t.get_tvm_args() == (512,512,512,'float32'))