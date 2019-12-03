
import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from VectorAdd import vectoradd_auto,vectoradd_numpy,vectoradd_input_generator,vectoradd_naive,vectoradd_default_args
from GEMM import matmul_moderate,matmul_numpy,matmul_input_generator,matmul_naive,matmul_default_args,matmul_conservative
from ConvNCHW import conv_moderate,conv_numpy,conv_input_generator,conv_naive,conv_default_args,conv_conservative
import Upsample
import softmax
import resize
import relu
import reorg

class TestableKernel:

    def __init__(self,autokernels,numpy_kernel,input_generator,default_args,name):
        self.autokernels = autokernels
        self.numpy_kernel = numpy_kernel
        self.input_generator = input_generator
        self.default_args = default_args
        self.name = name

    def get_tunable_kernels(self,):
        return self.autokernels
        

testing_kernels = {'VectorAdd.py' : 
    TestableKernel(autokernels = [vectoradd_auto,vectoradd_naive],
        numpy_kernel = vectoradd_numpy,
        input_generator = vectoradd_input_generator,
        default_args = vectoradd_default_args,
        name ="vectoradd"),
        'GEMM.py':
    TestableKernel(autokernels = [matmul_moderate, matmul_naive, matmul_conservative],
        numpy_kernel = matmul_numpy,
        input_generator = matmul_input_generator,
        default_args= matmul_default_args,
        name ="matmul"),
    'ConvNCHW.py':
    TestableKernel(autokernels = [conv_moderate, conv_naive,conv_conservative],
        numpy_kernel = conv_numpy,
        input_generator = conv_input_generator,
        default_args= conv_default_args,
        name ="conv")}

results_kernels = {    'softmax.py':TestableKernel(autokernels = [softmax.softmax_naive,softmax.softmax_moderate,  softmax.softmax_conservative],
        numpy_kernel = softmax.softmax_numpy,
        input_generator = softmax.softmax_input_generator,
        default_args = softmax.softmax_default_args,
        name = "softmax"),
    'resize.py' : TestableKernel(autokernels = [resize.resize_naive,resize.resize_moderate,  resize.resize_conservative],
        numpy_kernel = resize.resize_numpy,
        input_generator = resize.resize_input_generator,
        default_args = resize.resize_default_args,
        name = "resize"),
    'Upsample.py' :
    TestableKernel(autokernels = [ Upsample.upsample_naive, Upsample.upsample_moderate, Upsample.upsample_conservative],
        numpy_kernel = Upsample.upsample_numpy,
        input_generator = Upsample.upsample_input_generator,
        default_args = Upsample.upsample_default_args,
        name = "upsample"),
    'relu.py' :
    TestableKernel(autokernels = [relu.relu_naive, relu.relu_moderate,  relu.relu_conservative],
        numpy_kernel = relu.relu_numpy,
        input_generator = relu.relu_input_generator,
        default_args = relu.relu_default_args,
        name = "relu"),
    'reorg.py':
        TestableKernel(autokernels = [reorg.reorg_naive, reorg.reorg_moderate,  reorg.reorg_conservative],
        numpy_kernel = reorg.reorg_numpy,
        input_generator = reorg.reorg_input_generator,
        default_args = reorg.reorg_default_args,
        name = "reorg")}