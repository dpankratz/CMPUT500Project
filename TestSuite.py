
import logging
import sys
from os import path,remove
import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from TestableKernel import TestableKernel

class TestSuite:
    Instance = None

    def __init__(self,testable_kernel,test_parameters,clear_logs = True):
        #This class relies on static dependencies (logging, logfiles) so it should be singleton for correctness
        if Instance != None:
            raise Exception("TestSuite is a singleton!")
        TestSuite.Instance = self
        self.testable_kernel = testable_kernel
        self.test_parameters = test_parameters
        if(clear_logs)
            #logs are used to maintain the history of previous tuning. Thus clearing logs removes the knowledge of the best previous configuration
            self._clear_logs()
        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


    def _autotune_kernel(self,autokernel):
        task = autotvm.task.create(autokernel, args=test_parameters.get_tvm_args(), target='llvm')

        measure_option = test_parameters.get_measure_option()

        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=test_parameters.trials_runs,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(testable_kernel.logfile_name(autokernel))])


    def _clear_logs(self):
        kernels = self.testable_kernel
        for kernel in kernels.get_tunable_kernels():
            file_name = kernels.logfile_name(kernel)
            if path.exists(file_name):
                remove(file_name)
                print("Cleared " + file_name)

    def run(self):
        params = self.test_parameters
        kernels = self.testable_kernel.get_tunable_kernels

        for kernel in kernels:
            _autotune_kernel(kernel)

        for kernel in kernels:
            _test_correctness_of_best(kernel)

        
    def _test_correctness_of_best(self,kernel):
        #it is assumed that the numpy implementation is canonical
        params = self.test_parameters
        kernels = self.testable_kernel
        with autotvm.apply_history_best(kernels.logfile_name(kernel)):
            with tvm.target.create("llvm"):
                s, arg_bufs = kernel(params.get_tvm_args())
                runnable_kernel = tvm.build(s, arg_bufs)

        inputs_np = kernels.input_generator()

        canoncial_result = kernels.numpy_kernel(*inputs_np)

        workingset_tvm = tvm.nd.empty(canoncial_result.shape)

        inputs_tvm = map(tvm.nd.array,inputs_np)

        runnable_kernel(*inputs_tvm, tvm_placeholder)

        tvm.testing.assert_allclose(canoncial_result, workingset_tvm.asnumpy(), rtol=1e-2)



if __name__ == "__main__":
    test_kernel = TestableKernel(auto_autokernel = vectoradd_auto,
        manual_autokernel =vectoradd_manual,
        numpy_kernel = vectoradd_numpy,
        name ="vectoradd")
    test_parameters = TestParameters(variance_resistance_runs = 5, trial_runs = 20,dims = [4096],dtype = 'float32')

    suite = TestSuite(testable_kernel = test_kernel, 
        test_parameters = test_parameters)

    suite.run()
