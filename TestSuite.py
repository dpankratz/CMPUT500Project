
import logging
import sys
from os import path,remove
import numpy as np
import tvm
from TestParameters import TestParameters
# the module is called `autotvm`
from tvm import autotvm
from TestableKernel import TestableKernel
from VectorAdd import vectoradd_auto,vectoradd_manual,vectoradd_numpy,vectoradd_input_generator

class TestSuite:
    Instance = None

    def __init__(self,testable_kernel,test_parameters,clear_logs = True, logging_level = logging.WARN):
        #This class relies on static dependencies (logging, logfiles) so it should be singleton for correctness
        if TestSuite.Instance != None:
            raise Exception("TestSuite is a singleton!")
        TestSuite.Instance = self
        self.testable_kernel = testable_kernel
        self.test_parameters = test_parameters
        if(clear_logs):
            #logs are used to maintain the history of previous tuning. Thus clearing logs removes the knowledge of the best previous configuration
            self._clear_logs()
        logging.getLogger('autotvm').setLevel(logging_level)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


    def _autotune_kernel(self,autokernel):
        task = autotvm.task.create(autokernel, args=test_parameters.get_tvm_args(), target='llvm')

        params = self.test_parameters

        measure_option = autotvm.measure_option(
            builder='local',
            runner=autotvm.LocalRunner(number=params.variance_resistance_runs))

        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=params.trial_runs,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(self.testable_kernel.logfile_name(autokernel))])


    def _clear_logs(self):
        kernels = self.testable_kernel
        for kernel in kernels.get_tunable_kernels():
            file_name = kernels.logfile_name(kernel)
            if path.exists(file_name):
                remove(file_name)
                print("Cleared: " + file_name)

    def run(self):
        params = self.test_parameters
        kernels = self.testable_kernel.get_tunable_kernels()

        for kernel in kernels:
            print("Tuning: " + self.testable_kernel.kernel_name(kernel))
            self._autotune_kernel(kernel)

        for kernel in kernels:
            print("Testing: " + self.testable_kernel.kernel_name(kernel))
            self._test_correctness_of_best(kernel)

        print("All tests passed.")

        #TODO get measurements

        
    def _test_correctness_of_best(self,kernel):
        #it is assumed that the numpy implementation is canonical
        params = self.test_parameters
        kernels = self.testable_kernel
        with autotvm.apply_history_best(kernels.logfile_name(kernel)):
            with tvm.target.create("llvm"):
                s, arg_bufs = kernel(*params.get_tvm_args())
                runnable_kernel = tvm.build(s, arg_bufs)

        inputs_np = kernels.input_generator(*params.get_tvm_args())

        canoncial_result = kernels.numpy_kernel(*inputs_np)

        workingset_tvm = tvm.nd.empty(canoncial_result.shape)

        inputs_tvm = list(map(tvm.nd.array,inputs_np))

        runnable_kernel(*inputs_tvm, workingset_tvm)

        tvm.testing.assert_allclose(canoncial_result, workingset_tvm.asnumpy(), rtol=1e-2)



if __name__ == "__main__":
    test_kernel = TestableKernel(auto_autokernel = vectoradd_auto,
        manual_autokernel =vectoradd_manual,
        numpy_kernel = vectoradd_numpy,
        input_generator = vectoradd_input_generator,
        name ="vectoradd")
    test_parameters = TestParameters(variance_resistance_runs = 5, trial_runs = 20,dims = [12],dtype = 'float32')

    suite = TestSuite(testable_kernel = test_kernel, 
        test_parameters = test_parameters, clear_logs=False)

    suite.run()
