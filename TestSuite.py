
import logging
import sys
from os import path,remove
import numpy as np
import tvm
from TestSuiteArgParser import test_suite_argparser as parser
from TestSuiteArgParser import build_parameters_from_args
from TestParameters import TestParameters
# the module is called `autotvm`
from tvm import autotvm


from TestableKernel import TestableKernel
from TestableKernel import testing_kernels as targets_dict
from VectorAdd import vectoradd_auto,vectoradd_manual,vectoradd_numpy,vectoradd_input_generator



class TestSuite:
    TESTING = 1
    HISTORY = 2
    CORRECTNESS = 4
    FULL = 8

    #define what operations should be performed under each mode
    TUNE_MASK = FULL | TESTING
    CORRECTNESS_MASK = FULL | TESTING | CORRECTNESS
    MEASURE_MASK = FULL | TESTING | HISTORY

    Instance = None

    def __init__(self, mode, logging_level = logging.ERROR):
        #This class relies on static dependencies (logging, logfiles) so it should be singleton for correctness
        if TestSuite.Instance != None:
            raise Exception("TestSuite is a singleton!")
        TestSuite.Instance = self
        self.mode = mode
        logging.getLogger('autotvm').setLevel(logging_level)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


    def load_test(self,testable_kernel,test_parameters):
        self.testable_kernel = testable_kernel
        self.test_parameters = test_parameters

    def _autotune_kernel(self,autokernel):
        params = self.test_parameters
        task = autotvm.task.create(autokernel, args=params.get_tvm_args(), target='llvm')

        print(task.config_space)
        measure_option = autotvm.measure_option(
            builder='local',
            runner=autotvm.LocalRunner(number=params.variance_resistance_runs))

        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=params.trial_runs,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(self.testable_kernel.logfile_name(autokernel))])


    def clear_log(testable_kernel,kernel):
        file_name = testable_kernel.logfile_name(kernel)
        if path.exists(file_name):
            remove(file_name)
            print("Cleared: " + file_name)
        else:
            print("Failed to load logfile: " + str(file_name))

    def print_log(testable_kernel,kernel):
        file_name = testable_kernel.logfile_name(kernel)
        if path.exists(file_name):
            for line in autotvm.record.load_from_file(file_name):
                print(autotvm.record.measure_str_key(line[0]))
        else:
            print("Failed to load logfile: " + str(file_name))

    def _report_measurement(self,testable_kernel,kernel):
        #From autotvm/task/dispatcher.py the criteria for defining a 'best' run is as follows:
        #                    if np.mean(other_res.costs) > np.mean(res.costs):



        file_name = self.testable_kernel.logfile_name(kernel)
        if (path.exists(file_name)):

            s,_ = kernel(*self.test_parameters.get_tvm_args())
            gflop =  autotvm.task.task.compute_flop(s) / 1e9

            context = autotvm.record.load_from_file(file_name)

            historical_runs = []
            best = (-100000,None)
            worst = (100000,None)

            for inp, res in context:
                avg_time = np.mean(res.costs)
                gflops = gflop/avg_time
                historical_runs.append(gflops)
                if(gflops > best[0]):
                    best = (gflops,inp)
                if(gflops < worst[0]):
                    worst = (gflops,inp)


            print("Loaded {0} records for {1} with average GFLOPS = {2}.".format(len(historical_runs),self.testable_kernel.kernel_name(kernel),np.mean(historical_runs)))

            print("The best schedule had gflops {0} with config {1}.".format(best[0],autotvm.record.measure_str_key(best[1])))
            print("The worst schedule had gflops {0} with config {1}.".format(worst[0],autotvm.record.measure_str_key(worst[1])))
        else:
            print("Failed to load logfile: " + str(file_name))

    def run(self):
        kernels = self.testable_kernel.get_tunable_kernels()

        def test_mask(mode,mask):
            return mode & mask != 0

        if(test_mask(self.mode,TestSuite.TUNE_MASK)):
            #Tune
            for kernel in kernels:
                print("Tuning: " + self.testable_kernel.kernel_name(kernel))
                self._autotune_kernel(kernel)

        if(test_mask(self.mode,TestSuite.CORRECTNESS_MASK)):
            #Test
            failed = False
            for kernel in kernels:
                print("Testing: " + self.testable_kernel.kernel_name(kernel))
                try:
                    self._test_correctness_of_best(kernel)
                except Exception as e:
                    print("Test failed for {0}!".format(self.testable_kernel.kernel_name(kernel)))
                    failed = True

            if not failed:
                print("All tests passed!")

        if(test_mask(self.mode,TestSuite.MEASURE_MASK)):
            #Measure
            for kernel in kernels:
                print("Results: " + self.testable_kernel.kernel_name(kernel))
                self._report_measurement(kernels,kernel)


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

    args = parser.parse_args()

    def run_testsuite(targets,mode):
        params_array = build_parameters_from_args(args)
        if(len(params_array) != len(targets)):
            print("Dimensions were not formatted correctly for {0} kernels.".format(len(targets_dict.values())))
            sys.exit(0)

        suite = TestSuite(mode = mode)

        for i in range(len(targets)):
            target = targets[i]
            test_kernel = targets_dict[target]    
            suite.load_test(testable_kernel = test_kernel, test_parameters = params_array[i])
            suite.run()

    if(args.DeleteLogsTargets != None):        
        for target in args.DeleteLogsTargets:
            test_kernel = targets_dict[target]
            for kernel in test_kernel.get_tunable_kernels():
                TestSuite.clear_log(test_kernel,kernel)

    elif(args.IsTestRun):
        test_kernel = TestableKernel(auto_autokernel = vectoradd_auto,
            manual_autokernel =vectoradd_manual,
            numpy_kernel = vectoradd_numpy,
            input_generator = vectoradd_input_generator,
            name ="vectoradd")

        suite = TestSuite(mode = TestSuite.TESTING)
        suite.load_test(testable_kernel = test_kernel, test_parameters = build_parameters_from_args(args)[0])
        suite.run()

    elif(args.HistoryRunTargets != None):
        run_testsuite(args.HistoryRunTargets,TestSuite.HISTORY)
    elif(args.CorrectnessRunTargets != None):
        run_testsuite(args.CorrectnessRunTargets,TestSuite.CORRECTNESS)
    elif(args.FullRunTargets != None):
        run_testsuite(args.FullRunTargets,TestSuite.FULL)
    else:
        print("No mode selected. Exiting")
        sys.exit(0)