
import logging
import sys
from os import path,remove,mkdir
from datetime import datetime
import numpy as np
import tvm
import shutil
from TestSuiteArgParser import test_suite_argparser as parser
from TestSuiteArgParser import build_parameters_from_args
from TestParameters import TestParameters
# the module is called `autotvm`
from tvm import autotvm
from tvm.autotvm import feature
from tvm.autotvm.util import get_func_name
import TestSuiteGraphs

from TestableKernel import TestableKernel
from TestableKernel import testing_kernels as targets_dict



class TestSuite:
    OLD_LOG_FILES_DIR = 'archived_logs'
    INFO_FILE_PREFIX = 'info_'

    TESTING = 1
    HISTORY = 2
    CORRECTNESS = 4
    FULL = 8
    PRINT_CONFIG = 16

    #define what operations should be performed under each mode
    TUNE_MASK = FULL | TESTING
    CORRECTNESS_MASK = FULL | TESTING | CORRECTNESS
    MEASURE_MASK = FULL | TESTING | HISTORY
    PRINT_CONFIG_MASK = PRINT_CONFIG
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
        task = autotvm.task.create(autokernel, args=self._kernel_args(autokernel), target='llvm')

        TestSuite._write_to_infofile(autokernel,TestSuite._config_space_info(task.config_space),'w')

        print(task.config_space)
        measure_option = autotvm.measure_option(
            builder='local',
            runner=autotvm.LocalRunner(number=params.variance_resistance_runs))

        tuner = autotvm.tuner.XGBTuner(task)
        tuner.tune(n_trial=params.trial_runs,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(TestSuite._logfile_path(autokernel))])


    def clear_log(testable_kernel,kernel):
        file_name = TestSuite._logfile_path(kernel)
        if not path.exists(TestSuite.OLD_LOG_FILES_DIR):
            mkdir(TestSuite.OLD_LOG_FILES_DIR)
        if path.exists(file_name):

            timestr = str(datetime.now()).replace(' ','_')

            shutil.move(file_name, TestSuite.OLD_LOG_FILES_DIR +"/" + timestr[1:len(timestr)-1] + '_' + file_name)
            print("Cleared: " + file_name)
        else:
            print("Failed to load logfile: " + str(file_name))

    def print_config_space(self,autokernel):
        params = self.test_parameters
        task = autotvm.task.create(autokernel, args=self._kernel_args(autokernel), target='llvm')
        print(task.config_space)

    def print_log(testable_kernel,kernel):
        file_name = _logfile_name(kernel)
        if path.exists(file_name):
            for line in autotvm.record.load_from_file(file_name):
                print(autotvm.record.measure_str_key(line[0]))
        else:
            print("Failed to load logfile: " + str(file_name))

    def _report_measurement(self,testable_kernel,kernel):
        #From autotvm/task/dispatcher.py the criteria for defining a 'best' run is as follows:
        #                    if np.mean(other_res.costs) > np.mean(res.costs):



        file_name = TestSuite._logfile_path(kernel)
        if (path.exists(file_name)):

            gflop = 0
            s,a = kernel(*self._kernel_args(kernel))
            if(TestSuite.kernel_name(kernel).startswith("upsample_")):
                gflop = 2 ** 24 / 1e9
            else:
                gflop =  autotvm.task.task.compute_flop(s) / 1e9

            context = autotvm.record.load_from_file(file_name)
            annotated_points = []
            historical_runs = []
            best = (-100000,None)
            worst = (100000,None)
            i = 0
            for inp, res in context:
                avg_time = np.mean(res.costs)
                gflops = gflop/avg_time                
                if(gflops > best[0]):
                    best = (gflops,inp)
                    annotated_points.append((i,gflops,autotvm.record.measure_str_key(inp)))
                if(gflops < worst[0]):
                    worst = (gflops,inp)
                historical_runs.append(best[0])
                i += 1



            print("Loaded {0} records for {1}.".format(len(historical_runs),TestSuite.kernel_name(kernel)))

            print("The best schedule had gflops {0} with config {1}.".format(best[0],autotvm.record.measure_str_key(best[1])))
            print("The worst schedule had gflops {0} with config {1}.".format(worst[0],autotvm.record.measure_str_key(worst[1])))

            return historical_runs,annotated_points

        else:
            print("Failed to load logfile: " + str(file_name))

    def run(self):
        kernels = self.testable_kernel.get_tunable_kernels()

        def test_mask(mode,mask):
            return mode & mask != 0

        if(test_mask(self.mode,TestSuite.PRINT_CONFIG_MASK)):
            for kernel in kernels:
                print("Config: " + TestSuite.kernel_name(kernel))
                self.print_config_space(kernel)

        if(test_mask(self.mode,TestSuite.TUNE_MASK)):
            #Tune
            for kernel in kernels:
                print("Tuning: " + TestSuite.kernel_name(kernel))
                tune_start = datetime.now()
                self._autotune_kernel(kernel)
                tune_time = datetime.now() - tune_start
                print("Tuning time: ",tune_time)
                TestSuite._write_to_infofile(kernel,"tunetime={0}\n".format(tune_time),"a")


        if(test_mask(self.mode,TestSuite.CORRECTNESS_MASK)):
            #Test
            failed = False
            for kernel in kernels:
                print("Testing: " + TestSuite.kernel_name(kernel))
                try:
                    self._test_correctness_of_best(kernel)
                except Exception as e:
                    print("Test failed for {0}! Error message: {1}.".format(TestSuite.kernel_name(kernel),e))
                    failed = True

            if not failed:
                print("All tests passed!")

        if(test_mask(self.mode,TestSuite.MEASURE_MASK)):
            #Measure
            points = []
            measurements = []
            labels = []
            for kernel in kernels:
                print("Results: " + TestSuite.kernel_name(kernel))
                gflops, annotated_points = (self._report_measurement(kernels,kernel))
                labels.append(TestSuite.kernel_name(kernel))
                measurements.append(gflops)
                points.append(annotated_points)

            TestSuiteGraphs.plot_gflops(measurements,points,labels)

    def _test_correctness_of_best(self,kernel):
        #it is assumed that the numpy implementation is canonical
        params = self.test_parameters
        kernels = self.testable_kernel
        with autotvm.apply_history_best(TestSuite._logfile_path(kernel)):
            with tvm.target.create("llvm"):
                s, arg_bufs = kernel(*self._kernel_args(kernel))
                runnable_kernel = tvm.build(s, arg_bufs)


        inputs_np = kernels.input_generator(*self._kernel_args(kernel))
        canoncial_result = kernels.numpy_kernel(*inputs_np)
        workingset_tvm = tvm.nd.empty(canoncial_result.shape)
        inputs_tvm = list(map(tvm.nd.array,inputs_np))
        runnable_kernel(*inputs_tvm, workingset_tvm)
        tvm.testing.assert_allclose(canoncial_result, workingset_tvm.asnumpy(), rtol=1e-2)

    def _kernel_args(self,kernel):
        if(self.test_parameters.dims == None):
            return self.testable_kernel.default_args()
        return self.test_parameters.get_tvm_args()
        
            
        return args

    def kernel_name(kernel):
        return get_func_name(kernel)

    def _logfile_path(kernel):
        return TestSuite.kernel_name(kernel) + ".log"

    def infofile_path(kernel):
        return TestSuite.INFO_FILE_PREFIX + TestSuite._logfile_path(kernel)

    def _write_to_infofile(kernel,string,mode):
        
        info_file = open(TestSuite.infofile_path(kernel),mode)
        info_file.write(string)
        info_file.close()

    def _config_space_info(space):
        s = "len={0}\n".format(len(space))
        for (name,space) in space.space_map.items():
            s += "{0}:{1}\n".format(name,space.entities)
        return s


if __name__ == "__main__":

    args = parser.parse_args()

    def run_testsuite(targets,mode):
        params = build_parameters_from_args(args)
        if(type(params) == list):
            if(len(params) != len(targets)):
                print("Dimensions were not formatted correctly for {0} kernels.".format(len(targets_dict.values())))
                sys.exit(0)

        suite = TestSuite(mode = mode)

        for i in range(len(targets)):
            target = targets[i]
            test_kernel = targets_dict[target]    
            suite.load_test(testable_kernel = test_kernel, test_parameters = params[i] if type(params) == list else params)
            suite.run()



    if(args.DeleteLogsTargets != None):        
        for target in args.DeleteLogsTargets:
            test_kernel = targets_dict[target]
            for kernel in test_kernel.get_tunable_kernels():
                TestSuite.clear_log(test_kernel,kernel)

    elif(args.IsTestRun):
        test_kernel = targets_dict['VectorAdd.py']

        suite = TestSuite(mode = TestSuite.TESTING)
        suite.load_test(testable_kernel = test_kernel, test_parameters = build_parameters_from_args(args))
        suite.run()

    elif(args.HistoryRunTargets != None):
        run_testsuite(args.HistoryRunTargets,TestSuite.HISTORY)
    elif(args.CorrectnessRunTargets != None):
        run_testsuite(args.CorrectnessRunTargets,TestSuite.CORRECTNESS)
    elif(args.FullRunTargets != None):
        run_testsuite(args.FullRunTargets,TestSuite.FULL)
    elif(args.SearchSpaceTargets != None):
        run_testsuite(args.SearchSpaceTargets,TestSuite.PRINT_CONFIG)
    else:
        print("No mode selected. Exiting")
        sys.exit(0)