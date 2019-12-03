import argparse
from TestParameters import TestParameters

test_suite_argparser = argparse.ArgumentParser(description = 'Launch actions in the autoautoTVM testsuite.')
#These parameters uniquely define what type of program run is occuring and thus at most one should ever be set
program_run_group = test_suite_argparser.add_mutually_exclusive_group()
program_run_group.add_argument('--T',dest='IsTestRun', action='store_const',
                    const=True, default=False,
                    help='Run a test run of the suite by tuning VectorAdd.')
program_run_group.add_argument('--History',dest='HistoryRunTargets', metavar='XXXX.py',type=str,nargs='+', default=None,
                    help='Get the history of a given kernel without tuning. Usage is --B GEMM.py.')
program_run_group.add_argument('--Verification',dest='CorrectnessRunTargets', metavar='XXXX.py',nargs='+', default=None,
                    help='Get the correctness of the best tuned kernel in the history. Usage is --C GEMM.py.')
program_run_group.add_argument('--Full',dest='FullRunTargets', metavar='XXXX.py',nargs='+', default=None,
                    help='Perform a full run of the suite to create project results')
program_run_group.add_argument('--Delete',dest='DeleteLogsTargets', metavar='XXXX.py',nargs='+', default=None,
                    help='Delete the logs of previous runs.')
program_run_group.add_argument('--Space', dest='SearchSpaceTargets',metavar='XXXX.py',nargs='+',default=None,
					help='Print the search space for the kernels of a target.')
program_run_group.add_argument('--Tune', dest='TuningTargets',metavar='XXXX.py',nargs='+',default=None,
					help='Tune the kernels of a target.')

#These parameters are common to all types of program runs
test_suite_argparser.add_argument('--trials',dest='trial_runs',metavar='N',type=int,default=20,
					help='Number of trials to perform (points in search space to consider.')
test_suite_argparser.add_argument('--variance_runs',dest='variance_resistance_runs',metavar='N',type=int,default=5,
					help='Number of repetitions to perform at each point in search space.')
test_suite_argparser.add_argument('--dims',dest='dims',metavar='N',type=int,nargs='+',
					help='Array of input sizes. First value N is number of dimensions, followed by N values. E.g. GEMM --dims 3 512 512 512')
test_suite_argparser.add_argument('--dtype',dest='dtype',metavar='TYPE',type=str,default='float32',
					help='Type of element. e.g. --dtype float32.')


def build_parameters_from_args(args):
    if args.dims == None:
        return TestParameters(variance_resistance_runs = args.variance_resistance_runs, trial_runs = args.trial_runs,dims = None,dtype = args.dtype)
    dims = args.dims

    parameters = []
    kernel_dims = []
    todo = dims[0]
    i = 1
    while(i < len(dims)):
    	if(todo == 0):
    		parameters.append(TestParameters(variance_resistance_runs = args.variance_resistance_runs, trial_runs = args.trial_runs,dims = kernel_dims,dtype = args.dtype))
    		todo = dims[i]
    		kernel_dims = []
    	else:
    		kernel_dims.append(dims[i])
    		todo -= 1
    	i += 1

    parameters.append(TestParameters(variance_resistance_runs = args.variance_resistance_runs, trial_runs = args.trial_runs,dims = kernel_dims,dtype = args.dtype))

    return parameters


if(__name__ == "__main__"):
	print("Testing argparser")
	args = test_suite_argparser.parse_args(['--D','GEMM.py'])
	assert(args.DeleteLogsTargets == ['GEMM.py'])

	args = test_suite_argparser.parse_args(['--T'])
	assert(args.IsTestRun)
	assert(args.trial_runs == 20)
	assert(args.variance_resistance_runs == 5)
	assert(args.dims == None)
	assert(args.dtype == 'float32')
	args = test_suite_argparser.parse_args(['--F','GEMM.py','VectorAdd.py','--dims','3','512','512','512','1','4096'])
	assert(args.dims == [3,512,512,512,1,4096])
	params = build_parameters_from_args(args)
	assert(params[0].dims == [512,512,512])
	assert(params[1].dims == [4096])
	args = test_suite_argparser.parse_args(['--B','GEMM.py', '--dims','1','768'])
	assert(args.HistoryRunTargets == ['GEMM.py'])
	assert(build_parameters_from_args(args)[0].dims == [768])
	args = test_suite_argparser.parse_args(['--C','VectorAdd.py'])
	assert(args.CorrectnessRunTargets == ['VectorAdd.py'])
	args = test_suite_argparser.parse_args(['--F','VectorAdd.py','GEMM.py'])
	assert(args.FullRunTargets == ['VectorAdd.py','GEMM.py'])

