# CMPUT500Project

This project includes a tool that assists in conducting experiments of different analyses targeting search space generated. It further includes a set of analyses for this purpose.

# Files:
- **Kernels:** GEMM.py, VectorAdd.py,ConvNCHW.py,relu.py,reorg.py,resize.py,softmax.py,upsample.py

- **Passes.py** Contains the 3 analyses, naive, moderate and conservative

- **TestSuite.py** Contains the main tool, running modes are `Full`,`Space`,`Tune`,`Verification`,`History`,`Testing`,`Delete`. Options args are `trials`,`variance-runs`, and `dims`

- **TestSuiteGraphs.py** Contains a utility for generating graphs based on the logs generated by TestSuite

- **TestSuiteArgParser.py** Parses the command line and generates variables to be consumed by TestSuite

- **TestableKernel.py** Defines what a 'kernel' is to be passed to the TestSuite. At a high level it contains a reference to the tunable kernel, a reference to an input generator, parameter generator and numpy verification function.

- **TestParameters.py** Defines the parameters for a test run of the TestSuite. Includes number of trial runs, variance runs, and dims.
