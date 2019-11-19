import numpy as np
import matplotlib.pyplot as plt

def plot_gflops(gflops):
	plt.plot(gflops)
	plt.ylabel('GFLOPS')
	plt.xlabel('Run')
	plt.show()

if(__name__ == "__main__"):
	plot_gflops([100,10,100,10,4])
