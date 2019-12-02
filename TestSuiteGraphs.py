import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import TestSuite
import os
import datetime
from TestableKernel import testing_kernels as targets_dict

def plot_gflops(data,annotated_points,labels):
	
	print(annotated_points)
	for array in data:
		plt.plot(array)



	visible_annotation = None
	annotations = []
	for points_array in annotated_points:
		for x,y,txt in points_array:
			annotations.append(plt.annotate(txt,(x,y),xytext = (-1,-0.5),bbox=dict(boxstyle="round", fc="w"),arrowprops=dict(arrowstyle="->")))
			annotations[-1].set_visible(False)

	plt.legend(labels,loc=4)
	plt.ylabel('GFLOPS')
	plt.xlabel('Run')

	fig = plt.gcf()
	fig.set_size_inches(18,8)

	def onclick(event):
		nonlocal visible_annotation,fig
		for annotation in annotations:
			x,y = annotation.xy
			if(abs(x - event.xdata) < 1 and abs(y - event.ydata) < 1):
				annotation.set_visible(True)
				if visible_annotation != None:
					if visible_annotation == annotation:
						break
					visible_annotation.set_visible(False)
				visible_annotation = annotation
				fig.canvas.draw()
				fig.canvas.flush_events()
				break
		

	fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()

def plot_space(labels,space_sizes):

	fig,ax = plt.subplots()
	ax.bar(labels,space_sizes)
	ax.set_ylabel("Points")
	ax.set_yscale('log')
	plt.show()

def plot_time(labels,times):
	plt.plot_date(labels,dates)
	plt.show()

if(__name__ == "__main__"):
	labels = []
	spaces = []
	times  = []
	for kernels in targets_dict.values():
		for kernel in kernels.get_tunable_kernels():
			if(os.path.exists(TestSuite.TestSuite.infofile_path(kernel))):
				labels.append(TestSuite.TestSuite.kernel_name(kernel))
				info = open(TestSuite.TestSuite.infofile_path(kernel))
				spaces.append(int(info.readline().split('=')[1]))
				line = info.readline()
				while(len(line.split('=')) == 1):
					line = info.readline()
				times.append(datetime.datetime.strptime(line.split('=')[1].split('.')[0], "%H:%M:%S"))


	plot_space(labels,spaces)
	plot_time(labels,times)
