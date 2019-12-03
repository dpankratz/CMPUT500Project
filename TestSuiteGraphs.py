import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import TestSuite
import os
import datetime
from TestableKernel import testing_kernels as targets_dict
from TestableKernel import results_kernels as results

SHOW_PLOTS = 0

def plot_gflops(data,annotated_points,labels,save_name):
	
	plt.clf()
	print(annotated_points)
	for array in data:
		plt.plot(array)



	visible_annotation = None
	annotations = []
	for points_array in annotated_points:
		for x,y,txt in points_array:
			annotations.append(plt.annotate(txt,(x,y),xytext = (0,0),bbox=dict(boxstyle="round", fc="w"),arrowprops=dict(arrowstyle="->")))
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
	
	if(SHOW_PLOTS):
		plt.show()
	else:
		plt.savefig(save_name)

colors = ['brown','r','orange','y','pink']

def plot_space(labels,space_sizes):
	fig,ax = plt.subplots()
	bars = ax.bar(labels,space_sizes)
	for i in range(len(bars)):
		bars[i].set_color(colors[i//3])
	ax.set_ylabel("Points")
	ax.set_yscale('log')
	ax.set_title("Search Space Sizes")
	plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor") 
	fig.set_size_inches(15,6)
	plt.tight_layout()
	if(SHOW_PLOTS):
		plt.savefig('SearchSpace.png')
	else:
		plt.show()
	

def plot_time(labels,times):
	fig,ax = plt.subplots()
	bars = ax.bar(labels,times)
	for i in range(len(bars)):
		bars[i].set_color(colors[i//3])
	ax.set_ylabel("Minutes")
	plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor") 
	fig.set_size_inches(15,6)
	plt.tight_layout()
	if(SHOW_PLOTS):
		plt.show()
	else:
		plt.savefig('SearchTime.png')
	

if(__name__ == "__main__"):
	labels = []
	spaces = []
	times  = []
	for kernels in results.values():
		for kernel in kernels.get_tunable_kernels():
			if(os.path.exists(TestSuite.TestSuite.infofile_path(kernel))):
				labels.append(TestSuite.TestSuite.kernel_name(kernel))
				info = open(TestSuite.TestSuite.infofile_path(kernel))
				spaces.append(int(info.readline().split('=')[1]))
				line = info.readline()
				while(len(line.split('=')) == 1):
					line = info.readline()
				time = datetime.datetime.strptime(line.split('=')[1].split('.')[0], "%H:%M:%S")
				times.append(time.hour * 60 + time.minute + time.second/60)



	plot_space(labels,spaces)
	plot_time(labels,times)
