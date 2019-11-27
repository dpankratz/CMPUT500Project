import numpy as np
import matplotlib.pyplot as plt
from TestableKernel import testing_kernels as targets_dict

def plot(data,annotated_points,labels):
	
	print(annotated_points)
	for array in data:
		plt.plot(array)



	visible_annotation = None
	annotations = []
	for points_array in annotated_points:
		for x,y,txt in points_array:
			annotations.append(plt.annotate(txt,(x,y),xytext = (2,2),bbox=dict(boxstyle="round", fc="w"),arrowprops=dict(arrowstyle="->")))
			annotations[-1].set_visible(False)

	plt.legend(labels,loc=4)
	plt.ylabel('GFLOPS')
	plt.xlabel('Run')

	fig = plt.gcf()
	fig.set_size_inches(15,8)

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

if(__name__ == "__main__"):
	plot([([1,2,3,4],"GEMM"),([4,3,2,1],"VectorAdd")])
