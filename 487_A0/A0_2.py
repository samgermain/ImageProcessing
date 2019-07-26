import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Make points for a sine wave.
x = np.arange(0, 2*pi, step=0.01 )
y = np.sin(4*x)
plt.plot(x,y)
plt.show()
#Better sine wave graph
plt.title('A 4Hz Sine Wave')
plt.ylim(-1.1, 1.1)
plt.xlim(-pi/4, (9*pi)/4)
blue_line = plt.plot(x, y, label='4hz')
y=np.sin(7*x)
red_line = plt.plot(x,y, linestyle='dashed', color='red', label='7hz')
plt.legend()
plt.show()

#Scatter plot
from sklearn import datasets

iris = datasets.load_iris()
species1samples = iris.data[iris.target==0, :]
species2samples = iris.data[iris.target==1, :]
species3samples = iris.data[iris.target==2, :]
s1_length = [i[0] for i in species1samples]
s1_width = [j[1] for j in species1samples]
s2_length = [i[0] for i in species2samples]
s2_width = [j[1] for j in species2samples]
s3_length = [i[0] for i in species3samples]
s3_width = [j[1] for j in species3samples]

plt.figure(figsize=(8,6))
plt.scatter(s1_length, s1_width, color='red')
plt.scatter(s2_length, s2_width, color='green', marker='v')
plt.scatter(s3_length, s3_width, color='blue', marker='s')
plt.xlabel('Sepial Length')
plt.ylabel('Sepial Width')
plt.title('Iris Data')
plt.show()

#Bar plot
mean_s1Length = np.mean(s1_length)
mean_s2Length = np.mean(s2_length)
mean_s3Length = np.mean(s3_length)
plt.bar(0.75, mean_s1Length, width=0.5)
plt.bar(1.75, mean_s2Length, width=0.5)
plt.bar(2.75, mean_s2Length, width=0.5)
plt.ylabel('Average Sepial Length')
plt.title('Average Sepial Length of Three Iris Species')
plt.xticks([0.75, 1.75, 2.75],('Species1', 'Species2', 'Species3'))
plt.show()