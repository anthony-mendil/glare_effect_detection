#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# line 1 points
x1 = [5,10,15,20]
y1 = [78.88,81.12,78.37,81.25]
s1 = ['sd9x', 'sd0x', 'sd0x', 'sd2x']
# plotting the line 1 points 
plt.plot(x1, y1, label = "1D CNN",marker='o', markerfacecolor='blue', markersize=10)
# line 2 points
x2 = [5,10,15,20]
y2 = [73,80,81.5,85]
s2 = ['sd20x', 'sd2x', 'sd9x', 'sd4x']
# plotting the line 2 points 
plt.plot(x2, y2, label = "2D CNN",marker='o', markerfacecolor='orange', markersize=10)
plt.xlabel('Number Of Rounds')
# Set the y axis label of the current axis.
plt.ylabel('Accuracy')
# Set a title of the current axes.
#plt.title('Two or more lines on same plot with suitable legends ')
# show a legend on the plot
plt.legend()
# Display a figure.

for i,j,s in zip(x1,y1,s1):
    corr = -0.5# -0.05 # adds a little correction to put annotation in marker's centrum
    plt.annotate(s,  xy=(i + corr, j + 0.5))

for i,j,s in zip(x2,y2,s2):
    corr = -0.5# -0.05 # adds a little correction to put annotation in marker's centrum
    plt.annotate(s,  xy=(i + corr, j - 1.2))

plt.xticks(np.arange(5, 25, step=5))
plt.xlim(4, 21)
plt.ylim(70, 90)

plt.show()