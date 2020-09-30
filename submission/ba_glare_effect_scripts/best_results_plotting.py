#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Used to plot the best classification results of the 1D and
# 2D convnets for differnet numbers of rounds. 

import numpy as np
import matplotlib.pyplot as plt

# Accuracies of 1D convnets for the different numbers of rounds. 
x1 = [5,10,15,20]
y1 = [74.88,81.88,78.63,81.12]
# Ratios of simualated to real games. 
s1 = ['sd9x', 'sd0x', 'sd0x', 'sd4x']
# Plotting them.  
plt.plot(x1, y1, label = "1D convnet",marker='o', markerfacecolor='blue', markersize=10)

# Accuracies of 2D convnets for the different numbers of rounds. 
x2 = [5,10,15,20]
y2 = [73,80,81.5,85]
# Ratios of simualated to real games. 
s2 = ['sd20x', 'sd2x', 'sd9x', 'sd4x']
# Plotting them. 
plt.plot(x2, y2, label = "2D convnet",marker='o', markerfacecolor='orange', markersize=10)

# Setting the labels of the axes. 
plt.xlabel('Number Of Rounds')
plt.ylabel('Accuracy')

# Show a legend on the plot.
plt.legend()

# Adding annotations for the ratios of simulated to real games. 
for i,j,s in zip(x1,y1,s1):
    corr = -0.5 # Adds a little correction to put annotation in marker's centrum.
    plt.annotate(s,  xy = (i + corr, j + 0.5))

for i,j,s in zip(x2,y2,s2):
    corr = -0.5 # Adds a little correction to put annotation in marker's centrum.
    plt.annotate(s,  xy = (i + corr, j - 1.2))

# Setting the ranges and ticks. 
plt.xticks(np.arange(5, 25, step=5))
plt.xlim(4, 21)
plt.ylim(70, 90)

# Plot the figure. 
plt.show()