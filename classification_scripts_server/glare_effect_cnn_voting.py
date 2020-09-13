#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
import os 
import json
from datetime import datetime
import pandas as pd

models_base_path = ''
test_data_base_path = 'C:\\Users\\dylin\Documents\\BA_Glare_Effect\\classification_data_initial\\features\\real\\'
steps = 20

X_splits_test_data = []
y_splits_test_data = []
for i in range(1, 21):
    X_split_test_data = np.load(test_data_base_path + 'Split%s\\for_testing\\X_realData_test.npy' %i)
    y_split_test_data = np.load(test_data_base_path + 'Split%s\\for_testing\\y_realData_test.npy' %i)
    X_split_test_data = X_split_test_data[:, :steps, :]
    X_splits_test_data.append(X_split_test_data)
    y_splits_test_data.append(y_split_test_data)

print(X_splits_test_data[0]) 
print(y_splits_test_data[0])

print(X_splits_test_data[0].shape) 
print(y_splits_test_data[0].shape)

#models_for_splits = []
#for i in range(1, 21): 
#    for l in range(20):


# TODO: 
'''
- models laden aus einen directory (sd-dir)
- load test data for each split (data-dir)
- for each test data for one split do the voting with the according 20 models (print decisions of each repition to check)
    --> result in final labels for the 20 glare effect and 20 no obstacle test data
- use the 40 labels for each split and determine accuracy of each split indipendently
- calculate average over splits
'''



