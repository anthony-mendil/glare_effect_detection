#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
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
'''

from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import keras

models_base_path = 'C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\split_histories_sd0x\\' # sdnx
test_data_base_path = 'C:\\Users\\dylin\Documents\\BA_Glare_Effect\\classification_data_initial\\features\\real\\'
steps = 20 # must correspond to the input dimensions of the trained model!


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

models_for_splits = []
for i in range(1, 21): 
    models_for_split = []
    for l in range(20):
        model = keras.models.load_model(models_base_path + 'split%s\\model_run%s' %(i, l))
        models_for_split.append(model)
    models_for_splits.append(models_for_split)

def vote(models_for_split, X_test_data):

    final_labels = []
    labels = []
    for model in models_for_split:
        pred = model.predict(X_test_data)
        run_labels = []
        for p in pred:
            if p[0] >= 0.5:
                run_labels.append([1.0, 0.0])
            elif p[1] >= 0.5:
                run_labels.append([0.0, 1.0])
            else: 
                print('Wrong!')
                exit()
        labels.append(run_labels)
    
    

    for i in range(2):
        no_obst_votes = 0
        glare_votes = 0
        for l in range(len(labels)):
            label = labels[l][i]
            if label[0] == 1:
                no_obst_votes += 1
            elif label[1] == 1:
                glare_votes += 1
        if no_obst_votes >= glare_votes:
            final_labels.append([1.0, 0.0])
        else:
            final_labels.append([0.0, 1.0])
        print(no_obst_votes, glare_votes)
    
    return final_labels
    

accuracies = []

for i in range(20):
    votes_for_split = vote(models_for_splits[i], X_splits_test_data[i])

    score = accuracy_score(y_splits_test_data[i] , votes_for_split)

    print(score)

    accuracies.append(score)

print(accuracies)

mean = np.mean(accuracies)

print(mean)


# TODO: 
'''
- models laden aus einen directory (sd-dir)
- load test data for each split (data-dir)
- for each test data for one split do the voting with the according 20 models (print decisions of each repition to check)
    --> result in final labels for the 20 glare effect and 20 no obstacle test data
- use the 40 labels for each split and determine accuracy of each split indipendently
- calculate average over splits
'''



