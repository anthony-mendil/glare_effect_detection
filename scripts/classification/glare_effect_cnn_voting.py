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
import math


models_base_path = base_path= '/share/documents/students/amendil/classification_models_initial_optimal_epoch/2d_cnn_2020-09-15 23:58:31.9583_40_steps_1e-05_lr/split_histories_sd4x/'


# '/share/documents/students/amendil/classification_models_initial_optimal_epoch/1d_cnn_2020-09-14 08:59:43.5671_20_steps_0.0001_lr/split_histories_sd0x/'

test_data_base_path = '/share/documents/students/amendil/classification_data_initial/features/real/'
steps = 40 # must correspond to the input dimensions of the trained model!
is_2d = True


def create_image(game, components=[True, True, True, True, True]):
    card_codes = np.zeros((7, steps))
    cards_left = np.zeros((8, steps))
    never_revealed_cards = np.zeros((14, steps))
    max_same_card_reveals = np.zeros((20, steps))
    rounds_since_done = np.zeros((27, steps))
    
    x_position = 0
    
    for step in game:
        card_code = math.floor(step[0])
        first_or_second = int(round((step[0] % 1) * 10))
        
        if card_code != 0:
            card_codes[card_code - 1][x_position] = first_or_second
            
        cards_left[int(step[1] / 2)][x_position] = 1
        never_revealed_cards[int(step[2])][x_position] = 1
        max_same_card_reveals[int(step[3])][x_position] = 1
        rounds_since_done[int(step[4])][x_position] = 1
        
        x_position += 1
        
    # Try leaving out some features and compare results!
    image = np.zeros((0, steps))
    if components[0]:   # Good visual feature for cnn.
        image = np.vstack((image, card_codes))
    if components[1]:
        image = np.vstack((image, max_same_card_reveals))
    if components[2]:   # I think good visual feature for cnn. 
        image = np.vstack((image, rounds_since_done))
    if components[3]:
        image = np.vstack((image, cards_left))
    if components[4]:   # I think this feature is not very usefull for the cnn. 
                        # No big visual difference between being blinded an not. 
        image = np.vstack((image, never_revealed_cards))
    #switched order of statistival features so that they have some space between them.
        
    return image

def create_images(X_data, components=[True, True, True, True, True]):
    images = []
    for game in range(len(X_data)):
        image = create_image(X_data[game], components)
        images.append(image)
    return np.asarray(images)


X_splits_test_data = []
y_splits_test_data = []
for i in range(1, 21):
    X_split_test_data = np.load(test_data_base_path + 'Split%s/for_testing/X_realData_test.npy' %i)
    y_split_test_data = np.load(test_data_base_path + 'Split%s/for_testing/y_realData_test.npy' %i)
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
        model = keras.models.load_model(models_base_path + 'split%s/model_run%s' %(i, l))
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

def determine_avg_acc(models_for_splits, X_splits_test_data, y_splits_test_data):
    split_averages = []

    for i in range(20):
        split_accuracies = []

        split_models = models_for_splits[i]

        X_test_data = X_splits_test_data[i]
        if is_2d:
            X_test_data = create_images(X_test_data)
            X_test_data = X_test_data.reshape(X_test_data.shape[0], X_test_data.shape[1], X_test_data.shape[2], 1)

        y_test_data = y_splits_test_data[i]

        for model in split_models:
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

            acc = accuracy_score(y_test_data, np.asarray(run_labels))
            split_accuracies.append(acc)

        split_averages.append(np.mean(split_accuracies))

    return np.mean(split_averages)
    

accuracies = []

for i in range(20):
    test_data = X_splits_test_data[i]

    if is_2d:
        test_data = create_images(test_data)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)

    votes_for_split = vote(models_for_splits[i], test_data)

    print(y_splits_test_data[i])
    print(np.asarray(votes_for_split))

    score = accuracy_score(y_splits_test_data[i] , np.asarray(votes_for_split))

    print(score)

    accuracies.append(score)

print(accuracies)

mean = np.mean(accuracies)

print(mean)

print('Average accuracy without voting:')
avg_acc = determine_avg_acc(models_for_splits, X_splits_test_data, y_splits_test_data)
print(avg_acc)

# TODO: 
'''
- models laden aus einen directory (sd-dir)
- load test data for each split (data-dir)
- for each test data for one split do the voting with the according 20 models (print decisions of each repition to check)
    --> result in final labels for the 20 glare effect and 20 no obstacle test data
- use the 40 labels for each split and determine accuracy of each split indipendently
- calculate average over splits
'''



