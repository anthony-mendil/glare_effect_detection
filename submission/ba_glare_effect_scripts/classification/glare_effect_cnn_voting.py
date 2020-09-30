#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Implements the ensemble based systems either using 
# 1D convnets or 2D convnets. 

from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import keras
import math

# The base path of the models for the system.
models_base_path = base_path= '/share/documents/students/amendil/classification_models_initial_optimal_epoch/2d_cnn_2020-09-15 23:58:31.9583_40_steps_1e-05_lr/split_histories_sd4x/'
# The path to the test data.
test_data_base_path = '/share/documents/students/amendil/classification_data_initial/features/real/'
# The number of steps the models were trained on. 
steps = 40 
# Whether 2D convnets are used. 
is_2d = True


def create_image(game, components=[True, True, True, True, True]):
    '''
    Creates an synthetic image out of the statistical features.  
    
    :param game: The statistical features of a game. 
    :param components: Defines which statistical features are included in the synthetic image. 
    :return: The synthetic image.
    '''

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
        
    image = np.zeros((0, steps))
    if components[0]:   
        image = np.vstack((image, card_codes))
    if components[1]:
        image = np.vstack((image, max_same_card_reveals))
    if components[2]:   
        image = np.vstack((image, rounds_since_done))
    if components[3]:
        image = np.vstack((image, cards_left))
    if components[4]:   
        image = np.vstack((image, never_revealed_cards))
    # Note: Switched order of statistival features so that they have some space between them.
        
    return image

def create_images(X_data, components=[True, True, True, True, True]):
    '''
    Creates synthetic images out of the statistical features.  
    
    :param X_data: The statistical features of all games. 
    :param components: Defines which statistical features are included in the synthetic images. 
    :return: The synthetic images.
    '''

    images = []
    for game in range(len(X_data)):
        image = create_image(X_data[game], components)
        images.append(image)
    return np.asarray(images)

#######################################

# Loading the test data. 
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

# Loading the models for the ensemble based system. 
models_for_splits = []
for i in range(1, 21): 
    models_for_split = []
    for l in range(20):
        model = keras.models.load_model(models_base_path + 'split%s/model_run%s' %(i, l))
        models_for_split.append(model)
    models_for_splits.append(models_for_split)

#######################################

def vote(models_for_split, X_test_data):
    '''
    Creates synthetic images out of the statistical features.  
    
    :param models_for_split: The models trained on a split. 
    :param X_test_data: The test data. 
    :return: The voted prediciton.
    '''

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
    '''
    Determines the average accuracy of the models without voting.  
    
    :param models_for_splits: All models for the different splits. 
    :param X_splits_test_data: The test data. 
    :param y_splits_test_data: The target labels.
    :return: The average accuracy of all models without voting. 
    '''

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

#######################################

# Determining the overall accuracies of individual splits. 
accuracies = []
for i in range(20):
    test_data = X_splits_test_data[i]

    # If 2D convnets are used it is necessary to create the synthetic images. 
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

# The average accuracy over all splits. 
mean = np.mean(accuracies)

print(mean)

print('Average accuracy without voting:')
# The average accuracy of all models without voting. 
avg_acc = determine_avg_acc(models_for_splits, X_splits_test_data, y_splits_test_data)
print(avg_acc)