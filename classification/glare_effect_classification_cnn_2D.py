#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder
import keras
from keras.layers import Conv2D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling2D, Flatten
from keras.models import Model, Sequential
#from keras import backend as K
import os 
import json
from datetime import datetime
from multiprocessing import Pool
import pandas as pd


n_participants_per_split = 19 # 20 but one is removed in each split for testing
simulations_per_participant = 1000
numbers_of_simulation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
#n_added_simulations_per_participant = 20
n_runs = 1#20
n_epochs = 10#30
steps = 40


def create_image(game, components=[True, True, True, True, True]):
    card_codes = np.zeros((7, steps))
    cards_left = np.zeros((8, steps))
    never_revealed_cards = np.zeros((14, steps))
    max_same_card_reveals = np.zeros((21, steps))
    rounds_since_done = np.zeros((25, steps))
    
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
    #data_images = []
    #for split in range(len(data)):
    #    X = X_data[split][0]
    #    y = data[split][1]
    images = []
    for game in range(len(X_data)):
        image = create_image(X_data[game], components)
        images.append(image)
    #split_data = ((images, y))
    #data_images.append(split_data)
    return np.asarray(images)
    
def add_simulated_data(X_train, y_train, simulated_train_set, n_added_simulations_per_participant):
    for n in range(n_added_simulations_per_participant):
        for i in range(n_participants_per_split):

            X_train_simulated_1 = simulated_train_set[0][(i * simulations_per_participant) + n]
            y_train_simulated_1 = simulated_train_set[1][(i * simulations_per_participant) + n]
            X_train_simulated_2 = simulated_train_set[0][(simulations_per_participant * n_participants_per_split) \
                                                         + (i * simulations_per_participant) + n]
            y_train_simulated_2 = simulated_train_set[1][(simulations_per_participant * n_participants_per_split) \
                                                         + (i * simulations_per_participant) + n]
            
            X_train_simulated = np.concatenate((X_train_simulated_1[np.newaxis, :, :], \
                                               X_train_simulated_2[np.newaxis, :, :]), axis=0)
            y_train_simulated = np.concatenate((y_train_simulated_1[np.newaxis, :], \
                                               y_train_simulated_2[np.newaxis, :]), axis=0)

            X_train = np.concatenate((X_train, X_train_simulated), axis=0)
            y_train = np.concatenate((y_train, y_train_simulated), axis=0)
    return X_train, y_train


def mean_score_of_split(histories, epochs):
    mean_val_losses = []
    mean_val_accuracies = []
    mean_losses = []
    mean_accuracies = []
    for i in range(epochs):
        val_losses = []
        val_accuracies = []
        losses = []
        accuracies = []
        for l in range(len(histories)):
            history = histories[l]
            val_losses.append(history['val_loss'][i])
            val_accuracies.append(history['val_accuracy'][i])
            losses.append(history['loss'][i])
            accuracies.append(history['accuracy'][i])
        mean_val_losses.append(np.mean(val_losses))
        mean_val_accuracies.append(np.mean(val_accuracies))
        mean_losses.append(np.mean(losses))
        mean_accuracies.append(np.mean(accuracies))
    return mean_val_losses, mean_val_accuracies, mean_losses, mean_accuracies

def mean_score_over_all_splits(mean_split_scores, n_splits):
    val_losses = np.asarray(mean_split_scores[0][0])
    val_accuracies = np.asarray(mean_split_scores[0][1])
    losses = np.asarray(mean_split_scores[0][2])
    accuracies = np.asarray(mean_split_scores[0][3])
                            
    for i in range(1, n_splits):
        val_losses += np.asarray(mean_split_scores[i][0])
        val_accuracies += np.asarray(mean_split_scores[i][1])
        losses += np.asarray(mean_split_scores[i][2])
        accuracies += np.asarray(mean_split_scores[i][3])
                                 
    val_losses /= n_splits
    val_accuracies /= n_splits
    losses /= n_splits
    accuracies /= n_splits
    
    return val_losses, val_accuracies, losses, accuracies


def create_and_train(X_train, y_train, X_test, y_test, run ,split, file_dir, sd, n_epochs=n_epochs):
    opt = keras.optimizers.Adam(learning_rate=0.00001)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    cnn_input_shape = X_train[0].shape
    #epochs = n_epochs
    #cnn_batch_size = 32 #1000
    #verbose = 0
    
    #print(cnn_input_shape)
    verbose, epochs, batch_size = 0, n_epochs, 32 
    
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=cnn_input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    #model.add(Conv2D(10, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                        shuffle=True, validation_data=(X_test, y_test))
    
    #print(history.history['val_accuracy'])
    
    #histories.append(history)
    df = pd.DataFrame.from_dict(history.history, orient="index")
    df.to_csv(file_dir + '/split_histories_sd%sx/split%s/history_run%s' %(sd, split + 1, run))

def prepare_directories(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    for sd in numbers_of_simulation:
        if not os.path.exists(file_dir + '/split_histories_sd%sx' %sd):
            os.makedirs(file_dir + '/split_histories_sd%sx' %sd)
        for i in range(1, 21):
            if not os.path.exists(file_dir + '/split_histories_sd%sx/split%s' %(sd, i)):
                os.makedirs(file_dir + '/split_histories_sd%sx/split%s' %(sd, i))

def load_histories(file_dir, sd):
    splits = 20
    splits_histories = []
    for split in range(splits):
        histories = []
        for run in range(n_runs):
            df = pd.read_csv(file_dir + '/split_histories_sd%sx/split%s/history_run%s' %(sd, split + 1, run), index_col=0)
            history = df.to_dict('split')
            history = dict(zip(history['index'], history['data']))
            histories.append(history)
        splits_histories.append(histories)
    return splits_histories

def split_training(split):

    #base_path= '/home/anthony/Dokumente/Uni/Bachelor/6.Semester/ba_glare_effect/classification_data/features/'
    base_path= '/share/documents/students/amendil/classification_data/features/'

    # Real data for training
    X_train = np.load(base_path + 'real/Split%s/for_simulation/X_realData_train.npy' %str(split + 1))
    #X_train = X_train[:, :steps, :]
    
    y_train = np.load(base_path + 'real/Split%s/for_simulation/y_realData_train.npy' %str(split + 1))
    #real_data_splits_train.append((X_realData_train, y_realData_train))
    
    # Real data for testing
    X_test = np.load(base_path + 'real/Split%s/for_testing/X_realData_test.npy' %str(split + 1))
    X_test = create_images(X_test)
    #X_test = X_test[:, :steps, :]
    
    y_test = np.load(base_path + 'real/Split%s/for_testing/y_realData_test.npy' %str(split + 1))
    #real_data_splits_test.append((X_realData_test, y_realData_test))

    # Simulated data for training
    X_simulatedData_train = np.load(base_path + 'simulated/Split%s/X_simulatedData_train.npy' %str(split + 1))
    #X_simulatedData_train = X_simulatedData_train[:, :steps, :]
    
    y_simulatedData_train = np.load(base_path + 'simulated/Split%s/y_simulatedData_train.npy' %str(split + 1))
    #simulated_data_splits_train.append((X_simulatedData_train, y_simulatedData_train))
    
    # Adding simulated data. 
    X_train, y_train = add_simulated_data(X_train, y_train, (X_simulatedData_train, y_simulatedData_train), 0)#n_added_simulations_per_participant)
    X_train = create_images(X_train)

    # Shuffling training data
    temp_train = list(zip(X_train.tolist(), y_train.tolist()))
    random.shuffle(temp_train)
    X_train, y_train = zip(*temp_train)

    for i in range(n_runs): 
        create_and_train(np.asarray(X_train), np.asarray(y_train), X_test, y_test, i, split, file_dir, n_added_simulations_per_participant)


if __name__ == "__main__":

    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-2]
    global file_dir
    #file_dir = '/home/anthony/Dokumente/Uni/Bachelor/6.Semester/ba_glare_effect/classification_results' + '/2d_cnn_%s' %time
    file_dir = '/share/documents/students/amendil/classification_results' + '/2d_cnn_%s' %time

    prepare_directories(file_dir)


    for n_added_simulations_per_participant in numbers_of_simulation:
        
        pool = Pool(20)
        pool.map(split_training, list(range(20))) 

        splits_histories = load_histories(file_dir, n_added_simulations_per_participant)

        mean_split_scores = []
        for i in range(20):
            mean_split_score = mean_score_of_split(histories=splits_histories[i], epochs=n_epochs)
            mean_split_scores.append(mean_split_score)

        mean_val_losses, mean_val_accuracies, mean_losses, mean_accuracies = \
            mean_score_over_all_splits(mean_split_scores, 20)
        

        fig = plt.figure()
        plt.title('glareObs_noAdapt: Best validation accuracy %s%% (at epoch %s)' \
            %(round(mean_val_accuracies.max() * 100, 2), np.argmax(mean_val_accuracies) + 1))
        plt.plot(mean_accuracies)
        plt.plot(mean_val_accuracies)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train(RD_SD%sx)' %n_added_simulations_per_participant, 'validation(RD)'], loc='lower right')
        fig.savefig(file_dir + '/sd%sx_acc.png' %n_added_simulations_per_participant)
        
        fig = plt.figure()
        plt.title('glareObs_noAdapt: Lowest validation loss %s (at epoch %s)' \
            %(round(mean_val_losses.min(), 4), np.argmin(mean_val_losses) + 1))
        plt.plot(mean_losses)
        plt.plot(mean_val_losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train(RD_SD%sx)' %n_added_simulations_per_participant, 'validation(RD)'], loc='upper right')
        fig.savefig(file_dir + '/sd%sx_loss.png' %n_added_simulations_per_participant)