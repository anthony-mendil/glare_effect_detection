#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
import os 
import json
from datetime import datetime
from multiprocessing import Pool
import pandas as pd

n_participants_per_split = 19 # 20 but one is removed in each split for testing
simulations_per_participant = 1000
numbers_of_simulation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
n_runs = 20
n_epochs = 1500
steps = 20

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
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    
    verbose, epochs, batch_size = 0, n_epochs, 32 
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                        shuffle=True, validation_data=(X_test, y_test))
    
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

    base_path= '/share/documents/students/amendil/classification_data/features/'

    # Real data for training
    X_train = np.load(base_path + 'real/Split%s/for_simulation/X_realData_train.npy' %str(split + 1))
    X_train = X_train[:, :steps, :]
    
    y_train = np.load(base_path + 'real/Split%s/for_simulation/y_realData_train.npy' %str(split + 1))
    #real_data_splits_train.append((X_realData_train, y_realData_train))
    
    # Real data for testing
    X_test = np.load(base_path + 'real/Split%s/for_testing/X_realData_test.npy' %str(split + 1))
    X_test = X_test[:, :steps, :]
    
    y_test = np.load(base_path + 'real/Split%s/for_testing/y_realData_test.npy' %str(split + 1))
    #real_data_splits_test.append((X_realData_test, y_realData_test))

    # Simulated data for training
    X_simulatedData_train = np.load(base_path + 'simulated/Split%s/X_simulatedData_train.npy' %str(split + 1))
    X_simulatedData_train = X_simulatedData_train[:, :steps, :]
    
    y_simulatedData_train = np.load(base_path + 'simulated/Split%s/y_simulatedData_train.npy' %str(split + 1))
    #simulated_data_splits_train.append((X_simulatedData_train, y_simulatedData_train))
    
    # Adding simulated data. 
    X_train, y_train = add_simulated_data(X_train, y_train, (X_simulatedData_train, y_simulatedData_train), 0)#n_added_simulations_per_participant)

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
    file_dir = '/share/documents/students/amendil/classification_results' + '/1d_cnn_%s' %time

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