#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Used to train and test the 1D convnets 
# for various different configurations. 

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

# The numbers of participants used for training in each split.
n_participants_per_split = 19
# The number of available simulations per real game.
simulations_per_participant = 1000
# The different ratios of simulated to real games that should be trained on.
numbers_of_simulation = [0] #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
# The number of runs. 
n_runs = 20
# The number of epochs.
n_epochs = 2400 #2500 
# Number of game steps (one round consists of 2 steps).
steps = 20
# The learning rate.
lr = 0.0001

def add_simulated_data(X_train, y_train, simulated_train_set, n_added_simulations_per_participant):
    '''
    Add a specific number of simulated game features and labels to the real training data.  
    
    :param X_train: The real game features for training. 
    :param y_train: The real game labels for training. 
    :param simulated_train_set: The simulated features and the according labels. 
    :param n_added_simulations_per_participant: The number of simulations added to the traing data. 
    :return: The training data containing real and simulated data.
    '''

    for i in range(n_participants_per_split):

        X_train_simulated_noObst = simulated_train_set[0][(i * simulations_per_participant) : (i * simulations_per_participant) + n_added_simulations_per_participant]
        y_train_simulated_noObst = simulated_train_set[1][(i * simulations_per_participant) : (i * simulations_per_participant) + n_added_simulations_per_participant]

        X_train_simulated_glare = simulated_train_set[0][(simulations_per_participant * n_participants_per_split) \
                                                     + (i * simulations_per_participant) : (simulations_per_participant * n_participants_per_split) \
                                                     + (i * simulations_per_participant) + n_added_simulations_per_participant]
        y_train_simulated_glare = simulated_train_set[1][(simulations_per_participant * n_participants_per_split) \
                                                     + (i * simulations_per_participant) : (simulations_per_participant * n_participants_per_split) \
                                                     + (i * simulations_per_participant) + n_added_simulations_per_participant]
        
        X_train_simulated = np.concatenate((X_train_simulated_noObst, \
                                           X_train_simulated_glare), axis=0)
        y_train_simulated = np.concatenate((y_train_simulated_noObst, \
                                           y_train_simulated_glare), axis=0)

        X_train = np.concatenate((X_train, X_train_simulated), axis=0)
        y_train = np.concatenate((y_train, y_train_simulated), axis=0)
    return X_train, y_train


def mean_score_of_split(histories, epochs):
    '''
    Calculates the mean scores of the repitions for a split.
    
    :param histories: The histories. 
    :param epochs: The number of training epochs. 
    :return: The mean scores of that split.
    '''

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
    '''
    Calculats the mean scores over all splits.
    
    :param mean_split_scores: The scores of the individual splits. 
    :param n_splits: The number of splits. 
    :return: The mean scores over all splits.
    '''

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
    '''
    Creates and trains the model. 
    
    :param X_train: Train X data.
    :param y_train: Train y data.
    :param X_test: Test X data.
    :param y_test: Test y data.
    :param run: The number of the run.
    :param split: The number of the split.
    :param file_dir: The base path for saving the history and model.
    :param sd: The ratio of simulated to real data.
    :param n_epochs: The number of training epochs.
    '''

    adam = keras.optimizers.Adam(learning_rate=lr)
    
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
    
    model.save(file_dir + '/split_histories_sd%sx/split%s/model_run%s' %(sd, split + 1, run))

def prepare_directories(file_dir):
    '''
    Preparing the necessary directories.
    
    :param file_dir: The base directory to create. 
    '''

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    for sd in numbers_of_simulation:
        if not os.path.exists(file_dir + '/split_histories_sd%sx' %sd):
            os.makedirs(file_dir + '/split_histories_sd%sx' %sd)
        for i in range(1, 21):
            if not os.path.exists(file_dir + '/split_histories_sd%sx/split%s' %(sd, i)):
                os.makedirs(file_dir + '/split_histories_sd%sx/split%s' %(sd, i))

def load_histories(file_dir, sd):
    '''
    Loading the training histories.
    
    :param file_dir: The base directory where the files were saved. 
    They are stored in the structure created by prepare_directories. 
    :param sd: The ratio of simulated to real data. 
    :return: The training histories.
    '''

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
    '''
    Perfroming the training for each split with multiple repititions.
    
    :param split: The number of the split.
    '''

    base_path= '/share/documents/students/amendil/classification_data_initial/features/'

    # Real data for training
    X_train = np.load(base_path + 'real/Split%s/for_simulation/X_realData_train.npy' %str(split + 1))
    X_train = X_train[:, :steps, :]
    
    y_train = np.load(base_path + 'real/Split%s/for_simulation/y_realData_train.npy' %str(split + 1))
    
    # Real data for testing
    X_test = np.load(base_path + 'real/Split%s/for_testing/X_realData_test.npy' %str(split + 1))
    X_test = X_test[:, :steps, :]
    
    y_test = np.load(base_path + 'real/Split%s/for_testing/y_realData_test.npy' %str(split + 1))

    # Simulated data for training
    X_simulatedData_train = np.load(base_path + 'simulated/Split%s/X_simulatedData_train.npy' %str(split + 1))
    X_simulatedData_train = X_simulatedData_train[:, :steps, :]
    
    y_simulatedData_train = np.load(base_path + 'simulated/Split%s/y_simulatedData_train.npy' %str(split + 1))
    
    # Adding simulated data. 
    X_train, y_train = add_simulated_data(X_train, y_train, (X_simulatedData_train, y_simulatedData_train), n_added_simulations_per_participant)

    #print(X_train.shape)

    # Shuffling training data.
    temp_train = list(zip(X_train.tolist(), y_train.tolist()))
    random.shuffle(temp_train)
    X_train, y_train = zip(*temp_train)

    for i in range(n_runs): 
        create_and_train(np.asarray(X_train), np.asarray(y_train), X_test, y_test, i, split, file_dir, n_added_simulations_per_participant)


if __name__ == "__main__":

    # Determining unique name for the base directory that includes all important
    # informations. All files and models are saved in that directory (but with more subfolders). 
    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-2]
    global file_dir
    file_dir = '/share/documents/students/amendil/classification_models_initial_optimal_epoch' + '/1d_cnn_%s_%s_steps_%s_lr' %(time, steps, str(lr))

    # Preparing the necessary directories. 
    prepare_directories(file_dir)

    # Training and testing for each of the specified ratios of simulated to real games. 
    for n_added_simulations_per_participant in numbers_of_simulation:
        
        # Multiprocessed training of all splits (each split is additionally trained 20 times). 
        with Pool(20) as pool:
            pool.map(split_training, list(range(20))) 

        # Loading the training histories. 
        splits_histories = load_histories(file_dir, n_added_simulations_per_participant)

        # Calculating the mean scores of each split.
        mean_split_scores = []
        for i in range(20):
            mean_split_score = mean_score_of_split(histories=splits_histories[i], epochs=n_epochs)
            mean_split_scores.append(mean_split_score)

        # Calculating the mean scores over all splits. 
        mean_val_losses, mean_val_accuracies, mean_losses, mean_accuracies = \
            mean_score_over_all_splits(mean_split_scores, 20)
        
        # Plotting the average train and test accuracy in each epoch.
        fig = plt.figure()
        plt.title('glareObs_noAdapt: Best validation accuracy %s%% (at epoch %s)' \
            %(round(mean_val_accuracies.max() * 100, 2), np.argmax(mean_val_accuracies) + 1))
        plt.plot(mean_accuracies)
        plt.plot(mean_val_accuracies)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train(RD_SD%sx)' %n_added_simulations_per_participant, 'validation(RD)'], loc='lower right')
        fig.savefig(file_dir + '/sd%sx_acc.png' %n_added_simulations_per_participant)
        
        # Plotting the average train and test loss in each epoch.
        fig = plt.figure()
        plt.title('glareObs_noAdapt: Lowest validation loss %s (at epoch %s)' \
            %(round(mean_val_losses.min(), 4), np.argmin(mean_val_losses) + 1))
        plt.plot(mean_losses)
        plt.plot(mean_val_losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train(RD_SD%sx)' %n_added_simulations_per_participant, 'validation(RD)'], loc='upper right')
        fig.savefig(file_dir + '/sd%sx_loss.png' %n_added_simulations_per_participant)