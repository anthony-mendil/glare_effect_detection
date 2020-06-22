#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import os

class FeaturesGenerator():
    """
    This class generates features from a giving memory game log
    Note:
    When testing the correctness of this class's functions, game_log was passed as pd-Series, while its usage in testing
    the real data will pass it as pd.DataFrame, therefore the game_log here is modified accordingly
    """

    def __init__(self, game_log, sequence_length=40, num_of_cards=14):
        self.game_log = game_log
        self.sequence_length = sequence_length
        self.num_of_cards = num_of_cards
        self.game_schema = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]
        self.all_cards = [i+0.1 for i in range(1,8)]
        self.all_cards.extend([i+0.2 for i in range(1,8)])

    # Lda features functions
    def get_undiscovered_cards(self):
        return self.num_of_cards - self.get_matching_cards()

    def get_unseen_cards(self):
        return len(list(set(self.all_cards) - set(self.game_log.values[0])))

    def get_max_seen_number(self):
        max_seen_number = 0
        for i in self.all_cards:
            i_count = self.game_log.values[0].tolist().count(i)
            if i_count > max_seen_number:
                max_seen_number = i_count

        return max_seen_number

    def get_zero_cards(self):
        return self.game_log.values[0].tolist().count(0.0)

    #  Auxiliary functions
    def get_matching_cards(self):
        matching_cards = 0
        num_of_rounds = [i for i in range(1, int(self.sequence_length/2)+1)]

        for r in num_of_rounds:
            card_1 = self.game_log.loc[:, 'card_' + str((r * 2) - 1)]
            card_2 = self.game_log.loc[:, 'card_' + str((r * 2))]

            if abs(card_1.values[0] - card_2.values[0]) < 0.12 and abs(card_1.values[0] - card_2.values[0]) != 0:
                matching_cards += 2

        return matching_cards

class Data_preparator():
    def __init__(self,
                sequence_length=40,
                num_of_subjects=20,
                num_subject_game_trials=1,
                cardsCodes_statistics_features=[True, True, True, True, True],
                ):

        self.sequence_length = sequence_length
        self.num_of_subjects = num_of_subjects
        self.num_subject_game_trials = num_subject_game_trials
        self.cardsCodes_statistics_features = cardsCodes_statistics_features 

    def load_and_prepare_real_data_all(self, 
                                    base_path, 
                                    base_path_features, 
                                    two_labels=True,
                                    num_of_rounds=20,
                                    pooled_levels=[True, True, False, False, False, False, False,
                                                    False, False]):

        """This function loads the real data, according to base_path,
        from for_simulation, for_validation or for testing"""

        # data path and schema
        path_prefix = base_path
        first_level = 'valid_noObst_logs.txt'
        second_level = 'valid_glare_effect_logs.txt'
        # columns
        similarity_cols_noObst = ['sim_' + str(i) for i in range(1, 22)]
        similarity_cols_glare = ['sim_' + str(i) for i in range(1, 29)]
        cards_cols = ['card_' + str(i) for i in range(1, 41)]
        timestamps_cols = ['timestamp_' + str(i) for i in range(1, 41)]
        metrics_cols = ['metric_' + str(i) for i in range(1, 5)]
        label_col = ['label']
        all_cols_noObst = similarity_cols_noObst + cards_cols + timestamps_cols + metrics_cols + label_col
        all_cols_glare = similarity_cols_glare + cards_cols + timestamps_cols + metrics_cols + label_col

        # I. load realData
        rounds = ['card_' + str(i) for i in range(1, (num_of_rounds * 2) + 1)]
        rounds_and_label = rounds + ['label']

        if pooled_levels[0]:
            firstLevelMemory_realData = pd.read_csv(
                path_prefix + first_level, names=all_cols_noObst, header=None)

            firstLevelMemory_realData.label = 0
            firstLevelMemory_realData = firstLevelMemory_realData.loc[:, rounds_and_label]

            print('firstLevelMemory_realData has been loaded!')

        # if (not two_labels) or memory_obstacle_newApp:
        if pooled_levels[1]:
            secondLevelMemory_realData = pd.read_csv(
                path_prefix + second_level, names=all_cols_glare, header=None)

            secondLevelMemory_realData.label = 1
            secondLevelMemory_realData = secondLevelMemory_realData.loc[:, rounds_and_label]

            print('secondLevelMemory_realData has been loaded!')

        # II. concatenate and shuffle validationData
        self.realData = []
        if pooled_levels[0]:
            self.realData.extend([firstLevelMemory_realData])
        if pooled_levels[1]:
            self.realData.extend([secondLevelMemory_realData])
        

        self.realData = pd.concat(self.realData, ignore_index=True)
        #self.realData = self.realData.sample(frac=1)  # Shuffle  # Dont shuffle!!
        print("realData: " + str(self.realData.shape))

        # III Preparation of X_validation, y_validation, lda_X_validation and lda_y_validation

        ############
        # 2. Baseline Model: prepare X_lda, y_lda & trainingData_with_features
        
        game_lda_features = []
        game_cards_with_lda_features = []
        all_games_lda_features = []
        all_games_cards_with_lda_features = []
        for game in range(0, len(self.realData)):
            # prepare the game lda features
            temp = self.realData.iloc[game:game + 1, :]
            cards_used = ['card_' + str(i) for i in range(1, (num_of_rounds * 2) + 1)]
            lda_features_generator = FeaturesGenerator(game_log=temp.loc[:, cards_used],
                                                            sequence_length=num_of_rounds * 2)
            game_lda_features.append(lda_features_generator.get_undiscovered_cards())
            game_lda_features.append(lda_features_generator.get_unseen_cards())
            game_lda_features.append(lda_features_generator.get_max_seen_number())
            game_lda_features.append(lda_features_generator.get_zero_cards())

            if True in self.cardsCodes_statistics_features[1:]:  # i.e. statistical features == True
            # prepare statistical features
                cards_no_features = temp.loc[:, cards_used]  # .values.tolist()
                incremental_df = pd.DataFrame()
                incremental_df['card_1'] = cards_no_features['card_1']
                for card, _ in zip(cards_no_features, range(0, len(cards_no_features.columns))):
                    incremental_df[card] = cards_no_features[card]

                    lda_features_generator = FeaturesGenerator(game_log=incremental_df,
                                                                    sequence_length=len(incremental_df.columns))
                    undiscovered_cards = lda_features_generator.get_undiscovered_cards()
                    unseen_cards = lda_features_generator.get_unseen_cards()
                    max_seen_number = lda_features_generator.get_max_seen_number()
                    zero_cards = lda_features_generator.get_zero_cards()
                    card_code = math.floor(incremental_df.iloc[0][card] * 10) / 10
                    # features_used
                    all_features = [card_code, undiscovered_cards, unseen_cards, max_seen_number, zero_cards]
                    features_used = [all_features[i] for i, flag in enumerate(self.cardsCodes_statistics_features)
                                        if flag]
                    game_cards_with_lda_features.append(features_used)

            # append to all_games lists
            all_games_lda_features.append(game_lda_features)
            all_games_cards_with_lda_features.append(game_cards_with_lda_features)
            game_lda_features = []
            game_cards_with_lda_features = []

        # lda_X_validation
        self.lda_X_realData = np.array(all_games_lda_features)
        # X_validation and y_validation
        if True in self.cardsCodes_statistics_features[1:]:
            self.X_realData = np.array(all_games_cards_with_lda_features)
        else:
            self.X_realData = np.array(self.realData.loc[:, rounds])
            # in case of one feature (card code), reshape to ensure the (#samples, #timesteps, #features) for LSTM
            self.X_realData = np.reshape(self.X_realData, (self.X_realData.shape[0], self.X_realData.shape[1], 1))

        self.y_realData = self.realData.label
        self.y_realData = np.array(self.y_realData)
        # reshaping y_validation
        self.y_realData = np.reshape(self.y_realData, (self.y_realData.shape[0], 1))
        # lda_y_validation
        self.lda_y_realData = self.y_realData
        # reshaping lda_y_validation
        self.lda_y_realData = np.reshape(self.lda_y_realData, (len(self.lda_y_realData)))
        # one hot encoding for y_validation
        num_classes = 2
        self.y_realData = to_categorical(self.y_realData, num_classes=num_classes)

        print("X_realData: " + str(np.shape(self.X_realData)))
        print("y_realData: " + str(np.shape(self.y_realData)))

        print("lda_X_realData: " + str(np.shape(self.lda_X_realData)))
        print("lda_y_realData: " + str(np.shape(self.lda_y_realData)))

        np.save(base_path_features + 'realData', self.realData)
        np.save(base_path_features + 'X_realData', self.X_realData)
        np.save(base_path_features + 'y_realData', self.y_realData)
        np.save(base_path_features + 'lda_X_realData', self.lda_X_realData)
        np.save(base_path_features + 'lda_y_realData', self.lda_y_realData)

    def load_and_prepare_simulated_data_all(self, 
                                    base_path, 
                                    base_path_features, 
                                    two_labels=True,
                                    num_of_rounds=20,
                                    pooled_levels=[True, True, False, False, False, False, False,
                                                    False, False]):

        """This function loads the real data, according to base_path,
        from for_simulation, for_validation or for testing"""

        # data path and schema
        path_prefix = base_path
        first_level = 'simulated_noObst_sorted.log'
        second_level = 'simulated_glare_effect_sorted.log'
        # columns
        #similarity_cols_noObst = ['sim_' + str(i) for i in range(1, 22)]
        #similarity_cols_glare = ['sim_' + str(i) for i in range(1, 29)]
        cards_cols = ['card_' + str(i) for i in range(1, 41)]
        #timestamps_cols = ['timestamp_' + str(i) for i in range(1, 41)]
        metrics_cols = ['metric_' + str(i) for i in range(1, 5)]
        label_col = ['label']
        rmse = ['rmse']
        #all_cols_noObst = similarity_cols_noObst + cards_cols + timestamps_cols + metrics_cols + label_col
        #all_cols_glare = similarity_cols_glare + cards_cols + timestamps_cols + metrics_cols + label_col

        all_cols_simulated = rmse + cards_cols + metrics_cols + label_col


        # I. load simulatedData
        rounds = ['card_' + str(i) for i in range(1, (num_of_rounds * 2) + 1)]
        rounds_and_label = rounds + ['label']

        if pooled_levels[0]:
            firstLevelMemory_simulatedData = pd.read_csv(
                path_prefix + first_level, names=all_cols_simulated, header=None)

            firstLevelMemory_simulatedData.label = 0
            firstLevelMemory_simulatedData = firstLevelMemory_simulatedData.loc[:, rounds_and_label]

            print('firstLevelMemory_simulatedData has been loaded!')

        # if (not two_labels) or memory_obstacle_newApp:
        if pooled_levels[1]:
            secondLevelMemory_simulatedData = pd.read_csv(
                path_prefix + second_level, names=all_cols_simulated, header=None)

            secondLevelMemory_simulatedData.label = 1
            secondLevelMemory_simulatedData = secondLevelMemory_simulatedData.loc[:, rounds_and_label]

            print('secondLevelMemory_simulatedData has been loaded!')

        # II. concatenate and shuffle validationData
        self.simulatedData = []
        if pooled_levels[0]:
            self.simulatedData.extend([firstLevelMemory_simulatedData])
        if pooled_levels[1]:
            self.simulatedData.extend([secondLevelMemory_simulatedData])
        

        self.simulatedData = pd.concat(self.simulatedData, ignore_index=True)
        #self.simulatedData = self.simulatedData.sample(frac=1)  # Shuffle #Dont shuffle!!!
        print("simulatedData: " + str(self.simulatedData.shape))

        # III Preparation of X_validation, y_validation, lda_X_validation and lda_y_validation

        ############
        # 2. Baseline Model: prepare X_lda, y_lda & trainingData_with_features
        
        game_lda_features = []
        game_cards_with_lda_features = []
        all_games_lda_features = []
        all_games_cards_with_lda_features = []
        for game in range(0, len(self.simulatedData)):
            # prepare the game lda features
            temp = self.simulatedData.iloc[game:game + 1, :]
            cards_used = ['card_' + str(i) for i in range(1, (num_of_rounds * 2) + 1)]
            lda_features_generator = FeaturesGenerator(game_log=temp.loc[:, cards_used],
                                                            sequence_length=num_of_rounds * 2)
            game_lda_features.append(lda_features_generator.get_undiscovered_cards())
            game_lda_features.append(lda_features_generator.get_unseen_cards())
            game_lda_features.append(lda_features_generator.get_max_seen_number())
            game_lda_features.append(lda_features_generator.get_zero_cards())

            if True in self.cardsCodes_statistics_features[1:]:  # i.e. statistical features == True
            # prepare statistical features
                cards_no_features = temp.loc[:, cards_used]  # .values.tolist()
                incremental_df = pd.DataFrame()
                incremental_df['card_1'] = cards_no_features['card_1']
                for card, _ in zip(cards_no_features, range(0, len(cards_no_features.columns))):
                    incremental_df[card] = cards_no_features[card]

                    lda_features_generator = FeaturesGenerator(game_log=incremental_df,
                                                                    sequence_length=len(incremental_df.columns))
                    undiscovered_cards = lda_features_generator.get_undiscovered_cards()
                    unseen_cards = lda_features_generator.get_unseen_cards()
                    max_seen_number = lda_features_generator.get_max_seen_number()
                    zero_cards = lda_features_generator.get_zero_cards()
                    card_code = math.floor(incremental_df.iloc[0][card] * 10) / 10
                    # features_used
                    all_features = [card_code, undiscovered_cards, unseen_cards, max_seen_number, zero_cards]
                    features_used = [all_features[i] for i, flag in enumerate(self.cardsCodes_statistics_features)
                                        if flag]
                    game_cards_with_lda_features.append(features_used)

            # append to all_games lists
            all_games_lda_features.append(game_lda_features)
            all_games_cards_with_lda_features.append(game_cards_with_lda_features)
            game_lda_features = []
            game_cards_with_lda_features = []

        # lda_X_validation
        self.lda_X_simulatedData = np.array(all_games_lda_features)
        # X_validation and y_validation
        if True in self.cardsCodes_statistics_features[1:]:
            self.X_simulatedData = np.array(all_games_cards_with_lda_features)
        else:
            self.X_simulatedData = np.array(self.simulatedData.loc[:, rounds])
            # in case of one feature (card code), reshape to ensure the (#samples, #timesteps, #features) for LSTM
            self.X_simulatedData = np.reshape(self.X_simulatedData, (self.X_simulatedData.shape[0], self.X_simulatedData.shape[1], 1))

        self.y_simulatedData = self.simulatedData.label
        self.y_simulatedData = np.array(self.y_simulatedData)
        # reshaping y_validation
        self.y_simulatedData = np.reshape(self.y_simulatedData, (self.y_simulatedData.shape[0], 1))
        # lda_y_validation
        self.lda_y_simulatedData = self.y_simulatedData
        # reshaping lda_y_validation
        self.lda_y_simulatedData = np.reshape(self.lda_y_simulatedData, (len(self.lda_y_simulatedData)))
        # one hot encoding for y_validation
        num_classes = 2
        self.y_simulatedData = to_categorical(self.y_simulatedData, num_classes=num_classes)

        print("X_simulatedData: " + str(np.shape(self.X_simulatedData)))
        print("y_simulatedData: " + str(np.shape(self.y_simulatedData)))

        print("lda_X_simulatedData: " + str(np.shape(self.lda_X_simulatedData)))
        print("lda_y_simulatedData: " + str(np.shape(self.lda_y_simulatedData)))

        np.save(base_path_features + 'simulatedData', self.simulatedData)
        np.save(base_path_features + 'X_simulatedData', self.X_simulatedData)
        np.save(base_path_features + 'y_simulatedData', self.y_simulatedData)
        np.save(base_path_features + 'lda_X_simulatedData', self.lda_X_simulatedData)
        np.save(base_path_features + 'lda_y_simulatedData', self.lda_y_simulatedData)

    def create_data_split(self, real, split_nr): 

        if real:
            self.Data_train = self.realData
            self.Data_train = self.Data_train.drop([split_nr - 1, split_nr - 1 + 20])
            self.Data_test = self.realData.iloc[[split_nr - 1, split_nr - 1 + 20], :]

            self.X_Data_train = np.delete(self.X_realData, [split_nr - 1, split_nr - 1 + 20], 0)
            self.y_Data_train = np.delete(self.y_realData, [split_nr - 1, split_nr - 1 + 20], 0)

            self.X_Data_test = self.X_realData[[split_nr - 1, split_nr - 1 + 20], :]
            self.y_Data_test = self.y_realData[[split_nr - 1, split_nr - 1 + 20], :]

            self.lda_X_Data_train = np.delete(self.lda_X_realData, [split_nr - 1, split_nr - 1 + 20], 0)
            self.lda_y_Data_train = np.delete(self.lda_y_realData, [split_nr - 1, split_nr - 1 + 20], 0)

            self.lda_X_Data_test = self.lda_X_realData[[split_nr - 1, split_nr - 1 + 20], :]
            self.lda_y_Data_test = np.take(self.lda_y_realData, [split_nr - 1, split_nr - 1 + 20], 0)   

        else:
            test_noObst_indices = list(range((split_nr - 1) * 1000, split_nr * 1000))
            test_glare_indices = list(range((split_nr - 1 + 20) * 1000, (split_nr + 20) * 1000))
            test_indices = test_noObst_indices
            test_indices.extend(test_glare_indices)

            self.Data_train = self.simulatedData
            self.Data_train = self.Data_train.drop(test_indices)

            self.X_Data_train = np.delete(self.X_simulatedData, test_indices, 0)
            self.y_Data_train = np.delete(self.y_simulatedData, test_indices, 0)
            
            self.lda_X_Data_train = np.delete(self.lda_X_simulatedData, test_indices, 0)
            self.lda_y_Data_train = np.delete(self.lda_y_simulatedData, test_indices, 0)

        
    def calculate_features(self, data, num_of_rounds, rounds, train):
        game_lda_features = []
        game_cards_with_lda_features = []
        all_games_lda_features = []
        all_games_cards_with_lda_features = []
        for game in range(0, len(data)):
            # prepare the game lda features
            temp = data.iloc[game:game + 1, :]
            cards_used = ['card_' + str(i) for i in range(1, (num_of_rounds * 2) + 1)]
            lda_features_generator = FeaturesGenerator(game_log=temp.loc[:, cards_used],
                                                            sequence_length=num_of_rounds * 2)
            game_lda_features.append(lda_features_generator.get_undiscovered_cards())
            game_lda_features.append(lda_features_generator.get_unseen_cards())
            game_lda_features.append(lda_features_generator.get_max_seen_number())
            game_lda_features.append(lda_features_generator.get_zero_cards())

            if True in self.cardsCodes_statistics_features[1:]:  # i.e. statistical features == True
            # prepare statistical features
                cards_no_features = temp.loc[:, cards_used]  # .values.tolist()
                incremental_df = pd.DataFrame()
                incremental_df['card_1'] = cards_no_features['card_1']
                for card, _ in zip(cards_no_features, range(0, len(cards_no_features.columns))):
                    incremental_df[card] = cards_no_features[card]

                    lda_features_generator = FeaturesGenerator(game_log=incremental_df,
                                                                    sequence_length=len(incremental_df.columns))
                    undiscovered_cards = lda_features_generator.get_undiscovered_cards()
                    unseen_cards = lda_features_generator.get_unseen_cards()
                    max_seen_number = lda_features_generator.get_max_seen_number()
                    zero_cards = lda_features_generator.get_zero_cards()
                    card_code = math.floor(incremental_df.iloc[0][card] * 10) / 10
                    # features_used
                    all_features = [card_code, undiscovered_cards, unseen_cards, max_seen_number, zero_cards]
                    features_used = [all_features[i] for i, flag in enumerate(self.cardsCodes_statistics_features)
                                        if flag]
                    game_cards_with_lda_features.append(features_used)

            # append to all_games lists
            all_games_lda_features.append(game_lda_features)
            all_games_cards_with_lda_features.append(game_cards_with_lda_features)
            game_lda_features = []
            game_cards_with_lda_features = []

        if train:
            # lda_X_validation
            self.lda_X_Data_train = np.array(all_games_lda_features)
            # X_validation and y_validation
            if True in self.cardsCodes_statistics_features[1:]:
                self.X_Data_train = np.array(all_games_cards_with_lda_features)
            else:
                self.X_Data_train = np.array(data.loc[:, rounds])
                # in case of one feature (card code), reshape to ensure the (#samples, #timesteps, #features) for LSTM
                self.X_Data_train = np.reshape(self.X_Data_train, (self.X_Data_train.shape[0], self.X_Data_train.shape[1], 1))

            self.y_Data_train = data.label
            self.y_Data_train = np.array(self.y_Data_train)
            # reshaping y_validation
            self.y_Data_train = np.reshape(self.y_Data_train, (self.y_Data_train.shape[0], 1))
            # lda_y_validation
            self.lda_y_Data_train = self.y_Data_train
            # reshaping lda_y_validation
            self.lda_y_Data_train = np.reshape(self.lda_y_Data_train, (len(self.lda_y_Data_train)))
            # one hot encoding for y_validation
            num_classes = 2
            self.y_Data_train = to_categorical(self.y_Data_train, num_classes=num_classes)

            print("X_Data_train: " + str(np.shape(self.X_Data_train)))
            print("y_Data_train: " + str(np.shape(self.y_Data_train)))

            print("lda_X_Data_train: " + str(np.shape(self.lda_X_Data_train)))
            print("lda_y_Data_train: " + str(np.shape(self.lda_y_Data_train)))
        else:
            # lda_X_validation
            self.lda_X_Data_test = np.array(all_games_lda_features)
            # X_validation and y_validation
            if True in self.cardsCodes_statistics_features[1:]:
                self.X_Data_test = np.array(all_games_cards_with_lda_features)
            else:
                self.X_Data_test = np.array(data.loc[:, rounds])
                # in case of one feature (card code), reshape to ensure the (#samples, #timesteps, #features) for LSTM
                self.X_Data_test = np.reshape(self.X_Data_test, (self.X_Data_test.shape[0], self.X_Data_test.shape[1], 1))

            self.y_Data_test = data.label
            self.y_Data_test = np.array(self.y_Data_test)
            # reshaping y_validation
            self.y_Data_test = np.reshape(self.y_Data_test, (self.y_Data_test.shape[0], 1))
            # lda_y_validation
            self.lda_y_Data_test = self.y_Data_test
            # reshaping lda_y_validation
            self.lda_y_Data_test = np.reshape(self.lda_y_Data_test, (len(self.lda_y_Data_test)))
            # one hot encoding for y_validation
            num_classes = 2
            self.y_Data_test = to_categorical(self.y_Data_test, num_classes=num_classes)

            print("X_Data_test: " + str(np.shape(self.X_Data_test)))
            print("y_Data_test: " + str(np.shape(self.y_Data_test)))

            print("lda_X_Data_test: " + str(np.shape(self.lda_X_Data_test)))
            print("lda_y_Data_test: " + str(np.shape(self.lda_y_Data_test)))

    def save_features(self, base_path_features, real):

        if not os.path.exists(base_path_features):
            os.mkdir(base_path_features)

        if real: 
            if not os.path.exists(base_path_features + 'for_simulation'):
                os.mkdir(base_path_features + 'for_simulation')

            if not os.path.exists(base_path_features + 'for_testing'):
                os.mkdir(base_path_features + 'for_testing')

        if real: 
            np.save(base_path_features + 'for_simulation\\' + 'realData_train', self.Data_train)
            np.save(base_path_features + 'for_testing\\' + 'realData_test', self.Data_test)

            np.save(base_path_features + 'for_simulation\\' + 'X_realData_train', self.X_Data_train)
            np.save(base_path_features + 'for_simulation\\' + 'y_realData_train', self.y_Data_train)
            np.save(base_path_features + 'for_testing\\' + 'X_realData_test', self.X_Data_test)
            np.save(base_path_features + 'for_testing\\' + 'y_realData_test', self.y_Data_test)

            np.save(base_path_features + 'for_simulation\\' + 'lda_X_realData_train', self.lda_X_Data_train)
            np.save(base_path_features + 'for_simulation\\' + 'lda_y_realData_train', self.lda_y_Data_train)
            np.save(base_path_features + 'for_testing\\' + 'lda_X_realData_test', self.lda_X_Data_test)
            np.save(base_path_features + 'for_testing\\' + 'lda_y_realData_test', self.lda_y_Data_test)
        else: 
            np.save(base_path_features + 'simulatedData_train', self.Data_train)

            np.save(base_path_features + 'X_simulatedData_train', self.X_Data_train)
            np.save(base_path_features + 'y_simulatedData_train', self.y_Data_train)

            np.save(base_path_features + 'lda_X_simulatedData_train', self.lda_X_Data_train)
            np.save(base_path_features + 'lda_y_simulatedData_train', self.lda_y_Data_train)


if __name__ == "__main__":
    num_of_subjects = 20
    prefix = 'C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\'
 
    # Features for all the data
    base_path_real_all = prefix + 'raw\\real\\all\\'
    base_path_real_features_all = prefix + 'features\\real\\all\\'
    base_path_simulated_all = prefix + 'raw\\simulated\\all\\'
    base_path_simulated_features_all = prefix + 'features\\simulated\\all\\'
    feature_generator = Data_preparator()
    feature_generator.load_and_prepare_real_data_all(base_path_real_all, base_path_real_features_all)
    feature_generator.load_and_prepare_simulated_data_all(base_path_simulated_all, base_path_simulated_features_all)

    # Feature splits for real and simulated data 
    for split in range(1, num_of_subjects + 1):
        base_path_real = prefix + 'raw\\real\\' + 'Split' + str(split) + '\\'
        base_path_features_real = prefix + 'features\\real\\' + 'Split' + str(split) + '\\'
        base_path_simulated = prefix + 'raw\\simulated\\' + 'Split' + str(split) + '\\'
        base_path_features_simulated = prefix + 'features\\simulated\\' + 'Split' + str(split) + '\\'
        feature_generator.create_data_split(True, split)
        feature_generator.save_features(base_path_features_real, real=True)
        feature_generator.create_data_split(False, split)
        feature_generator.save_features(base_path_features_simulated, real=False)
