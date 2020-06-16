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

    def load_and_prepare_real_data_splits(self,
                                    real, 
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
        # columns
        similarity_cols_noObst = ['sim_' + str(i) for i in range(1, 22)]
        similarity_cols_glare = ['sim_' + str(i) for i in range(1, 29)]
        cards_cols = ['card_' + str(i) for i in range(1, 41)]
        timestamps_cols = ['timestamp_' + str(i) for i in range(1, 41)]
        metrics_cols = ['metric_' + str(i) for i in range(1, 5)]
        label_col = ['label']
        rmse = ['rmse']
        all_cols_noObst = similarity_cols_noObst + cards_cols + timestamps_cols + metrics_cols + label_col
        all_cols_glare = similarity_cols_glare + cards_cols + timestamps_cols + metrics_cols + label_col
        all_cols_simulated = rmse + cards_cols + metrics_cols + label_col

        if not real:
            all_cols_noObst = all_cols_simulated
            all_cols_glare = all_cols_simulated

            first_level_filename = 'FirstLevelMemoryTrainingData.log'
            second_level_filename = 'SecondLevelMemoryTrainingData.log'
        else:
            first_level_filename = 'firstLevel_realData_memory.log'
            second_level_filename = 'secondLevel_realData_memory.log'

        # I. load realData
        rounds = ['card_' + str(i) for i in range(1, (num_of_rounds * 2) + 1)]
        rounds_and_label = rounds + ['label']

        if pooled_levels[0]:
            firstLevelMemory_realData_train = pd.read_csv(
                path_prefix + 'for_simulation\\separated_levels\\%s' %first_level_filename, names=all_cols_noObst, header=None)

            firstLevelMemory_realData_train.label = 0
            firstLevelMemory_realData_train = firstLevelMemory_realData_train.loc[:, rounds_and_label]

            firstLevelMemory_realData_test = pd.read_csv(
                path_prefix + 'for_testing\\separated_levels\\%s' %first_level_filename, names=all_cols_noObst, header=None)

            firstLevelMemory_realData_test.label = 0
            firstLevelMemory_realData_test = firstLevelMemory_realData_test.loc[:, rounds_and_label]

            print('firstLevelMemory_realData_train has been loaded!')

        # if (not two_labels) or memory_obstacle_newApp:
        if pooled_levels[1]:
            secondLevelMemory_realData_train = pd.read_csv(
                path_prefix + 'for_simulation\\separated_levels\\%s' %second_level_filename, names=all_cols_glare, header=None)

            secondLevelMemory_realData_train.label = 1
            secondLevelMemory_realData_train = secondLevelMemory_realData_train.loc[:, rounds_and_label]

            secondLevelMemory_realData_test = pd.read_csv(
                path_prefix + 'for_testing\\separated_levels\\%s' %second_level_filename, names=all_cols_glare, header=None)

            secondLevelMemory_realData_test.label = 1
            secondLevelMemory_realData_test = secondLevelMemory_realData_test.loc[:, rounds_and_label]

            print('secondLevelMemory_realData_train has been loaded!')

        # II. concatenate and shuffle validationData
        self.realData_train = []
        self.realData_test = []
        if pooled_levels[0]:
            self.realData_train.extend([firstLevelMemory_realData_train])
            self.realData_test.extend([firstLevelMemory_realData_test])
        if pooled_levels[1]:
            self.realData_train.extend([secondLevelMemory_realData_train])
            self.realData_test.extend([secondLevelMemory_realData_test])
        
        self.realData_train = pd.concat(self.realData_train, ignore_index=True)
        self.realData_train = self.realData_train.sample(frac=1)  # Shuffle
        print("realData_train: " + str(self.realData_train.shape))

        self.realData_test = pd.concat(self.realData_test, ignore_index=True)
        self.realData_test = self.realData_test.sample(frac=1)  # Shuffle
        print("realData_test: " + str(self.realData_test.shape))

        self.calculate_features_real(self.realData_train, 20, rounds, True)
        self.calculate_features_real(self.realData_test, 20, rounds, False)

        
    def calculate_features_real(self, real_data, num_of_rounds, rounds, train):
        game_lda_features = []
        game_cards_with_lda_features = []
        all_games_lda_features = []
        all_games_cards_with_lda_features = []
        for game in range(0, len(real_data)):
            # prepare the game lda features
            temp = real_data.iloc[game:game + 1, :]
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
            self.lda_X_realData_train = np.array(all_games_lda_features)
            # X_validation and y_validation
            if True in self.cardsCodes_statistics_features[1:]:
                self.X_realData_train = np.array(all_games_cards_with_lda_features)
            else:
                self.X_realData_train = np.array(real_data.loc[:, rounds])
                # in case of one feature (card code), reshape to ensure the (#samples, #timesteps, #features) for LSTM
                self.X_realData_train = np.reshape(self.X_realData_train, (self.X_realData_train.shape[0], self.X_realData_train.shape[1], 1))

            self.y_realData_train = real_data.label
            self.y_realData_train = np.array(self.y_realData_train)
            # reshaping y_validation
            self.y_realData_train = np.reshape(self.y_realData_train, (self.y_realData_train.shape[0], 1))
            # lda_y_validation
            self.lda_y_realData_train = self.y_realData_train
            # reshaping lda_y_validation
            self.lda_y_realData_train = np.reshape(self.lda_y_realData_train, (len(self.lda_y_realData_train)))
            # one hot encoding for y_validation
            num_classes = 2
            self.y_realData_train = to_categorical(self.y_realData_train, num_classes=num_classes)

            print("X_realData_train: " + str(np.shape(self.X_realData_train)))
            print("y_realData_train: " + str(np.shape(self.y_realData_train)))

            print("lda_X_realData_train: " + str(np.shape(self.lda_X_realData_train)))
            print("lda_y_realData_train: " + str(np.shape(self.lda_y_realData_train)))
        else:
            # lda_X_validation
            self.lda_X_realData_test = np.array(all_games_lda_features)
            # X_validation and y_validation
            if True in self.cardsCodes_statistics_features[1:]:
                self.X_realData_test = np.array(all_games_cards_with_lda_features)
            else:
                self.X_realData_test = np.array(real_data.loc[:, rounds])
                # in case of one feature (card code), reshape to ensure the (#samples, #timesteps, #features) for LSTM
                self.X_realData_test = np.reshape(self.X_realData_test, (self.X_realData_test.shape[0], self.X_realData_test.shape[1], 1))

            self.y_realData_test = real_data.label
            self.y_realData_test = np.array(self.y_realData_test)
            # reshaping y_validation
            self.y_realData_test = np.reshape(self.y_realData_test, (self.y_realData_test.shape[0], 1))
            # lda_y_validation
            self.lda_y_realData_test = self.y_realData_test
            # reshaping lda_y_validation
            self.lda_y_realData_test = np.reshape(self.lda_y_realData_test, (len(self.lda_y_realData_test)))
            # one hot encoding for y_validation
            num_classes = 2
            self.y_realData_test = to_categorical(self.y_realData_test, num_classes=num_classes)

            print("X_realData_test: " + str(np.shape(self.X_realData_test)))
            print("y_realData_test: " + str(np.shape(self.y_realData_test)))

            print("lda_X_realData_test: " + str(np.shape(self.lda_X_realData_test)))
            print("lda_y_realData_test: " + str(np.shape(self.lda_y_realData_test)))


    def load_and_prepare_simulated_data_splits(self, two_labels=True,
                                    num_of_rounds=20,
                                    pooled_levels=[True, True, False, False, False, False, False,
                                                    False, False]):
        pass

    def save_features(self, base_path_features, real):

        if real: 
            for split in range(1, self.num_of_subjects+1):
                # create output folders
                if not os.path.exists(base_path_features):
                    os.mkdir(base_path_features)

                if not os.path.exists(base_path_features + '/for_simulation'):
                    #os.mkdir(base_path_features)
                    os.mkdir(base_path_features + '/for_simulation')
                    #os.mkdir(base_path_features + '/for_simulation/separated_levels')

                if not os.path.exists(base_path_features + '/for_testing'):
                    os.mkdir(base_path_features + '/for_testing')
                    #os.mkdir(base_path_features + '/for_testing/separated_levels')

            np.save(base_path_features + '/for_simulation/' + 'realData_train', self.realData_train)
            np.save(base_path_features + '/for_testing/' + 'realData_test', self.realData_test)

            np.save(base_path_features + '/for_simulation/' + 'X_realData_train', self.X_realData_train)
            np.save(base_path_features + '/for_simulation/' + 'y_realData_train', self.y_realData_train)
            np.save(base_path_features + '/for_testing/' + 'X_realData_test', self.X_realData_test)
            np.save(base_path_features + '/for_testing/' + 'y_realData_test', self.y_realData_test)

            np.save(base_path_features + '/for_simulation/' + 'lda_X_realData_train', self.lda_X_realData_train)
            np.save(base_path_features + '/for_simulation/' + 'lda_y_realData_train', self.lda_y_realData_train)
            np.save(base_path_features + '/for_testing/' + 'lda_X_realData_test', self.lda_X_realData_test)
            np.save(base_path_features + '/for_testing/' + 'lda_y_realData_test', self.lda_y_realData_test)
        else: 
            for split in range(1, self.num_of_subjects+1):
                # create output folders
                if not os.path.exists(base_path_features):
                    os.mkdir(base_path_features)

                if not os.path.exists(base_path_features + '/for_simulation'):
                    #os.mkdir(base_path_features)
                    os.mkdir(base_path_features + '/for_simulation')
                    #os.mkdir(base_path_features + '/for_simulation/separated_levels')

                if not os.path.exists(base_path_features + '/for_testing'):
                    os.mkdir(base_path_features + '/for_testing')
                    #os.mkdir(base_path_features + '/for_testing/separated_levels')

            np.save(base_path_features + '/for_simulation/' + 'realData_train', self.realData_train)
            np.save(base_path_features + '/for_testing/' + 'realData_test', self.realData_test)

            np.save(base_path_features + '/for_simulation/' + 'X_realData_train', self.X_realData_train)
            np.save(base_path_features + '/for_simulation/' + 'y_realData_train', self.y_realData_train)

            np.save(base_path_features + '/for_simulation/' + 'lda_X_realData_train', self.lda_X_realData_train)
            np.save(base_path_features + '/for_simulation/' + 'lda_y_realData_train', self.lda_y_realData_train)
            
    

if __name__ == "__main__":
    num_of_subjects = 20
    prefix = 'C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\'

    # All features for real data 
    base_path_real_all = 'C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\raw\\real\\all\\',
    base_path_real_features_all ='C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\features\\real\\all\\'
    base_path_simulated_all ='C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\raw\\simulated\\all\\',
    base_path_simulated_features_all ='C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\features\\simulated\\all\\'

    # Features for all the data
    #TODO: Add methods from other script. 

    # Feature splits for real and simulated data 
    for split in range(1, num_of_subjects + 1):
        base_path_real = prefix + 'raw\\real\\' + 'Split' + str(split) + '\\'
        base_path_features_real = prefix + 'features\\real\\' + 'Split' + str(split) + '\\'
        base_path_simulated = prefix + 'raw\\simulated\\' + 'Split' + str(split) + '\\'
        base_path_features_simulated = prefix + 'features\\simulated\\' + 'Split' + str(split) + '\\'
        feature_generator = Data_preparator()
        feature_generator.load_and_prepare_real_data_splits(True, base_path_real, base_path_features_real)
        #feature_generator.load_and_prepare_simulated_data_splits(base_path_simulated, base_path_features_simulated)
        feature_generator.save_features(base_path_features_real, real=True)
        #feature_generator.save_features(base_path_features_simulated, real=False)
