#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

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
                 cardsCodes_statistics_features=[True, True, True, True, True],
                 ):

        self.sequence_length = sequence_length
        self.cardsCodes_statistics_features = cardsCodes_statistics_features 

    def load_and_prepare_real_data(self, two_labels=True,
                                    num_of_rounds=20,
                                    pooled_levels=[True, True, False, False, False, False, False,
                                                    False, False],
                                    base_path='C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\mapped_logs\\glare_effect_noObst_valid_logs\\',
                                    base_path_features='C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\real\\'):

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
        self.realData = self.realData.sample(frac=1)  # Shuffle
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

if __name__ == "__main__":
    feature_generator = Data_preparator()
    feature_generator.load_and_prepare_real_data()
