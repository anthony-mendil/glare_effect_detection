# This is a version of memory_data_analysis.py adjusted by Anthony Mendil in 2020
# in the context of a bachelor thesis regarding the glare effect. Only small changes were made. 

import numpy
import pandas as pd
import pickle

from pylab import *

from scipy.stats import ttest_ind

from itertools import product

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import os


class MemoryDataAnalyzer():
    """This class encapsulates multiple memory data analysis functions:
     1. to analyse memory-based data by calculating two statistics on them and to store then the statistics results
     2. to load already stored statistics and plot the results
     3. to compare between real-data-based memory levels using t-test
     4. to evaluate the similarity between simulated-data-based and real-data-based memory levels 
    
    Members/Attributes:
    sequence_length: 40 e.g. means first 40 chosen cards (first 20 rounds)
    numOfRounds: a list of rounds, e.g. [1,20]
    cards: a list of cards' headers: e.g. for sequence_length = 20, we have card_1,...card_40
    """

    def __init__(self, sequence_length=40):
        """
        Returns a data_analyzer instance whose 
        :param sequence_length: the number of chosen cards to be considered in the intended statistics  
        """
        self.sequence_length = sequence_length
        self.numOfRounds = [i for i in range(1, (int(self.sequence_length/2) + 1))]
        self.cards = ['card_' + str(i) for i in range(1, sequence_length+1)] # card_1...card_40

    def sort_simulated_data_quality(self, real_game, simulated_games):
        """
        This function sorts the simulated games regarding to how close they are to their reference :param real_game
        :param real_game: df contains of one real game
        :param simulated_games: df of multiple real game
        :return: sorted simulated_games, from best to worst, it has the following format: list((RMSE, sim_game_df))
        """
        # sim columns
        cards_cols = ['card_' + str(i) for i in range(1, 41)]
        metrics_cols = ['metric_' + str(i) for i in range(1, 5)]
        label_col = ['label']
        all_cols = cards_cols + metrics_cols + label_col

        sorted_simulated_games = []
        for sim_game in simulated_games.to_numpy():
            sim_game = pd.DataFrame([sim_game], columns=all_cols)
            real_game_matchingPairs, real_game_penalties, \
            sim_game_matchingPairs, sim_game_penalties = self.doStatistics(real_game, sim_game, max_num_rounds=10)

            matchingPars_rmse = self.calc_lists_rmse(real_game_matchingPairs, sim_game_matchingPairs)
            penalties_rmse = self.calc_lists_rmse(real_game_penalties, sim_game_penalties)
            sim_game_error = (matchingPars_rmse + penalties_rmse) / 2.0

            sorted_simulated_games.append((sim_game_error, sim_game))

        # do sorting
        sorted_simulated_games.sort(key=lambda err_game: err_game[0])
        return sorted_simulated_games


    def analyzeData(self,
                    data_1,
                    data_1_title,
                    data_2,
                    data_2_title,
                    data_3,
                    data_3_title,
                    data_4,
                    data_4_title,
                    metric_1_path = '/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                    metric_2_path = '/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/',
                    plt_prefix='mg'):
        """This function analyzes the given data and plots results according to different metrics"""
        # do statistics
        data_1_numofMatchingCards_allGames, data_1_ExploitingScore_allGames, data_2_numofMatchingCards_allGames, data_2_ExploitingScore_allGames = self.doStatistics(data_1, data_2)
        data_3_numofMatchingCards_allGames, data_3_ExploitingScore_allGames, data_4_numofMatchingCards_allGames, data_4_ExploitingScore_allGames = self.doStatistics(data_3, data_4)

        # statistics: further preparation & storage
        data_1_numofMatchingCards_allGames = numpy.array(data_1_numofMatchingCards_allGames)
        data_1_numofMatchingCards_mean = numpy.mean(data_1_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_1_title, 'wb') as fp:
            pickle.dump(data_1_numofMatchingCards_mean, fp)

        data_1_ExploitingScore_allGames = numpy.array(data_1_ExploitingScore_allGames)
        data_1_ExploitingScore_mean = numpy.mean(data_1_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_1_title, 'wb') as fp:
            pickle.dump(data_1_ExploitingScore_mean, fp)

        data_2_numofMatchingCards_allGames = numpy.array(data_2_numofMatchingCards_allGames)
        data_2_numofMatchingCards_mean = numpy.mean(data_2_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_2_title, 'wb') as fp:
            pickle.dump(data_2_numofMatchingCards_mean, fp)

        data_2_ExploitingScore_allGames = numpy.array(data_2_ExploitingScore_allGames)
        data_2_ExploitingScore_mean = numpy.mean(data_2_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_2_title, 'wb') as fp:
            pickle.dump(data_2_ExploitingScore_mean, fp)

        data_3_numofMatchingCards_allGames = numpy.array(data_3_numofMatchingCards_allGames)
        data_3_numofMatchingCards_mean = numpy.mean(data_3_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_3_title, 'wb') as fp:
            pickle.dump(data_3_numofMatchingCards_mean, fp)

        data_3_ExploitingScore_allGames = numpy.array(data_3_ExploitingScore_allGames)
        data_3_ExploitingScore_mean = numpy.mean(data_3_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_3_title, 'wb') as fp:
            pickle.dump(data_3_ExploitingScore_mean, fp)

        data_4_numofMatchingCards_allGames = numpy.array(data_4_numofMatchingCards_allGames)
        data_4_numofMatchingCards_mean = numpy.mean(data_4_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_4_title, 'wb') as fp:
            pickle.dump(data_4_numofMatchingCards_mean, fp)

        data_4_ExploitingScore_allGames = numpy.array(data_4_ExploitingScore_allGames)
        data_4_ExploitingScore_mean = numpy.mean(data_4_ExploitingScore_allGames, axis=0)
        with open(metric_2_path + data_4_title, 'wb') as fp:
            pickle.dump(data_4_ExploitingScore_mean, fp)
        # Do plotting
        self.plot_statistics(data_1_numofMatchingCards_mean, data_2_numofMatchingCards_mean, data_3_numofMatchingCards_mean,
                        data_4_numofMatchingCards_mean, data_1_ExploitingScore_mean, data_2_ExploitingScore_mean,
                        data_3_ExploitingScore_mean, data_4_ExploitingScore_mean, metric_1_path, metric_2_path,
                             plt_prefix=plt_prefix)

    def analyzeData_with_std(self,
                             data_1,
                             data_1_title,
                             data_2,
                             data_2_title,
                             data_3,
                             data_3_title,
                             data_4,
                             data_4_title,
                             metric_1_path = '/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                             metric_2_path = '/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/',
                             plt_prefix='mg'):
        """This function analyzes the given data and plots results according to different metrics"""
        # do statistics
        data_1_numofMatchingCards_allGames, data_1_ExploitingScore_allGames, data_2_numofMatchingCards_allGames, data_2_ExploitingScore_allGames = self.doStatistics(data_1, data_2)
        data_3_numofMatchingCards_allGames, data_3_ExploitingScore_allGames, data_4_numofMatchingCards_allGames, data_4_ExploitingScore_allGames = self.doStatistics(data_3, data_4)

        # statistics: further preparation & storage
        data_1_numofMatchingCards_allGames = numpy.array(data_1_numofMatchingCards_allGames)
        data_1_numofMatchingCards_mean = numpy.mean(data_1_numofMatchingCards_allGames, axis=0)
        data_1_numofMatchingCards_std = numpy.std(data_1_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_1_title, 'wb') as fp:
            pickle.dump(data_1_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_1_title+'_std', 'wb') as fp:
            pickle.dump(data_1_numofMatchingCards_std, fp)

        data_1_ExploitingScore_allGames = numpy.array(data_1_ExploitingScore_allGames)
        data_1_ExploitingScore_mean = numpy.mean(data_1_ExploitingScore_allGames, axis=0)
        data_1_ExploitingScore_std = numpy.std(data_1_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_1_title, 'wb') as fp:
            pickle.dump(data_1_ExploitingScore_mean, fp)
        with open(metric_2_path+data_1_title+'_std', 'wb') as fp:
            pickle.dump(data_1_ExploitingScore_std, fp)

        data_2_numofMatchingCards_allGames = numpy.array(data_2_numofMatchingCards_allGames)
        data_2_numofMatchingCards_mean = numpy.mean(data_2_numofMatchingCards_allGames, axis=0)
        data_2_numofMatchingCards_std = numpy.std(data_2_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_2_title, 'wb') as fp:
            pickle.dump(data_2_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_2_title+'_std', 'wb') as fp:
            pickle.dump(data_2_numofMatchingCards_std, fp)

        data_2_ExploitingScore_allGames = numpy.array(data_2_ExploitingScore_allGames)
        data_2_ExploitingScore_mean = numpy.mean(data_2_ExploitingScore_allGames, axis=0)
        data_2_ExploitingScore_std = numpy.std(data_2_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_2_title, 'wb') as fp:
            pickle.dump(data_2_ExploitingScore_mean, fp)
        with open(metric_2_path+data_2_title+'_std', 'wb') as fp:
            pickle.dump(data_2_ExploitingScore_std, fp)

        data_3_numofMatchingCards_allGames = numpy.array(data_3_numofMatchingCards_allGames)
        data_3_numofMatchingCards_mean = numpy.mean(data_3_numofMatchingCards_allGames, axis=0)
        data_3_numofMatchingCards_std = numpy.std(data_3_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_3_title, 'wb') as fp:
            pickle.dump(data_3_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_3_title+'_std', 'wb') as fp:
            pickle.dump(data_3_numofMatchingCards_std, fp)

        data_3_ExploitingScore_allGames = numpy.array(data_3_ExploitingScore_allGames)
        data_3_ExploitingScore_mean = numpy.mean(data_3_ExploitingScore_allGames, axis=0)
        data_3_ExploitingScore_std = numpy.std(data_3_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_3_title, 'wb') as fp:
            pickle.dump(data_3_ExploitingScore_mean, fp)
        with open(metric_2_path+data_3_title+'_std', 'wb') as fp:
            pickle.dump(data_3_ExploitingScore_std, fp)

        data_4_numofMatchingCards_allGames = numpy.array(data_4_numofMatchingCards_allGames)
        data_4_numofMatchingCards_mean = numpy.mean(data_4_numofMatchingCards_allGames, axis=0)
        data_4_numofMatchingCards_std = numpy.std(data_4_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_4_title, 'wb') as fp:
            pickle.dump(data_4_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_4_title+'_std', 'wb') as fp:
            pickle.dump(data_4_numofMatchingCards_std, fp)

        data_4_ExploitingScore_allGames = numpy.array(data_4_ExploitingScore_allGames)
        data_4_ExploitingScore_mean = numpy.mean(data_4_ExploitingScore_allGames, axis=0)
        data_4_ExploitingScore_std = numpy.std(data_4_ExploitingScore_allGames, axis=0)
        with open(metric_2_path + data_4_title, 'wb') as fp:
            pickle.dump(data_4_ExploitingScore_mean, fp)
        with open(metric_2_path + data_4_title+'_std', 'wb') as fp:
            pickle.dump(data_4_ExploitingScore_std, fp)

        # Do plotting
        self.plot_statistics_with_std(data_1_numofMatchingCards_mean, data_1_numofMatchingCards_mean, data_3_numofMatchingCards_mean,
                                      data_3_numofMatchingCards_mean, data_1_ExploitingScore_mean, data_2_ExploitingScore_mean,
                                      data_3_ExploitingScore_mean, data_4_ExploitingScore_mean,

                                      data_1_numofMatchingCards_std, data_1_numofMatchingCards_std,
                                      data_3_numofMatchingCards_std,
                                      data_3_numofMatchingCards_std, data_1_ExploitingScore_std,
                                      data_2_ExploitingScore_std,
                                      data_3_ExploitingScore_std, data_4_ExploitingScore_std)

    def analyzeData_visualObstacle(self,
                                   data_1,
                                   data_1_title,
                                   data_2,
                                   data_2_title,
                                   data_3,
                                   data_3_title,
                                   data_4,
                                   data_4_title,
                                   metric_1_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                                   metric_2_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/',
                                   plotted_levels=[True, True, True, True],
                                   do_scaling=False,
                                   data_1_num_pairs=7,
                                   data_2_num_pairs=7,
                                   data_3_num_pairs=7,
                                   data_4_num_pairs=7,
                                   max_num_rounds=20,
                                   matching_pairs_fig_path='',
                                   penalties_fig_path=''
                                   ):
        """This function analyzes the given data and plots results according to different metrics"""
        # do statistics
        data_1_numofMatchingCards_allGames, data_1_ExploitingScore_allGames, data_2_numofMatchingCards_allGames, data_2_ExploitingScore_allGames = self.doStatistics(data_1, data_2, max_num_rounds=max_num_rounds)
        data_3_numofMatchingCards_allGames, data_3_ExploitingScore_allGames, data_4_numofMatchingCards_allGames, data_4_ExploitingScore_allGames = self.doStatistics(data_3, data_4, max_num_rounds=max_num_rounds)

        # statistics: further preparation & storage
        data_1_numofMatchingCards_allGames = numpy.array(data_1_numofMatchingCards_allGames)
        data_1_numofMatchingCards_mean = numpy.mean(data_1_numofMatchingCards_allGames, axis=0)
        data_1_numofMatchingCards_std = numpy.std(data_1_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_1_title, 'wb') as fp:
            pickle.dump(data_1_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_1_title+'_std', 'wb') as fp:
            pickle.dump(data_1_numofMatchingCards_std, fp)

        data_1_ExploitingScore_allGames = numpy.array(data_1_ExploitingScore_allGames)
        data_1_ExploitingScore_mean = numpy.mean(data_1_ExploitingScore_allGames, axis=0)
        data_1_ExploitingScore_std = numpy.std(data_1_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_1_title, 'wb') as fp:
            pickle.dump(data_1_ExploitingScore_mean, fp)
        with open(metric_2_path+data_1_title+'_std', 'wb') as fp:
            pickle.dump(data_1_ExploitingScore_std, fp)

        data_2_numofMatchingCards_allGames = numpy.array(data_2_numofMatchingCards_allGames)
        data_2_numofMatchingCards_mean = numpy.mean(data_2_numofMatchingCards_allGames, axis=0)
        data_2_numofMatchingCards_std = numpy.std(data_2_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_2_title, 'wb') as fp:
            pickle.dump(data_2_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_2_title+'_std', 'wb') as fp:
            pickle.dump(data_2_numofMatchingCards_std, fp)

        data_2_ExploitingScore_allGames = numpy.array(data_2_ExploitingScore_allGames)
        data_2_ExploitingScore_mean = numpy.mean(data_2_ExploitingScore_allGames, axis=0)
        data_2_ExploitingScore_std = numpy.std(data_2_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_2_title, 'wb') as fp:
            pickle.dump(data_2_ExploitingScore_mean, fp)
        with open(metric_2_path+data_2_title+'_std', 'wb') as fp:
            pickle.dump(data_2_ExploitingScore_std, fp)

        data_3_numofMatchingCards_allGames = numpy.array(data_3_numofMatchingCards_allGames)
        data_3_numofMatchingCards_mean = numpy.mean(data_3_numofMatchingCards_allGames, axis=0)
        data_3_numofMatchingCards_std = numpy.std(data_3_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_3_title, 'wb') as fp:
            pickle.dump(data_3_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_3_title+'_std', 'wb') as fp:
            pickle.dump(data_3_numofMatchingCards_std, fp)

        data_3_ExploitingScore_allGames = numpy.array(data_3_ExploitingScore_allGames)
        data_3_ExploitingScore_mean = numpy.mean(data_3_ExploitingScore_allGames, axis=0)
        data_3_ExploitingScore_std = numpy.std(data_3_ExploitingScore_allGames, axis=0)
        with open(metric_2_path+data_3_title, 'wb') as fp:
            pickle.dump(data_3_ExploitingScore_mean, fp)
        with open(metric_2_path+data_3_title+'_std', 'wb') as fp:
            pickle.dump(data_3_ExploitingScore_std, fp)

        data_4_numofMatchingCards_allGames = numpy.array(data_4_numofMatchingCards_allGames)
        data_4_numofMatchingCards_mean = numpy.mean(data_4_numofMatchingCards_allGames, axis=0)
        data_4_numofMatchingCards_std = numpy.std(data_4_numofMatchingCards_allGames, axis=0)
        with open(metric_1_path+data_4_title, 'wb') as fp:
            pickle.dump(data_4_numofMatchingCards_mean, fp)
        with open(metric_1_path+data_4_title+'_std', 'wb') as fp:
            pickle.dump(data_4_numofMatchingCards_std, fp)

        data_4_ExploitingScore_allGames = numpy.array(data_4_ExploitingScore_allGames)
        data_4_ExploitingScore_mean = numpy.mean(data_4_ExploitingScore_allGames, axis=0)
        data_4_ExploitingScore_std = numpy.std(data_4_ExploitingScore_allGames, axis=0)
        with open(metric_2_path + data_4_title, 'wb') as fp:
            pickle.dump(data_4_ExploitingScore_mean, fp)
        with open(metric_2_path + data_4_title+'_std', 'wb') as fp:
            pickle.dump(data_4_ExploitingScore_std, fp)

        # Do plotting or return mse
        # if not calc_err:
        if not do_scaling:
            self.plot_visualObstacle_statistics(data_1_numofMatchingCards_mean,
                                                data_2_numofMatchingCards_mean,
                                                data_3_numofMatchingCards_mean,
                                                data_4_numofMatchingCards_mean,

                                                data_1_ExploitingScore_mean,
                                                data_2_ExploitingScore_mean,
                                                data_3_ExploitingScore_mean,
                                                data_4_ExploitingScore_mean,

                                                data_1_label=data_1_title,
                                                data_2_label=data_2_title,
                                                data_3_label=data_3_title,
                                                data_4_label=data_4_title,

                                                plotted_levels=plotted_levels,
                                                max_num_rounds=max_num_rounds,
                                                matching_pairs_fig_path=matching_pairs_fig_path,
                                                penalties_fig_path=penalties_fig_path)
            # With whiskers:
            self.plot_visualObstacle_statistics_whiskered(data_1_numofMatchingCards_mean,
                                                          data_2_numofMatchingCards_mean,
                                                          data_3_numofMatchingCards_mean,
                                                          data_4_numofMatchingCards_mean,

                                                          data_1_numofMatchingCards_std,
                                                          data_2_numofMatchingCards_std,
                                                          data_3_numofMatchingCards_std,
                                                          data_4_numofMatchingCards_std,
        
                                                          data_1_ExploitingScore_mean,
                                                          data_2_ExploitingScore_mean,
                                                          data_3_ExploitingScore_mean,
                                                          data_4_ExploitingScore_mean,

                                                          data_1_ExploitingScore_std,
                                                          data_2_ExploitingScore_std,
                                                          data_3_ExploitingScore_std,
                                                          data_4_ExploitingScore_std,
        
                                                          data_1_label=data_1_title,
                                                          data_2_label=data_2_title,
                                                          data_3_label=data_3_title,
                                                          data_4_label=data_4_title,
        
                                                          plotted_levels=plotted_levels,
                                                          max_num_rounds=max_num_rounds,
                                                          matching_pairs_fig_path=matching_pairs_fig_path,
                                                          penalties_fig_path=penalties_fig_path)
            
        else:
            self.plot_visualObstacle_statistics([mean / data_1_num_pairs * 100 for mean in data_1_numofMatchingCards_mean],
                                                [mean / data_2_num_pairs * 100 for mean in data_2_numofMatchingCards_mean],
                                                [mean / data_3_num_pairs * 100 for mean in data_3_numofMatchingCards_mean],
                                                [mean / data_4_num_pairs * 100 for mean in data_4_numofMatchingCards_mean],

                                                data_1_ExploitingScore_mean,
                                                data_2_ExploitingScore_mean,
                                                #[mean * data_1_num_pairs / data_3_num_pairs for mean in
                                                data_3_ExploitingScore_mean,#],
                                                #[mean * data_1_num_pairs / data_4_num_pairs for mean in
                                                data_4_ExploitingScore_mean,#],

                                                data_1_label=data_1_title,
                                                data_2_label=data_2_title,
                                                data_3_label=data_3_title,
                                                data_4_label=data_4_title,

                                                plotted_levels=plotted_levels,
                                                max_num_rounds=max_num_rounds,
                                                matching_pairs_fig_path=matching_pairs_fig_path,
                                                penalties_fig_path=penalties_fig_path
                                                )
        
        # else:
        # RMSE
        matching_score_data1_data2_rmse = self.calc_lists_rmse(data_1_numofMatchingCards_mean,
                                                               data_2_numofMatchingCards_mean)
        matching_score_data1_data3_rmse = self.calc_lists_rmse(data_1_numofMatchingCards_mean,
                                                               data_3_numofMatchingCards_mean)

        punishments_score_data1_data2_rmse = self.calc_lists_rmse(data_1_ExploitingScore_mean,
                                                                  data_2_ExploitingScore_mean)
        punishments_score_data1_data3_rmse = self.calc_lists_rmse(data_1_ExploitingScore_mean,
                                                                  data_3_ExploitingScore_mean)

        matching_score_data2_data3_rmse = self.calc_lists_rmse(data_2_numofMatchingCards_mean,
                                                               data_3_numofMatchingCards_mean)
        matching_score_data2_data4_rmse = self.calc_lists_rmse(data_2_numofMatchingCards_mean,
                                                               data_4_numofMatchingCards_mean)

        punishments_score_data2_data3_rmse = self.calc_lists_rmse(data_2_ExploitingScore_mean,
                                                                  data_3_ExploitingScore_mean)
        punishments_score_data2_data4_rmse = self.calc_lists_rmse(data_2_ExploitingScore_mean,
                                                                  data_4_ExploitingScore_mean)

        # MAR: Mean Absolute Error
        matching_score_data1_data2_mae = mean_absolute_error(data_1_numofMatchingCards_mean,
                                                             data_2_numofMatchingCards_mean)
        matching_score_data1_data3_mae = mean_absolute_error(data_1_numofMatchingCards_mean,
                                                             data_3_numofMatchingCards_mean)

        punishments_score_data1_data2_mae = mean_absolute_error(data_1_ExploitingScore_mean,
                                                                data_2_ExploitingScore_mean)
        punishments_score_data1_data3_mae = mean_absolute_error(data_1_ExploitingScore_mean,
                                                                data_3_ExploitingScore_mean)

        matching_score_data2_data3_mae = mean_absolute_error(data_2_numofMatchingCards_mean,
                                                             data_3_numofMatchingCards_mean)
        matching_score_data2_data4_mae = mean_absolute_error(data_2_numofMatchingCards_mean,
                                                             data_4_numofMatchingCards_mean)

        punishments_score_data2_data3_mae = mean_absolute_error(data_2_ExploitingScore_mean,
                                                                data_3_ExploitingScore_mean)
        punishments_score_data2_data4_mae = mean_absolute_error(data_2_ExploitingScore_mean,
                                                                data_4_ExploitingScore_mean)

        return data_1_numofMatchingCards_mean,\
               data_2_numofMatchingCards_mean,\
               data_3_numofMatchingCards_mean,\
               data_4_numofMatchingCards_mean,\
               \
               data_1_ExploitingScore_mean,\
               data_2_ExploitingScore_mean,\
               data_3_ExploitingScore_mean,\
               data_4_ExploitingScore_mean, \
               \
               matching_score_data1_data2_rmse, punishments_score_data1_data2_rmse, \
               matching_score_data1_data3_rmse, punishments_score_data1_data3_rmse, \
               matching_score_data2_data3_rmse, punishments_score_data2_data3_rmse, \
               matching_score_data2_data4_rmse, punishments_score_data2_data4_rmse, \
               \
               matching_score_data1_data2_mae, punishments_score_data1_data2_mae, \
               matching_score_data1_data3_mae, punishments_score_data1_data3_mae, \
               matching_score_data2_data3_mae, punishments_score_data2_data3_mae, \
               matching_score_data2_data4_mae, punishments_score_data2_data4_mae

    def calc_lists_rmse(self, list_1, list_2):
        """
        This function returns RMSE between two lists
        :param list_1: 
        :param list_2: 
        :return: 
        """
        return sqrt(mean_squared_error(list_1, list_2))


    def loadAndPlotStatistics(self,
                              metric_1_path = '/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                              metric_2_path = '/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/'):
        """This functions loads statistics from files and plot them"""
        # Metric 1
        with open(metric_1_path+'standardGame_simulatedData', 'rb') as fp:
            data_1_numofMatchingCards_mean = pickle.load(fp)

        with open(metric_1_path+'second_level_simulatedData', 'rb') as fp:
            data_2_numofMatchingCards_mean = pickle.load(fp)

        with open(metric_1_path+'visualObstacle_simulatedData', 'rb') as fp:
            data_3_numofMatchingCards_mean = pickle.load(fp)

        with open(metric_1_path+'fourth_level_simulatedData', 'rb') as fp:
            data_4_numofMatchingCards_mean = pickle.load(fp)

        # Metric 2
        with open(metric_2_path + 'standardGame_simulatedData', 'rb') as fp:
            data_1_ExploitingScore_mean = pickle.load(fp)

        with open(metric_2_path + 'second_level_simulatedData', 'rb') as fp:
            data_2_ExploitingScore_mean = pickle.load(fp)

        with open(metric_2_path + 'visualObstacle_simulatedData', 'rb') as fp:
            data_3_ExploitingScore_mean = pickle.load(fp)

        with open(metric_2_path + 'fourth_level_simulatedData', 'rb') as fp:
            data_4_ExploitingScore_mean = pickle.load(fp)

        # Do plotting
        self.plot_statistics(data_1_numofMatchingCards_mean, data_2_numofMatchingCards_mean, data_3_numofMatchingCards_mean,
                        data_4_numofMatchingCards_mean, data_1_ExploitingScore_mean, data_2_ExploitingScore_mean,
                        data_3_ExploitingScore_mean, data_4_ExploitingScore_mean, metric_1_path, metric_2_path)

    def loadAndPlotStatistics_twoLabes(self,
                                       label_1_matchingPairs_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics/MatchingPairs_Statistics/standardGame_realData',
                                       label_2_matchingPairs_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics/MatchingPairs_Statistics/visualObstacle_realData',
                                       label_1_punishments_path='share/documents/msalous/DINCO/Data_Collection/memo/Statistics/ExploitingScore_Statistics/standardGame_realData',
                                       label_2_punishments_path='share/documents/msalous/DINCO/Data_Collection/memo/Statistics/ExploitingScore_Statistics/visualObstacle_realData'
                                       ):
        """This functions loads statistics from files and plot them"""
        prefix = '/'
        prefix = '/run/user/1000/gvfs/smb-share:server=file.csl.uni-bremen.de,share='
        # Metric 1
        with open(label_1_matchingPairs_path, 'rb') as fp:
            data_1_numofMatchingCards_mean = pickle.load(fp)

        with open(label_1_matchingPairs_path, 'rb') as fp:
            data_2_numofMatchingCards_mean = pickle.load(fp)

        with open(label_2_matchingPairs_path, 'rb') as fp:
            data_3_numofMatchingCards_mean = pickle.load(fp)

        with open(label_2_matchingPairs_path, 'rb') as fp:
            data_4_numofMatchingCards_mean = pickle.load(fp)

        # Metric 2
        with open(prefix + label_1_punishments_path, 'rb') as fp:
            data_1_ExploitingScore_mean = pickle.load(fp)

        with open(prefix + label_1_punishments_path, 'rb') as fp:
            data_2_ExploitingScore_mean = pickle.load(fp)

        with open(prefix + label_2_punishments_path, 'rb') as fp:
            data_3_ExploitingScore_mean = pickle.load(fp)

        with open(prefix + label_2_punishments_path, 'rb') as fp:
            data_4_ExploitingScore_mean = pickle.load(fp)

        # Do plotting
        self.plot_statistics(data_1_numofMatchingCards_mean, data_2_numofMatchingCards_mean,
                             data_3_numofMatchingCards_mean,
                             data_4_numofMatchingCards_mean, data_1_ExploitingScore_mean,
                             data_2_ExploitingScore_mean,
                             data_3_ExploitingScore_mean, data_4_ExploitingScore_mean)


    def loadAndPlotStatistics_rd_vs_sd(self,
                                       rd_label_1_matchingPairs_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics_with_std/MatchingPairs_Statistics/standardGame_realData',
                                       rd_label_2_matchingPairs_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics_with_std/MatchingPairs_Statistics/visualObstacle_realData',
                                       rd_label_1_matchingStd_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics_with_std/MatchingPairs_Statistics/first_level_realData_std',
                                       rd_label_2_matchingStd_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics_with_std/MatchingPairs_Statistics/third_level_realData_std',
                                       sd_label_1_matchingPairs_path='/share/documents/msalous/DINCO/Data_Collection/memo/SimulationResults/WinStrategy_TrainingData/RealDataStatisticsBased/Statistics/20_Rounds/MatchingPairs_Statistics/standardGame_simulatedData',
                                       sd_label_2_matchingPairs_path='/share/documents/msalous/DINCO/Data_Collection/memo/SimulationResults/WinStrategy_TrainingData/RealDataStatisticsBased/Statistics/20_Rounds/MatchingPairs_Statistics/visualObstacle_simulatedData'
                                       ):
        """This functions loads statistics from files and plot them"""
        # REAL DATA
        with open(rd_label_1_matchingPairs_path, 'rb') as fp:
            rd_label_1_matchingPairs = pickle.load(fp)

        with open(rd_label_2_matchingPairs_path, 'rb') as fp:
            rd_label_2_matchingPairs = pickle.load(fp)

        with open(rd_label_1_matchingStd_path, 'rb') as fp:
            rd_label_1_matchingStd = pickle.load(fp)

        with open(rd_label_2_matchingStd_path, 'rb') as fp:
            rd_label_2_matchingStd = pickle.load(fp)

        # SIMULATED DATA
        with open(sd_label_1_matchingPairs_path, 'rb') as fp:
            sd_label_1_matchingPairs = pickle.load(fp)

        with open(sd_label_2_matchingPairs_path, 'rb') as fp:
            sd_label_2_matchingPairs = pickle.load(fp)

        # Do plotting
        self.plot_statistics_rd_vs_sd(rd_label_1_matchingPairs, rd_label_1_matchingStd,
                                      rd_label_2_matchingPairs, rd_label_2_matchingStd,
                                      sd_label_1_matchingPairs, sd_label_2_matchingPairs)


    def doStatistics(self, data_1, data_2, max_num_rounds=20):
        """This function calculates the statistics as two metrics:
        1. number of matching pairs in rounds
        2. Exploiting punishment"""
        # 10.12.18 FixMe remove this temp check
        # sim_visual_obstacle_means = [0.19210526, 0.35668421, 0.65110526, 0.94910526, 1.31131579,
        #                              1.70689474, 2.15073684, 2.63363158, 3.16568421, 3.71005263,
        #                              4.19510526, 4.60473684, 4.94957895, 5.23726316, 5.47857895,
        #                              5.67884211, 5.84784211, 5.99221053, 6.11521053, 6.21957895]
        # sim_visual_obstacles_stds = [0.39395537, 0.48708321, 0.72707694, 0.90964958, 1.12543716,
        #                              1.32284973, 1.54307092, 1.77263867, 2.03219386, 2.24997871,
        #                              2.36268964, 2.38530119, 2.35863137, 2.29762369, 2.21997252,
        #                              2.13208935, 2.03848643, 1.93962621, 1.84520437, 1.75487166]
        # valid_game_index = 0
        # 10.12.18 FixMe remove this temp check
        numOfRounds = [i for i in range(1, max_num_rounds+1)]
        data_1_numofMatchingCards_allGames = [None] * data_1.shape[0]
        data_1_ExploitingScore_allGames = [None] * data_1.shape[0]
        data_2_numofMatchingCards_allGames = [None] * data_2.shape[0]
        data_2_ExploitingScore_allGames = [None] * data_2.shape[0]
        # Outer loop: games
        for game in range(0, data_1.shape[0]):
            # 10.12.18 FixMe remove this temp check
            # skip_game = False
            # 10.12.18 FixMe remove this temp check
            data_1_numofMatchingCards = [None] * max_num_rounds
            data_1_match = 0
            data_1_ExploitingScore = [None] * max_num_rounds
            data_1_score = 0
            data_1_toExploit = []
            data_2_numofMatchingCards = [None] * max_num_rounds
            data_2_match = 0
            data_2_ExploitingScore = [None] * max_num_rounds
            data_2_score = 0
            data_2_toExploit = []
            # Inner loop: rounds
            for r in numOfRounds:
                # 08.03.19 iloc uses real orderd_index, better than loc which uses pandas_index:
                data_1_card_1 = data_1.loc[game, 'card_' + str((r * 2) - 1)]
                data_1_card_2 = data_1.loc[game, 'card_' + str((r * 2))]
                data_2_card_1 = data_2.loc[game, 'card_' + str((r * 2) - 1)]
                data_2_card_2 = data_2.loc[game, 'card_' + str((r * 2))]

                # 08.03.19 iloc uses real orderd_index, better than loc which uses pandas_index:
                # data_1_card_1 = data_1.iloc[game]['card_' + str((r * 2) - 1)]
                # data_1_card_2 = data_1.iloc[game]['card_' + str((r * 2))]
                # data_2_card_1 = data_2.iloc[game]['card_' + str((r * 2) - 1)]
                # data_2_card_2 = data_2.iloc[game]['card_' + str((r * 2))]

                # matching cards analysis
                # 10.12.18: This can be wrong, because e.g. abs(1.1 - 1.2) in python can be: 0.1, 0.1000x OR 0.099999x
                # if abs(data_1_card_1 - data_1_card_2) < 0.12 and abs(data_1_card_1 - data_1_card_2) != 0:
                #     data_1_match += 1
                # data_1_numofMatchingCards[r - 1] = data_1_match
                # if abs(data_2_card_1 - data_2_card_2) < 0.12 and abs(data_2_card_1 - data_2_card_2) != 0:
                #     data_2_match += 1
                # data_2_numofMatchingCards[r - 1] = data_2_match

                # 10.12.18: Try another approach using simple integers
                if int(data_1_card_1) == int(data_1_card_2) and int(data_1_card_1) != 0:
                    data_1_match += 1
                data_1_numofMatchingCards[r - 1] = data_1_match
                if int(data_2_card_1) == int(data_2_card_2) and int(data_2_card_1) != 0:
                    data_2_match += 1
                data_2_numofMatchingCards[r - 1] = data_2_match

                # exploiting scores analysis
                #######################1. data_1 exploiting checks########################################
                data_1_elapsed_cards = data_1.loc[
                    game, [seen_card for seen_card in self.cards if int(seen_card[5:]) < ((r * 2) - 1)]]
                # handle only non zero cards
                if numpy.isclose(data_1_card_1, 0.0, atol=0.1) == False:
                    # exploit list is not empty
                    if len(data_1_toExploit) != 0:
                        # card_1 follows exploit list
                        if numpy.isclose(data_1_toExploit, data_1_card_1, atol=0.1).any():
                            # card_2 follows exploit list
                            if numpy.isclose(data_1_card_1, data_1_card_2, atol=0.1):
                                if numpy.isclose(data_1_toExploit, data_1_card_1).any():
                                    data_1_toExploit.remove(data_1_card_1)
                                if numpy.isclose(data_1_toExploit, data_1_card_2).any():
                                    data_1_toExploit.remove(data_1_card_2)
                            # card_2 doesn't follow exploit list
                            else:
                                data_1_score -= 1
                                # card_2 has been seen before
                                if numpy.isclose(data_1_elapsed_cards, data_1_card_2, atol=0.1).any():
                                    data_1_toExploit.extend([data_1_card_2])
                        # card_1 doesn't follow exploit list
                        else:
                            data_1_score -= 1
                            # card_1 has been seen before
                            if numpy.isclose(data_1_elapsed_cards, data_1_card_1, atol=0.1).any():
                                data_1_toExploit.extend([data_1_card_1])
                                # No matching => punishment
                                if numpy.isclose(data_1_card_1, data_1_card_2, atol=0.1) == False:
                                    data_1_score -= 1
                                    # card_2 has been seen before
                                    if numpy.isclose(data_1_elapsed_cards, data_1_card_2, atol=0.1).any():
                                        data_1_toExploit.extend([data_1_card_2])
                                # Matching
                                else:
                                    if numpy.isclose(data_1_toExploit, data_1_card_1).any():
                                        data_1_toExploit.remove(data_1_card_1)
                                    if numpy.isclose(data_1_toExploit, data_1_card_2).any():
                                        data_1_toExploit.remove(data_1_card_2)
                            # card_1 has not been seen before
                            else:
                                # Lucky matching, because card_1 has not been seen before
                                if numpy.isclose(data_1_card_1, data_1_card_2, atol=0.1):
                                    if numpy.isclose(data_1_toExploit, data_1_card_1).any():
                                        data_1_toExploit.remove(data_1_card_1)
                                    if numpy.isclose(data_1_toExploit, data_1_card_2).any():
                                        data_1_toExploit.remove(data_1_card_2)
                                # No lucky matching
                                else:
                                    # card_2 has been seen before
                                    if numpy.isclose(data_1_elapsed_cards, data_1_card_2, atol=0.1).any():
                                        data_1_toExploit.extend([data_1_card_2])
                    # exploit list is empty
                    else:
                        # card_1 has been seen before
                        if numpy.isclose(data_1_elapsed_cards, data_1_card_1, atol=0.1).any():
                            data_1_toExploit.extend([data_1_card_1])
                            # Matching: card_2 follows exploit list
                            if numpy.isclose(data_1_card_1, data_1_card_2, atol=0.1):
                                if numpy.isclose(data_1_toExploit, data_1_card_1).any():
                                    data_1_toExploit.remove(data_1_card_1)
                                if numpy.isclose(data_1_toExploit, data_1_card_2).any():
                                    data_1_toExploit.remove(data_1_card_2)
                            # No matching: card_2 doesn't follow exploit list
                            else:
                                data_1_score -= 1
                                # card_2 has been seen before
                                if numpy.isclose(data_1_elapsed_cards, data_1_card_2, atol=0.1).any():
                                    data_1_toExploit.extend([data_1_card_2])
                        # card_1 has not been seen before
                        else:
                            # card_2 has been seen before
                            if numpy.isclose(data_1_elapsed_cards, data_1_card_2, atol=0.1).any():
                                data_1_toExploit.extend([data_1_card_2])


                                ##WRONG!! At the end, double check, remove from exploit test if there is matching:
                                # if numpy.isclose(data_1_card_1, data_1_card_2, atol=0.1) :
                                #    if numpy.isclose(data_1_toExploit, data_1_card_1).any() :
                                #        data_1_toExploit.remove(data_1_card_1)
                                #    if numpy.isclose(data_1_toExploit, data_1_card_2).any() :
                                #        data_1_toExploit.remove(data_1_card_2)
                # Write data_1_ExploitingScore of this round
                data_1_ExploitingScore[r - 1] = data_1_score
                #######################1. data_1 exploiting checks########################################


                #######################2. data_2 exploiting checks########################################
                data_2_elapsed_cards = data_2.loc[game, [
                    seen_card for seen_card in self.cards if int(seen_card[5:]) < ((r * 2) - 1)]]  # handle only non zero cards
                if numpy.isclose(data_2_card_1, 0.0, atol=0.1) == False:  # exploit list is not empty
                    if len(data_2_toExploit) != 0:
                        # card_1 follows exploit list
                        if numpy.isclose(data_2_toExploit, data_2_card_1, atol=0.1).any():  # card_2 follows exploit list
                            if numpy.isclose(data_2_card_1, data_2_card_2, atol=0.1):
                                if numpy.isclose(data_2_toExploit, data_2_card_1).any():
                                    data_2_toExploit.remove(data_2_card_1)
                                if numpy.isclose(data_2_toExploit, data_2_card_2).any():
                                    data_2_toExploit.remove(data_2_card_2)
                            # card_2 doesn't follow exploit list
                            else:
                                data_2_score -= 1
                                # card_2 has been seen before
                                if numpy.isclose(data_2_elapsed_cards, data_2_card_2, atol=0.1).any():
                                    data_2_toExploit.extend([data_2_card_2])
                        # card_1 doesn't follow exploit list
                        else:
                            data_2_score -= 1
                            # card_1 has been seen before
                            if numpy.isclose(data_2_elapsed_cards, data_2_card_1, atol=0.1).any():
                                data_2_toExploit.extend([data_2_card_1])
                                # No matching => punishment
                                if numpy.isclose(data_2_card_1, data_2_card_2, atol=0.1) == False:
                                    data_2_score -= 1
                                    # card_2 has been seen before
                                    if numpy.isclose(data_2_elapsed_cards, data_2_card_2, atol=0.1).any():
                                        data_2_toExploit.extend([data_2_card_2])
                                # Matching
                                else:
                                    if numpy.isclose(data_2_toExploit, data_2_card_1).any():
                                        data_2_toExploit.remove(data_2_card_1)
                                    if numpy.isclose(data_2_toExploit, data_2_card_2).any():
                                        data_2_toExploit.remove(data_2_card_2)
                            # card_1 has not been seen before
                            else:
                                # Lucky matching, because card_1 has not been seen before
                                if numpy.isclose(data_2_card_1, data_2_card_2, atol=0.1):
                                    if numpy.isclose(data_2_toExploit, data_2_card_1).any():
                                        data_2_toExploit.remove(data_2_card_1)
                                    if numpy.isclose(data_2_toExploit, data_2_card_2).any():
                                        data_2_toExploit.remove(data_2_card_2)
                                # No lucky matching
                                else:
                                    # card_2 has been seen before
                                    if numpy.isclose(data_2_elapsed_cards, data_2_card_2, atol=0.1).any():
                                        data_2_toExploit.extend([data_2_card_2])
                    # exploit list is empty
                    else:
                        # card_1 has been seen before
                        if numpy.isclose(data_2_elapsed_cards, data_2_card_1, atol=0.1).any():
                            data_2_toExploit.extend([data_2_card_1])
                            # Matching: card_2 follows exploit list
                            if numpy.isclose(data_2_card_1, data_2_card_2, atol=0.1):
                                if numpy.isclose(data_2_toExploit, data_2_card_1).any():
                                    data_2_toExploit.remove(data_2_card_1)
                                if numpy.isclose(data_2_toExploit, data_2_card_2).any():
                                    data_2_toExploit.remove(data_2_card_2)
                            # No matching: card_2 doesn't follow exploit list
                            else:
                                data_2_score -= 1
                                # card_2 has been seen before
                                if numpy.isclose(data_2_elapsed_cards, data_2_card_2, atol=0.1).any():
                                    data_2_toExploit.extend([data_2_card_2])
                        # card_1 has not been seen before
                        else:
                            # card_2 has been seen before
                            if numpy.isclose(data_2_elapsed_cards, data_2_card_2, atol=0.1).any():
                                data_2_toExploit.extend([data_2_card_2])


                                ##WRONG!! At the end, double check, remove from exploit test if there is matching:
                                # if numpy.isclose(data_2_card_1, data_2_card_2, atol=0.1) :
                                #    if numpy.isclose(data_2_toExploit, data_2_card_1).any() :
                                #        data_2_toExploit.remove(data_2_card_1)
                                #    if numpy.isclose(data_2_toExploit, data_2_card_2).any() :
                                #        data_2_toExploit.remove(data_2_card_2)
                # Write data_2_ExploitingScore of this round
                data_2_ExploitingScore[
                    r - 1] = data_2_score  #######################2. data_2 exploiting checks########################################

            # End of inner loop, add game's matching cards, exploiting scores, etc.

            # original: use game
            data_1_numofMatchingCards_allGames[game] = data_1_numofMatchingCards
            data_1_ExploitingScore_allGames[game] = data_1_ExploitingScore

            data_2_numofMatchingCards_allGames[game] = data_2_numofMatchingCards
            data_2_ExploitingScore_allGames[game] = data_2_ExploitingScore

        # End of outer loop
        # print(data_1_ExploitingScore_allGames)
        # print(data_2_ExploitingScore_allGames)

        print("len(data_1_numofMatchingCards_allGames) = " + str(len(data_1_numofMatchingCards_allGames)))
        print("len(data_2_numofMatchingCards_allGames) = " + str(len(data_2_numofMatchingCards_allGames)))

        print("data_1_numofMatchingCards_mean = " + str(np.mean(data_1_numofMatchingCards_allGames, axis=0)))
        print("data_2_numofMatchingCards_mean = " + str(np.mean(data_2_numofMatchingCards_allGames, axis=0)))

        return (data_1_numofMatchingCards_allGames, data_1_ExploitingScore_allGames, data_2_numofMatchingCards_allGames, data_2_ExploitingScore_allGames)


    def plot_statistics(self,
                        data_1_numofMatchingCards_mean,
                        data_2_numofMatchingCards_mean,
                        data_3_numofMatchingCards_mean,
                        data_4_numofMatchingCards_mean,
                        data_1_ExploitingScore_mean,
                        data_2_ExploitingScore_mean,
                        data_3_ExploitingScore_mean,
                        data_4_ExploitingScore_mean,):
                        # metric_1_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                        # metric_2_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/',
                        # plt_prefix='mg',):
        """This functions plots the statistics"""
        # A. Metric_1: Plot(X=NumOfRounds, Y=NumOfMatchingCards)
        fig = plt.figure()
        fig.suptitle('Two Levels Memory', fontsize=20, y=0.95)
        plt.xlabel('# rounds', fontsize=15)
        plt.ylabel('# matching pairs', fontsize=15)

        # A.1 data_1

        y = data_1_numofMatchingCards_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='o')

        # A.2 data_2

        y = data_2_numofMatchingCards_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='x')

        # A.3 data_3

        y = data_3_numofMatchingCards_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='+')

        # A.4 data_4

        y = data_4_numofMatchingCards_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='v')

        # Show and save
        plt.show()
        # fig.savefig(metric_1_path + plt_prefix + '_fourLabels.png')

        # B. Metric_2: Plot(X=NumOfRounds, Y=Exploiting_Punishment)
        fig = plt.figure()
        fig.suptitle('Two Levels Memory', fontsize=20, y=0.95)
        plt.xlabel('numOfRounds', fontsize=15)
        plt.ylabel('Exploiting_Punishment', fontsize=15)

        # B.1 data_1
        y = data_1_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='o')

        # B.2 data_2
        y = data_2_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='x')

        # B.3 data_3
        y = data_3_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='+')

        # B.4 data_4
        y = data_4_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='v')

        # Show and save
        plt.show()
        # fig.savefig(metric_2_path + plt_prefix + '_fourLabels.png')

    def plot_statistics_rd_vs_sd(self,
                                 rd_data_1_numofMatchingCards_mean,
                                 rd_data_1_numofMatchingCards_std,
                                 rd_data_2_numofMatchingCards_mean,
                                 rd_data_2_numofMatchingCards_std,
                                 sd_data_1_numofMatchingCards_mean,
                                 sd_data_2_numofMatchingCards_mean):
        """This functions plots the statistics"""
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        bold_big_font = {'weight': 'bold', 'size': 20}
        bold_small_font = {'weight': 'bold', 'size': 15}

        # A. Metric_1: Plot(X=NumOfRounds, Y=NumOfMatchingCards)
        fig = plt.figure(figsize=(9, 6))#plt.figure()
        # fig.suptitle('Two Levels Memory', fontsize=20, y=0.95)
        plt.xlabel('# rounds', fontsize=25)
        plt.ylabel('# matching pairs', fontsize=25)

        # A.1 data_1

        y = rd_data_1_numofMatchingCards_mean
        x = self.numOfRounds
        # plt.errorbar(x, y, yerr=rd_data_1_numofMatchingCards_std, ecolor='b', label='No Secondary Task: Real Sessions', color='b')
        plt.plot(x, y, label='Main Task: Real Sessions', color='b', marker='o')

        # A.2 data_2

        y = rd_data_2_numofMatchingCards_mean
        x = self.numOfRounds
        # xx = [i+0.2 for i in x]
        plt.xticks(x)
        # plt.errorbar(xx, y, yerr=rd_data_2_numofMatchingCards_std, ecolor='r', label='With Secondary Task: Real Sessions', color='r')
        plt.plot(x, y, label='Main+Secondary Task: Real Sessions', color='r', marker='o')

        # A.3 data_3

        y = sd_data_1_numofMatchingCards_mean
        x = self.numOfRounds
        plt.plot(x, y, '--', label='Main Task: Sim. Sessions', color='b', marker='o')

        # A.4 data_4

        y = sd_data_2_numofMatchingCards_mean
        x = self.numOfRounds
        plt.plot(x, y, '--', label='Main+Secondary Task: Sim. Sessions', color='r', marker='o')

        # Show and save
        plt.legend(loc="upper left", fontsize=10)
        plt.show()
        # fig.savefig(metric_1_path + plt_prefix + '_fourLabels.png')


    def plot_statistics_with_std(self,
                        data_1_numofMatchingCards_mean,
                        data_2_numofMatchingCards_mean,
                        data_3_numofMatchingCards_mean,
                        data_4_numofMatchingCards_mean,
                        data_1_ExploitingScore_mean,
                        data_2_ExploitingScore_mean,
                        data_3_ExploitingScore_mean,
                        data_4_ExploitingScore_mean,
                        
                        data_1_numofMatchingCards_std,
                        data_2_numofMatchingCards_std,
                        data_3_numofMatchingCards_std,
                        data_4_numofMatchingCards_std,
                        data_1_ExploitingScore_std,
                        data_2_ExploitingScore_std,
                        data_3_ExploitingScore_std,
                        data_4_ExploitingScore_std,
                         ):
                        # metric_1_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                        # metric_2_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/',
                        # plt_prefix='mg',):
        """This functions plots the statistics"""
        # A. Metric_1: Plot(X=NumOfRounds, Y=NumOfMatchingCards)
        # fig = plt.figure()
        fig, ax = plt.subplots(nrows=2, ncols=1)
        fig.suptitle('Two Levels Memory', fontsize=20, y=0.95)

        plt.subplot(2,1,1)
        # A.1 data_1
        y = data_1_numofMatchingCards_mean
        x = self.numOfRounds
        # plt.plot(x, y, marker='o')
        plt.errorbar(x, y, yerr=data_1_numofMatchingCards_std, ecolor='b')

        # A.2 data_2

        y = data_2_numofMatchingCards_mean
        x = self.numOfRounds
        # plt.plot(x, y, marker='x')
        plt.errorbar(x, y, yerr=data_2_numofMatchingCards_std, ecolor='b')

        # A.3 data_3

        y = data_3_numofMatchingCards_mean
        x = self.numOfRounds
        # plt.plot(x, y, marker='+')
        xx = [i+0.2 for i in x]
        plt.errorbar(xx, y, yerr=data_3_numofMatchingCards_std, ecolor='g', )

        # A.4 data_4

        y = data_4_numofMatchingCards_mean
        x = self.numOfRounds
        # plt.plot(x, y, marker='v')
        xx = [i + 0.2 for i in x]
        plt.errorbar(xx, y, yerr=data_4_numofMatchingCards_std, ecolor='g')

        #####################
        # A.2 data_2
        plt.subplot(2, 1, 2)
        plt.xlabel('# rounds', fontsize=15)
        plt.ylabel('# matching pairs', fontsize=15)
        y = data_2_numofMatchingCards_mean
        x = self.numOfRounds
        # plt.plot(x, y, marker='x')
        plt.errorbar(x, y, yerr=data_2_numofMatchingCards_std, ecolor='b')

        # A.3 data_3

        y = data_3_numofMatchingCards_mean
        x = self.numOfRounds
        # plt.plot(x, y, marker='+')
        xx = [i + 0.2 for i in x]
        plt.errorbar(xx, y, yerr=data_3_numofMatchingCards_std, ecolor='g', )
        #####################

        # Show and save
        plt.show()
        # fig.savefig(metric_1_path + plt_prefix + '_fourLabels.png')

        # B. Metric_2: Plot(X=NumOfRounds, Y=Exploiting_Punishment)
        fig = plt.figure()
        fig.suptitle('Two Levels Memory', fontsize=20, y=0.95)
        plt.xlabel('numOfRounds', fontsize=15)
        plt.ylabel('Exploiting_Punishment', fontsize=15)

        # B.1 data_1
        y = data_1_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='o')

        # B.2 data_2
        y = data_2_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='x')

        # B.3 data_3
        y = data_3_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='+')

        # B.4 data_4
        y = data_4_ExploitingScore_mean
        x = self.numOfRounds
        plt.plot(x, y, marker='v')

        # Show and save
        plt.show()
        # fig.savefig(metric_2_path + plt_prefix + '_fourLabels.png')

    def plot_visualObstacle_statistics(self,
                                       data_1_numofMatchingCards_mean,
                                       data_2_numofMatchingCards_mean,
                                       data_3_numofMatchingCards_mean,
                                       data_4_numofMatchingCards_mean,
                                       data_1_ExploitingScore_mean,
                                       data_2_ExploitingScore_mean,
                                       data_3_ExploitingScore_mean,
                                       data_4_ExploitingScore_mean,

                                       data_1_label,
                                       data_2_label,
                                       data_3_label,
                                       data_4_label,

                                       plotted_levels=[True, True, True, True],
                                       max_num_rounds=20,
                                       matching_pairs_fig_path='',
                                       penalties_fig_path=''):
        """This functions plots the statistics"""
        # A. Metric_1: Plot(X=NumOfRounds, Y=NumOfMatchingCards)
        fig = plt.figure()
        # fig.suptitle('Real vs. Simulated Data', fontsize=20, y=0.95)
        plt.xlabel('Number Of Rounds', fontsize=15)
        plt.ylabel('Matching Pairs', fontsize=15)
        plt.xticks(range(1,21))

        rounds = [i for i in range(1, max_num_rounds + 1)]

        # A.1 data_1
        y = data_1_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        # plt.plot(x, y, marker='o')
        if plotted_levels[0]:
            plt.plot(x, y, marker='o', label=data_1_label)

        # A.2 data_2
        y = data_2_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        # plt.plot(x, y, marker='x')
        if plotted_levels[1]:
            plt.plot(x, y, marker='x', label=data_2_label)

        # A.3 data_3
        y = data_3_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        # plt.plot(x, y, marker='+')
        # xx = [i + 0.2 for i in x]
        if plotted_levels[2]:
            plt.plot(x, y, marker='+', label=data_3_label)

        # A.4 data_4
        y = data_4_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        # plt.plot(x, y, marker='v')
        # xx = [i + 0.2 for i in x]
        if plotted_levels[3]:
            plt.plot(x, y, marker='v', label=data_4_label)

        # Show and save
        plt.legend()
        if matching_pairs_fig_path != '':
            fig.savefig(matching_pairs_fig_path)

        plt.show()


        # B. Metric_2: Plot(X=NumOfRounds, Y=Exploiting_Punishment)
        fig = plt.figure()
        # fig.suptitle('Real vs. Simulated Data', fontsize=20, y=0.95)
        plt.xlabel('Number Of Rounds', fontsize=15)
        plt.ylabel('Penalties', fontsize=15)
        plt.xticks(range(1, 21))

        # B.1 data_1
        y = data_1_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        if plotted_levels[0]:
            plt.plot(x, y, marker='o', label=data_1_label)

        # B.2 data_2
        y = data_2_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        if plotted_levels[1]:
            plt.plot(x, y, marker='x', label=data_2_label)

        # B.3 data_3
        y = data_3_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        if plotted_levels[2]:
            plt.plot(x, y, marker='+', label=data_3_label)

        # B.4 data_4
        y = data_4_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        if plotted_levels[3]:
            plt.plot(x, y, marker='v', label=data_4_label)

        # Show and save
        plt.legend()
        if penalties_fig_path != '':
            fig.savefig(penalties_fig_path)

        plt.show()
        
    #############################
    def plot_visualObstacle_statistics_whiskered(self,
                                                 data_1_numofMatchingCards_mean,
                                                 data_2_numofMatchingCards_mean,
                                                 data_3_numofMatchingCards_mean,
                                                 data_4_numofMatchingCards_mean,

                                                 data_1_numofMatchingCards_std,
                                                 data_2_numofMatchingCards_std,
                                                 data_3_numofMatchingCards_std,
                                                 data_4_numofMatchingCards_std,
                                                 
                                                 data_1_ExploitingScore_mean,
                                                 data_2_ExploitingScore_mean,
                                                 data_3_ExploitingScore_mean,
                                                 data_4_ExploitingScore_mean,

                                                 data_1_ExploitingScore_std,
                                                 data_2_ExploitingScore_std,
                                                 data_3_ExploitingScore_std,
                                                 data_4_ExploitingScore_std,
    
                                                 data_1_label,
                                                 data_2_label,
                                                 data_3_label,
                                                 data_4_label,
    
                                                 plotted_levels=[True, True, True, True],
                                                 max_num_rounds=20,
                                                 matching_pairs_fig_path='',
                                                 penalties_fig_path=''):
        """This functions plots the statistics"""
        # A. Metric_1: Plot(X=NumOfRounds, Y=NumOfMatchingCards)
        fig, ax = plt.subplots()
        # fig.suptitle('Real vs. Simulated Data', fontsize=20, y=0.95)
        plt.xlabel('Number Of Rounds', fontsize=15)
        plt.ylabel('Matching Pairs', fontsize=15)
        plt.xticks(range(1, 21))

        rounds = [i for i in range(1, max_num_rounds + 1)]

        # A.1 data_1
        y = data_1_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        err = data_1_numofMatchingCards_std
        # plt.plot(x, y, marker='o')
        if plotted_levels[0]:
            # plt.plot(x, y, marker='o', label=data_1_label)
            ax.errorbar(x, y, marker='o', label=data_1_label, yerr=err)

        # A.2 data_2
        y = data_2_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        xx = [i + 0.1 for i in x]
        err = data_2_numofMatchingCards_std
        # plt.plot(x, y, marker='x')
        if plotted_levels[1]:
            # plt.plot(x, y, marker='x', label=data_2_label)
            ax.errorbar(xx, y, marker='x', label=data_2_label, yerr=err)

        # A.3 data_3
        y = data_3_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        xx = [i + 0.2 for i in x]
        err = data_3_numofMatchingCards_std
        # plt.plot(x, y, marker='+')
        # xx = [i + 0.2 for i in x]
        if plotted_levels[2]:
            # plt.plot(x, y, marker='+', label=data_3_label)
            ax.errorbar(xx, y, marker='+', label=data_3_label, yerr=err)

        # A.4 data_4
        y = data_4_numofMatchingCards_mean
        x = rounds  # self.numOfRounds
        xx = [i - 0.1 for i in x]
        err = data_4_numofMatchingCards_std
        # plt.plot(x, y, marker='v')
        # xx = [i + 0.2 for i in x]
        if plotted_levels[3]:
            # plt.plot(x, y, marker='v', label=data_4_label)
            ax.errorbar(xx, y, marker='v', label=data_4_label, yerr=err)

        # Show and save
        plt.legend()
        if matching_pairs_fig_path != '':
            fig.savefig(matching_pairs_fig_path[:-4] + '_whiskers.png')

        plt.show()

        # B. Metric_2: Plot(X=NumOfRounds, Y=Exploiting_Punishment)
        fig, ax = plt.subplots()
        # fig.suptitle('Real vs. Simulated Data', fontsize=20, y=0.95)
        plt.xlabel('Number Of Rounds', fontsize=15)
        plt.ylabel('Penalties', fontsize=15)
        plt.xticks(range(1, 21))

        # B.1 data_1
        y = data_1_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        err = data_1_ExploitingScore_std
        if plotted_levels[0]:
            # plt.plot(x, y, marker='o', label=data_1_label)
            ax.errorbar(x, y, marker='o', label=data_1_label, yerr=err)

        # B.2 data_2
        y = data_2_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        xx = [i + 0.1 for i in x]
        err = data_2_ExploitingScore_std
        if plotted_levels[1]:
            # plt.plot(x, y, marker='x', label=data_2_label)
            ax.errorbar(xx, y, marker='x', label=data_2_label, yerr=err)

        # B.3 data_3
        y = data_3_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        xx = [i + 0.2 for i in x]
        err = data_3_ExploitingScore_std
        if plotted_levels[2]:
            # plt.plot(x, y, marker='+', label=data_3_label)
            ax.errorbar(xx, y, marker='+', label=data_3_label, yerr=err)

        # B.4 data_4
        y = data_4_ExploitingScore_mean
        x = rounds  # self.numOfRounds
        xx = [i - 0.1 for i in x]
        err = data_4_ExploitingScore_std
        if plotted_levels[3]:
            # plt.plot(x, y, marker='v', label=data_4_label)
            ax.errorbar(xx, y, marker='v', label=data_4_label, yerr=err)

        # Show and save
        plt.legend()
        if penalties_fig_path != '':
            fig.savefig(penalties_fig_path[:-4] + '_whiskers.png')

        plt.show()
    #############################


    def sort_and_relabel_memory_levels_data(self,
                                            fourLabels_realData_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Logs/',
                                            sorted_files_parentPath='/share/documents/msalous/DINCO/Data_Collection/memo/All_Memo_Separated_Levels/'):
        """This function reads memory games logs and sorts them into four separated files
         according to its labels: from first level till fourth level.
         In addition, it's very important to notice that sorting means also re-labeling levels from 0 to 3 just like 
         the simulated data, and this important step ensures a valid usage of the resulting data when testing a model
         trained with simulated data whose labels are consecutive from 0 as first level till 3 as fourth level. """
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        col_names = cards + ['card_t' + str(i) for i in
                             range(1, self.sequence_length + 1)]  # card_1...card_40, card_t1...card_t40
        col_names = col_names + ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']

        memo_all_logs = os.listdir(fourLabels_realData_basePath)

        fourLabels_realData = pd.concat(
            pd.read_csv(fourLabels_realData_basePath + mu_log, header=None, names=col_names)
            for mu_log in memo_all_logs)

        print('Real data: ' + str(fourLabels_realData.shape))

        # Separate the real data into four levels files and re-label accordingly:
        firstLevel_realData_memory = fourLabels_realData.loc[fourLabels_realData['label'] == 2]#.reset_index()
        firstLevel_realData_memory.loc[:, 'label'] = 0
        secondLevel_realData_memory = fourLabels_realData.loc[fourLabels_realData['label'] == 3]#.reset_index()
        secondLevel_realData_memory.loc[:, 'label'] = 1
        thirdLevel_realData_memory = fourLabels_realData.loc[fourLabels_realData['label'] == 1]#.reset_index()
        thirdLevel_realData_memory.loc[:, 'label'] = 2
        fourthLevel_realData_memory = fourLabels_realData.loc[fourLabels_realData['label'] == 0]#.reset_index()
        fourthLevel_realData_memory.loc[:, 'label'] = 3

        print('firstLevel_realData_memory: ' + str(firstLevel_realData_memory.shape))
        print('secondLevel_realData_memory: ' + str(secondLevel_realData_memory.shape))
        print('thirdLevel_realData_memory: ' + str(thirdLevel_realData_memory.shape))
        print('fourthLevel_realData_memory: ' + str(fourthLevel_realData_memory.shape))

        # write the sorted files:
        firstLevel_realData_memory.to_csv(sorted_files_parentPath + 'firstLevel_realData_memory.log', encoding='utf-8', header=None, index=False)
        secondLevel_realData_memory.to_csv(sorted_files_parentPath + 'secondLevel_realData_memory.log', encoding='utf-8', header=None, index=False)
        thirdLevel_realData_memory.to_csv(sorted_files_parentPath + 'thirdLevel_realData_memory.log', encoding='utf-8', header=None, index=False)
        fourthLevel_realData_memory.to_csv(sorted_files_parentPath + 'fourthLevel_realData_memory.log', encoding='utf-8', header=None, index=False)

    # 12.09.18
    def sort_all_memory_data_with_visualObstacle(self,
                                                 participants_logs_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/VisualObstacle_OriginalLogs/all_logs_with_similaritymatrix/',
                                                 sorted_files_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/VisualObstacle/AnaysisAndTests/',
                                                 ):
        """This function reads memory games logs with visual obstacles and sorts them into seven separated files
         according to its labels: 
         label = 0 standard_game
         label = 1 memory obstacle
         label = 2 visual obstacle,
         label = 3 memory obstacle with memory assistant
         label = 4 visual obstacle with visual assistant
         label = 5 memory obstacle with visual assistant
         label = 6 visual obstacle with memory assistant
         
         In addition, it's very important to notice that sorting means also re-labeling levels from 0 to 6 just like 
         the simulated data (we simulate now first three labels), and this important step ensures a valid usage of 
         the resulting data when testing a model trained with simulated data whose labels are consecutive 
         from 0 as first level till X (now 2) as last level. """

        # column names
        similarities_assignments = ['Assign_' + str(i) for i in range(1, 22)]
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        cards_ts = ['card_t' + str(i) for i in range(1, self.sequence_length + 1)]  # card_t1...card_t40
        statistics_and_label = ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']
        assistant_indicators = ['memoryAssistant', 'visualAssistant']
        col_names = similarities_assignments + cards + cards_ts + statistics_and_label + assistant_indicators
        required_cols = similarities_assignments + cards + cards_ts + statistics_and_label

        memo_all_logs = os.listdir(participants_logs_basePath)

        sevenLabels_realData = pd.concat(pd.read_csv(participants_logs_basePath + log, header=None, names=col_names)
                                         for log in memo_all_logs)

        print('sevenLabels_realData: ' + str(sevenLabels_realData.shape))

        # Separate the real data into seven levels files and re-label accordingly:
        standardGame_realData_memory = sevenLabels_realData.loc[sevenLabels_realData['label'] == 100, required_cols]
        standardGame_realData_memory.loc[:, 'label'] = 0

        
        memoryObstacle_realData_memory = sevenLabels_realData.loc[
            (sevenLabels_realData['label'] == 101) &
            (sevenLabels_realData['memoryAssistant'] == 0) &
            (sevenLabels_realData['visualAssistant'] == 0), required_cols]
        memoryObstacle_realData_memory.loc[:, 'label'] = 1

        visualObstacle_realData_memory = sevenLabels_realData.loc[
            (sevenLabels_realData['label'] == 102) &
            (sevenLabels_realData['memoryAssistant'] == 0) &
            (sevenLabels_realData['visualAssistant'] == 0), required_cols]
        visualObstacle_realData_memory.loc[:, 'label'] = 2

        memoryObstacle_memoryAssistant_realData_memory = sevenLabels_realData.loc[
            (sevenLabels_realData['label'] == 101) &
            (sevenLabels_realData['memoryAssistant'] == 1) &
            (sevenLabels_realData['visualAssistant'] == 0), required_cols]
        memoryObstacle_memoryAssistant_realData_memory.loc[:, 'label'] = 3

        visualObstacle_visualAssistant_realData_visual = sevenLabels_realData.loc[
            (sevenLabels_realData['label'] == 102) &
            (sevenLabels_realData['memoryAssistant'] == 0) &
            (sevenLabels_realData['visualAssistant'] == 1), required_cols]
        visualObstacle_visualAssistant_realData_visual.loc[:, 'label'] = 4

        memoryObstacle_visualAssistant_realData_memory = sevenLabels_realData.loc[
            (sevenLabels_realData['label'] == 101) &
            (sevenLabels_realData['memoryAssistant'] == 0) &
            (sevenLabels_realData['visualAssistant'] == 1), required_cols]
        memoryObstacle_visualAssistant_realData_memory.loc[:, 'label'] = 5

        visualObstacle_memoryAssistant_realData_visual = sevenLabels_realData.loc[
            (sevenLabels_realData['label'] == 102) &
            (sevenLabels_realData['memoryAssistant'] == 1) &
            (sevenLabels_realData['visualAssistant'] == 0), required_cols]
        visualObstacle_memoryAssistant_realData_visual.loc[:, 'label'] = 6

        print('standardGame_realData_memory: ' + str(standardGame_realData_memory.shape))
        print('memoryObstacle_realData_memory: ' + str(memoryObstacle_realData_memory.shape))
        print('visualObstacle_realData_memory: ' + str(visualObstacle_realData_memory.shape))
        print('memoryObstacle_memoryAssistant_realData_memory: ' + str(memoryObstacle_memoryAssistant_realData_memory.shape))
        print('visualObstacle_visualAssistant_realData_visual: ' + str(visualObstacle_visualAssistant_realData_visual.shape))
        print('memoryObstacle_visualAssistant_realData_memory: ' + str(memoryObstacle_visualAssistant_realData_memory.shape))
        print('visualObstacle_memoryAssistant_realData_visual: ' + str(visualObstacle_memoryAssistant_realData_visual.shape))

        # write
        standardGame_realData_memory.to_csv(sorted_files_basePath + 'standardGame_realData_memory.log',
                                                         encoding='utf-8', header=None, index=False)

        memoryObstacle_realData_memory.to_csv(sorted_files_basePath + 'memoryObstacle_realData_memory.log',
                                            encoding='utf-8', header=None, index=False)

        visualObstacle_realData_memory.to_csv(sorted_files_basePath + 'visualObstacle_realData_memory.log',
                                            encoding='utf-8', header=None, index=False)

        memoryObstacle_memoryAssistant_realData_memory.to_csv(sorted_files_basePath + 'memoryObstacle_memoryAssistant_realData_memory.log',
                                              encoding='utf-8', header=None, index=False)

        visualObstacle_visualAssistant_realData_visual.to_csv(sorted_files_basePath + 'visualObstacle_visualAssistant_realData_visual.log',
                                              encoding='utf-8', header=None, index=False)

        memoryObstacle_visualAssistant_realData_memory.to_csv(sorted_files_basePath + 'memoryObstacle_visualAssistant_realData_memory.log',
                                              encoding='utf-8', header=None, index=False)

        visualObstacle_memoryAssistant_realData_visual.to_csv(sorted_files_basePath + 'visualObstacle_memoryAssistant_realData_visual.log',
                                              encoding='utf-8', header=None, index=False)

    # 04.10.2018
    def leave_one_subject_out_cv(self,
                                 real_data_base_path="C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\raw\\real\\",
                                 simulated_data_base_path="C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\classification_data\\raw\\simulated\\",
                                 num_of_subjects=20,
                                 num_subject_game_trials=1,

                                 simulated_level1='simulated_noObst_sorted.log',
                                 simulated_level2='simulated_glare_effect_sorted.log',
                                 simulated_level3='',
                                 simulated_level4='',
                                 simulated_level5='',
                                 simulated_level6='',
                                 simulated_level7='',
                                 simulated_level8='',
                                 simulated_level9='',

                                 real_level1='valid_noObst_logs.txt',
                                 real_level2='valid_glare_effect_logs.txt',
                                 real_level3='',
                                 real_level4='',
                                 real_level5='',
                                 real_level6='',
                                 real_level7='',
                                 real_level8='',
                                 real_level9='',

                                 simulated_real=[True, True],
                                 rmse_col=True
                                 ):
        """
        This function reads real and simulated data and splits them according to leave_one_subject_out  
        :param real_data_base_path: 
        :param simulated_data_base_path: 
        :param num_subject_game_trials: this parameter indicates how many real trials we have from each subject_game
        :return: 
        """

        # column names
        similarities_assignments_noObst = ['Assign_' + str(i) for i in range(1, 22)]
        similarities_assignments_glare = ['Assign_' + str(i) for i in range(1, 29)]
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        cards_ts = ['card_t' + str(i) for i in range(1, self.sequence_length + 1)]  # card_t1...card_t40
        statistics_and_label = ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']

        real_cols_noObst = similarities_assignments_noObst + cards + cards_ts + statistics_and_label
        real_cols_glare = similarities_assignments_glare + cards + cards_ts + statistics_and_label
        simulated_cols = cards + statistics_and_label
        if rmse_col:
            simulated_cols = ['rmse'] + simulated_cols

        # read

        # simulated data
        if simulated_real[0]:
            if simulated_level1 != '':
                firstLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level1,
                                                              header=None, names=simulated_cols)
            if simulated_level2 != '':
                secondLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level2,
                                                               header=None, names=simulated_cols)
            
        # real data
        if simulated_real[1]:
            if real_level1 != '':
                firstLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level1,
                                                        header=None, names=real_cols_noObst)

            if real_level2 != '':
                secondLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level2,
                                                         header=None, names=real_cols_glare)

            
        # splits
        for split in range(1, num_of_subjects+1):
            # create output folders
            if not os.path.exists(simulated_data_base_path + 'Split' + str(split)) and simulated_real[0]:
                os.mkdir(simulated_data_base_path + 'Split' + str(split))

            if not os.path.exists(real_data_base_path + 'Split' + str(split) + '/for_simulation/separated_levels') \
                    and simulated_real[1]:
                os.mkdir(real_data_base_path + 'Split' + str(split))
                os.mkdir(real_data_base_path + 'Split' + str(split) + '/for_simulation')
                os.mkdir(real_data_base_path + 'Split' + str(split) + '/for_simulation/separated_levels')

            if not os.path.exists(real_data_base_path + 'Split' + str(split) + '/for_testing/separated_levels') \
                    and simulated_real[1]:
                os.mkdir(real_data_base_path + 'Split' + str(split) + '/for_testing')
                os.mkdir(real_data_base_path + 'Split' + str(split) + '/for_testing/separated_levels')

            # indices
            test_basic_index = split-1

            # 20.03.20 generalize the index(indices) based on num_trials
            # real_train_indices = [True] * num_of_subjects
            # real_train_indices[test_basic_index] = False

            # 20.03.20 generalize the index(indices) based on num_trials
            real_train_indices = [True] * num_of_subjects * num_subject_game_trials
            for trial in range(num_subject_game_trials):
                real_train_indices[(test_basic_index * num_subject_game_trials) + trial] = False

            simulated_train_indices = [True] * num_of_subjects * 1000
            simulated_train_indices[(test_basic_index*1000):(test_basic_index*1000)+1000] = [False for _ in range(0, 1000)]

            # write tested_split:
            # 1. simulated data: training
            if simulated_real[0]:
                if simulated_level1 != '':
                    firstLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(simulated_data_base_path + 'Split' + str(split) + '/FirstLevelMemoryTrainingData.log', encoding='utf-8', header=None, index=False)

                if simulated_level2 != '':
                    secondLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(simulated_data_base_path + 'Split' + str(split) + '/SecondLevelMemoryTrainingData.log', encoding='utf-8', header=None, index=False)

            # 2. real data: training
            if simulated_real[1]:
                if real_level1 != '':
                    firstLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(split) + '/for_simulation/separated_levels/firstLevel_realData_memory.log', encoding='utf-8', header=None, index=False)

                if real_level2 != '':
                    secondLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(split) + '/for_simulation/separated_levels/secondLevel_realData_memory.log', encoding='utf-8', header=None, index=False)

            # 3. real data: testing
            # 20.03.20 generalize the index(indices) based on num_trials
            # FixMe use to_frame().T in case of num_subject_game_trials=1, because the .iloc will return a Serie: e.g
            # Fixme .iloc[....].to_frame().T.to_csv(...
            if simulated_real[1]:
                if real_level1 != '':
                    firstLevel_real_data.iloc[test_basic_index * num_subject_game_trials:
                    (test_basic_index * num_subject_game_trials) + num_subject_game_trials][:].to_csv(real_data_base_path + 'Split' + str(
                        split) + '/for_testing/separated_levels/firstLevel_realData_memory.log', encoding='utf-8',
                                                                                       header=None, index=False)

                if real_level2 != '':
                    secondLevel_real_data.iloc[test_basic_index * num_subject_game_trials:
                    (test_basic_index * num_subject_game_trials) + num_subject_game_trials][:].to_csv(real_data_base_path + 'Split' + str(
                        split) + '/for_testing/separated_levels/secondLevel_realData_memory.log', encoding='utf-8',
                                                                                        header=None, index=False)


    # 04.10.2018

    #######################################
    # 21.05.2020
    def kfold_cv(self,
                 real_data_base_path="/share/documents/msalous/DINCO/Data_Collection/memo/VisualObstacle/Memo_Multiple_Splits/RealData_CLEAN/",
                 simulated_data_base_path="/share/documents/msalous/DINCO/Data_Collection/memo/SimulationResults/WinStrategy_TrainingData/VisualObstacle/RealDataStatisticsBased_Individual_MultipleSplits_RevealCoding_CLEAN/20_Rounds/",

                 num_of_subjects=49,
                 sim_episodes_per_subject=1000,
                 k=7,

                 simulated_level1='FirstLevelMemoryTrainingData.log',
                 simulated_level2='SecondLevelMemoryTrainingData.log',
                 simulated_level3='ThirdLevelMemoryTrainingData.log',
                 simulated_level4='',
                 simulated_level5='',
                 simulated_level6='',
                 simulated_level7='',
                 simulated_level8='',
                 simulated_level9='',

                 real_level1='0_standard.txt',
                 real_level2='1_memoryObstacle.txt',
                 real_level3='2_visualObstacle.txt',
                 real_level4='',
                 real_level5='',
                 real_level6='',
                 real_level7='',
                 real_level8='',
                 real_level9='',

                 simulated_real=[True, True],
                 rmse_col=False
                                 ):
        """
        This function reads real and simulated data and splits them according to leave_one_subject_out
        :param real_data_base_path:
        :param simulated_data_base_path:
        :param num_subject_game_trials: this parameter indicates how many real trials we have from each subject_game
        :return:
        """

        # column names
        similarities_assignments = ['Assign_' + str(i) for i in range(1, 22)]
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        cards_ts = ['card_t' + str(i) for i in range(1, self.sequence_length + 1)]  # card_t1...card_t40
        statistics_and_label = ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']

        real_cols = similarities_assignments + cards + cards_ts + statistics_and_label
        simulated_cols = cards + statistics_and_label
        if rmse_col:
            simulated_cols = ['rmse'] + simulated_cols

        # read

        # simulated data
        if simulated_real[0]:
            if simulated_level1 != '':
                firstLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level1,
                                                        header=None, names=simulated_cols)
            if simulated_level2 != '':
                secondLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level2,
                                                         header=None, names=simulated_cols)
            if simulated_level3 != '':
                thirdLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level3,
                                                        header=None, names=simulated_cols)

            if simulated_level4 != '':
                fourthLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level4,
                                                         header=None, names=simulated_cols)

            if simulated_level5 != '':
                fifthLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level5,
                                                        header=None, names=simulated_cols)

            if simulated_level6 != '':
                sixthLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level6,
                                                        header=None, names=simulated_cols)

            if simulated_level7 != '':
                seventhLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level7,
                                                          header=None, names=simulated_cols)

            if simulated_level8 != '':
                eighthLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level8,
                                                         header=None, names=simulated_cols)

            if simulated_level9 != '':
                ninthLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/" + simulated_level9,
                                                        header=None, names=simulated_cols)

        # real data
        if simulated_real[1]:
            if real_level1 != '':
                firstLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level1,
                                                   header=None, names=real_cols)

            if real_level2 != '':
                secondLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level2,
                                                    header=None, names=real_cols)

            if real_level3 != '':
                thirdLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level3,
                                                   header=None, names=real_cols)

            if real_level4 != '':
                fourthLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level4,
                                                    header=None, names=real_cols)
            if real_level5 != '':
                fifthLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level5,
                                                   header=None, names=real_cols)

            if real_level6 != '':
                sixthLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level6,
                                                   header=None, names=real_cols)

            if real_level7 != '':
                seventhLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level7,
                                                     header=None, names=real_cols)

            if real_level8 != '':
                eighthLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level8,
                                                    header=None, names=real_cols)

            if real_level9 != '':
                ninthLevel_real_data = pd.read_csv(real_data_base_path + "all/" + real_level9,
                                                   header=None, names=real_cols)

        # splits
        num_train = num_of_subjects - int(num_of_subjects / k)  # 42 from 49, given k=7
        val_and_test = num_of_subjects - num_train # 7 from 49, given k=7
        num_test = int(val_and_test / 2) # 3 from 7
        num_val = int(val_and_test - num_test) # 4 from 7

        for split in range(1, num_of_subjects + 1, val_and_test):
            # split_number
            split_number = int(split/7) + 1  # consecutive split numbers: 1,2,3,4,5,6,7
            # create output folders
            # simulated data: only train data
            if not os.path.exists(simulated_data_base_path + 'Split' + str(split_number)) and simulated_real[0]:
                os.mkdir(simulated_data_base_path + 'Split' + str(split_number))

            # real data: train, test and validation data
            if not os.path.exists(real_data_base_path + 'Split' + str(split_number) + '/for_simulation/separated_levels') \
                    and simulated_real[1]:
                os.mkdir(real_data_base_path + 'Split' + str(split_number))
                os.mkdir(real_data_base_path + 'Split' + str(split_number) + '/for_simulation')
                os.mkdir(real_data_base_path + 'Split' + str(split_number) + '/for_simulation/separated_levels')

            if not os.path.exists(real_data_base_path + 'Split' + str(split_number) + '/for_testing/separated_levels') \
                    and simulated_real[1]:
                os.mkdir(real_data_base_path + 'Split' + str(split_number) + '/for_testing')
                os.mkdir(real_data_base_path + 'Split' + str(split_number) + '/for_testing/separated_levels')

            if not os.path.exists(real_data_base_path + 'Split' + str(split_number) + '/for_validation/separated_levels') \
                    and simulated_real[1]:
                os.mkdir(real_data_base_path + 'Split' + str(split_number) + '/for_validation')
                os.mkdir(real_data_base_path + 'Split' + str(split_number) + '/for_validation/separated_levels')

            # indices
            print('num_test: ' + str(num_test))
            print('num_val: ' + str(num_val))
            real_test_indices = list(range(split - 1, split - 1 + num_test))  # for split=1 [0,1,2]
            real_val_indices = list(range(split - 1 + num_test, split - 1 + num_test + num_val))  # e.g. [3,4,5,6]
            real_eval_indices = real_test_indices + real_val_indices  # for split=1 [0,1,2,3,4,5,6]

            real_train_indices = [True] * num_of_subjects
            # important to have all real_eval_indices False, i.e. including real_eval_indices[-1]
            real_train_indices[real_eval_indices[0]: real_eval_indices[-1] + 1] = \
                [False for _ in range(0, len(real_eval_indices))]

            simulated_train_indices = [True] * num_of_subjects * sim_episodes_per_subject
            simulated_train_indices[(real_eval_indices[0] * sim_episodes_per_subject):
                                    (real_eval_indices[-1] * sim_episodes_per_subject) + sim_episodes_per_subject] = \
                [False for _ in range(0, sim_episodes_per_subject * len(real_eval_indices))]

            print('real_test_indices: ' + str(real_test_indices))
            print('real_val_indices: ' + str(real_val_indices))
            print('real_eval_indices: ' + str(real_eval_indices))
            print('############################')
            # write tested_split:
            # 1. simulated data: training
            if simulated_real[0]:
                if simulated_level1 != '':
                    firstLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/FirstLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level2 != '':
                    secondLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/SecondLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level3 != '':
                    thirdLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/ThirdLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level4 != '':
                    fourthLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/FourthLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level5 != '':
                    fifthLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/FifthLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level6 != '':
                    sixthLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/SixthLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level7 != '':
                    seventhLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/SeventhLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level8 != '':
                    eighthLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/EighthLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

                if simulated_level9 != '':
                    ninthLevel_simulated_data.iloc[simulated_train_indices][:].to_csv(
                        simulated_data_base_path + 'Split' + str(split_number) + '/NinthLevelMemoryTrainingData.log',
                        encoding='utf-8', header=None, index=False)

            # 2. real data: training
            if simulated_real[1]:
                if real_level1 != '':
                    firstLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/firstLevel_realData_memory.log',
                                                                            encoding='utf-8', header=None,
                                                                            index=False)

                if real_level2 != '':
                    secondLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/secondLevel_realData_memory.log',
                                                                             encoding='utf-8', header=None,
                                                                             index=False)

                if real_level3 != '':
                    thirdLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/thirdLevel_realData_memory.log',
                                                                            encoding='utf-8', header=None,
                                                                            index=False)

                if real_level4 != '':
                    fourthLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/fourthLevel_realData_memory.log',
                                                                             encoding='utf-8',
                                                                             header=None, index=False)

                if real_level5 != '':
                    fifthLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/fifthLevel_realData_memory.log',
                                                                            encoding='utf-8',
                                                                            header=None, index=False)

                if real_level6 != '':
                    sixthLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/sixthLevel_realData_memory.log',
                                                                            encoding='utf-8',
                                                                            header=None, index=False)

                if real_level7 != '':
                    seventhLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/seventhLevel_realData_memory.log',
                                                                              encoding='utf-8',
                                                                              header=None, index=False)

                if real_level8 != '':
                    eighthLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/eighthLevel_realData_memory.log',
                                                                             encoding='utf-8',
                                                                             header=None, index=False)

                if real_level9 != '':
                    ninthLevel_real_data.iloc[real_train_indices][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_simulation/separated_levels/ninthLevel_realData_memory.log',
                                                                            encoding='utf-8',
                                                                            header=None, index=False)

            # 3. real data: testing
            # 20.03.20 generalize the index(indices) based on num_trials
            # FixMe use to_frame().T in case of num_subject_game_trials=1, because the .iloc will return a Serie: e.g
            # Fixme .iloc[....].to_frame().T.to_csv(...
            if simulated_real[1]:
                if real_level1 != '':
                    firstLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/firstLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level2 != '':
                    secondLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/secondLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level3 != '':
                    thirdLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/thirdLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level4 != '':
                    fourthLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/fourthLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level5 != '':
                    fifthLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/fifthLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level6 != '':
                    sixthLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/sixthLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level7 != '':
                    seventhLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/seventhLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level8 != '':
                    eighthLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/eighthLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

                if real_level9 != '':
                    ninthLevel_real_data.iloc[real_test_indices[0]:real_test_indices[-1] + 1][:].to_csv(real_data_base_path + 'Split' + str(
                        split_number) + '/for_testing/separated_levels/ninthLevel_realData_memory.log', encoding='utf-8',
                              header=None, index=False)

            # 4. real data: validation
            if simulated_real[1]:
                if real_level1 != '':
                    firstLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/firstLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level2 != '':
                    secondLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/secondLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level3 != '':
                    thirdLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/thirdLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level4 != '':
                    fourthLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/fourthLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level5 != '':
                    fifthLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/fifthLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level6 != '':
                    sixthLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/sixthLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level7 != '':
                    seventhLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/seventhLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level8 != '':
                    eighthLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/eighthLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)

                if real_level9 != '':
                    ninthLevel_real_data.iloc[real_val_indices[0]:real_val_indices[-1] + 1][:].to_csv(
                        real_data_base_path + 'Split' + str(
                            split_number) + '/for_validation/separated_levels/ninthLevel_realData_memory.log',
                        encoding='utf-8',
                        header=None, index=False)
            

    # 21.05.2020
    #######################################

    # 02.11.2018
    def leave_one_episode_out_cv(self,
                                 real_data_base_path="/share/documents/msalous/DINCO/Data_Collection/memo/VisualObstacle/Memo_Multiple_Splits/RealData_CLEAN/",
                                 simulated_data_base_path="/share/documents/msalous/DINCO/Data_Collection/memo/SimulationResults/WinStrategy_TrainingData/VisualObstacle/RealDataStatisticsBased_Individual_MultipleSplits_RevealCoding_CLEAN/20_Rounds/",
                                 standard_game_indices=[13,15,9,18],
                                 memory_obstacle_indices=[17,8,18,5],
                                 num_of_all_subjects=19):
        """
        This function reads real data and their corresponding simulated data and splits them according to 
        leave_one_episode_out where all possible combinations between episodes in different labels are used as splits  
        :param real_data_base_path: 
        :param simulated_data_base_path: 
        :return: 
        """

        # column names
        similarities_assignments = ['Assign_' + str(i) for i in range(1, 22)]
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        cards_ts = ['card_t' + str(i) for i in range(1, self.sequence_length + 1)]  # card_t1...card_t40
        statistics_and_label = ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']

        real_cols = similarities_assignments + cards + cards_ts + statistics_and_label
        simulated_cols = cards + statistics_and_label

        # read
        # real data
        firstLevel_real_data = pd.read_csv(real_data_base_path + "all/standardGame_realData_memory.log",
                                           header=None, names=real_cols)
        secondLevel_real_data = pd.read_csv(real_data_base_path + "all/memoryObstacle_realData_memory.log",
                                            header=None, names=real_cols)

        firstLevel_real_data_indices = [False] * num_of_all_subjects
        for i in standard_game_indices:
            firstLevel_real_data_indices[i] = True
        firstLevel_real_data = firstLevel_real_data.loc[firstLevel_real_data_indices, :]

        secondLevel_real_data_indices = [False] * num_of_all_subjects
        for i in memory_obstacle_indices:
            secondLevel_real_data_indices[i] = True
        secondLevel_real_data = secondLevel_real_data.loc[secondLevel_real_data_indices, :]

        # simulated data
        firstLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/FirstLevelMemoryTrainingData.log",
                                                header=None, names=simulated_cols)
        secondLevel_simulated_data = pd.read_csv(simulated_data_base_path + "all/SecondLevelMemoryTrainingData.log",
                                                 header=None, names=simulated_cols)

        firstLevel_simulated_data_indices = [False] * num_of_all_subjects * 1000
        for basic_index in standard_game_indices:
            firstLevel_simulated_data_indices[(basic_index * 1000):(basic_index * 1000) + 1000] = [True for _ in
                                                                                                   range(0, 1000)]
        firstLevel_simulated_data = firstLevel_simulated_data.loc[firstLevel_simulated_data_indices, :]

        secondLevel_simulated_data_indices = [False] * num_of_all_subjects * 1000
        for basic_index in memory_obstacle_indices:
            secondLevel_simulated_data_indices[(basic_index * 1000):(basic_index * 1000) + 1000] = [True for _ in
                                                                                                   range(0, 1000)]
        secondLevel_simulated_data = secondLevel_simulated_data.loc[secondLevel_simulated_data_indices, :]

        print("Real Data:")
        print("firstLevel_real_data = " + str(firstLevel_real_data.shape))
        print("secondLevel_real_data = " + str(secondLevel_real_data.shape))
        print("Simulated Data:")
        print("firstLevel_simulated_data = " + str(firstLevel_simulated_data.shape))
        print("secondLevel_simulated_data = " + str(secondLevel_simulated_data.shape))

        # write all data
        # simulated data
        firstLevel_simulated_data.to_csv(
            simulated_data_base_path + 'bestRMSEs/all/FirstLevelMemoryTrainingData.log',
            encoding='utf-8', header=None, index=False)
        secondLevel_simulated_data.to_csv(
            simulated_data_base_path + 'bestRMSEs/all/SecondLevelMemoryTrainingData.log',
            encoding='utf-8', header=None, index=False)
        # real data
        firstLevel_real_data.to_csv(
            real_data_base_path + 'bestRMSEs/all/standardGame_realData_memory.log',
            encoding='utf-8', header=None, index=False)
        secondLevel_real_data.to_csv(
            real_data_base_path + 'bestRMSEs/all/memoryObstacle_realData_memory.log',
            encoding='utf-8', header=None, index=False)
        exit()

        # write splits
        split = 1
        for standard_game_test_index in standard_game_indices:
            for memory_obstacle_test_index in memory_obstacle_indices:
                # real data
                real_standard_game_train_indices = [index for index in standard_game_indices
                                                    if index != standard_game_test_index]
                real_memory_obstacle_train_indices = [index for index in memory_obstacle_indices
                                                      if index != memory_obstacle_test_index]

                # simulated data
                simulated_standard_game_train_indices = [list(range(index*1000, (index*1000) + 1000))
                                                         for index in standard_game_indices
                                                         if index != standard_game_test_index]
                # flatten
                simulated_standard_game_train_indices = [item for sublist in simulated_standard_game_train_indices
                                                         for item in sublist]
                simulated_memory_obstacle_train_indices = [list(range(index*1000, (index*1000) + 1000))
                                                           for index in memory_obstacle_indices
                                                           if index != memory_obstacle_test_index]
                # flatten
                simulated_memory_obstacle_train_indices = [item for sublist in simulated_memory_obstacle_train_indices
                                                           for item in sublist]

                # write tested_split:
                # 1. simulated data: training
                firstLevel_simulated_data.loc[simulated_standard_game_train_indices, :].to_csv(
                    simulated_data_base_path + 'bestRMSEs/Split' + str(split) + '/FirstLevelMemoryTrainingData.log',
                    encoding='utf-8', header=None, index=False)
                secondLevel_simulated_data.loc[simulated_memory_obstacle_train_indices, :].to_csv(
                    simulated_data_base_path + 'bestRMSEs/Split' + str(split) + '/SecondLevelMemoryTrainingData.log',
                    encoding='utf-8', header=None, index=False)

                # 2. real data: training
                firstLevel_real_data.loc[real_standard_game_train_indices, :].to_csv(
                    real_data_base_path + 'bestRMSEs/Split' + str(split) +
                    '/for_simulation/separated_levels/firstLevel_realData_memory.log',
                    encoding='utf-8', header=None, index=False)
                secondLevel_real_data.loc[real_memory_obstacle_train_indices, :].to_csv(
                    real_data_base_path + 'bestRMSEs/Split' + str(split) +
                    '/for_simulation/separated_levels/secondLevel_realData_memory.log',
                    encoding='utf-8', header=None, index=False)

                # 3. real data: testing
                firstLevel_real_data.loc[standard_game_test_index, :].to_frame().T.to_csv(
                    real_data_base_path + 'bestRMSEs/Split' + str(split) +
                    '/for_testing/separated_levels/firstLevel_realData_memory.log',
                    encoding='utf-8', header=None, index=False)
                secondLevel_real_data.loc[memory_obstacle_test_index, :].to_frame().T.to_csv(
                    real_data_base_path + 'bestRMSEs/Split' + str(split) +
                    '/for_testing/separated_levels/secondLevel_realData_memory.log',
                    encoding='utf-8', header=None, index=False)
                # prepare next tested_split
                split += 1

    # 02.11.2018

    def sort_and_split_memory_data_with_visualObstacle(self,
                                                       participants_logs_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/VisualObstacle_OriginalLogs/all_logs_with_similaritymatrix/',
                                                       sorted_files_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/VisualObstacle/Memo_Multiple_Splits/RealData/',
                                                       num_test_logs=6,
                                                       num_splits=20):
        """This function reads memory games logs with visual obstacles and sorts them into three separated files
         according to its labels: first level standard_game, second level memory obstacle and third is visual obstacle.
         For each created level, it splits the logs into num_splits splits.
         In addition, it's very important to notice that sorting means also re-labeling levels from 0 to 2 just like 
         the simulated data, and this important step ensures a valid usage of the resulting data when testing a model
         trained with simulated data whose labels are consecutive from 0 as first level till 2 as third level. """

        # column names
        similarities_assignments = ['Assign_' + str(i) for i in range(1, 22)]
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        cards_ts = ['card_t' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40, card_t1...card_t40
        statistics_and_label = ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']
        assistant_cols = ['Assistant_1', 'Assistant_2']
        col_names = similarities_assignments + cards + cards_ts + statistics_and_label + assistant_cols

        memo_all_logs = os.listdir(participants_logs_basePath)

        threeLabels_realData = pd.concat(pd.read_csv(participants_logs_basePath + log, header=None, names=col_names)
                                         for log in memo_all_logs)

        threeLabels_realData = threeLabels_realData.loc[(threeLabels_realData['Assistant_1'] == 0) &
                                                        (threeLabels_realData['Assistant_2'] == 0)]

        threeLabels_realData = threeLabels_realData.loc[:, similarities_assignments + cards + cards_ts + statistics_and_label]

        print('threeLabels_realData: ' + str(threeLabels_realData.shape))

        # Separate the real data into three levels files and re-label accordingly:
        firstLevel_realData_memory = threeLabels_realData.loc[threeLabels_realData['label'] == 100]#.reset_index()
        firstLevel_realData_memory.loc[:, 'label'] = 0
        secondLevel_realData_memory = threeLabels_realData.loc[threeLabels_realData['label'] == 101]#.reset_index()
        secondLevel_realData_memory.loc[:, 'label'] = 1
        thirdLevel_realData_memory = threeLabels_realData.loc[threeLabels_realData['label'] == 102]#.reset_index()
        thirdLevel_realData_memory.loc[:, 'label'] = 2

        print('firstLevel_realData_memory: ' + str(firstLevel_realData_memory.shape))
        print('secondLevel_realData_memory: ' + str(secondLevel_realData_memory.shape))
        print('thirdLevel_realData_memory: ' + str(thirdLevel_realData_memory.shape))

        # Fixme: 03.10 This way of shuffling can make it difficult for a model to learn the behaviour!
        # Fixme: Because shuffling can lead to data_person1_label1 in training data while data_person1_label2
        # Fixme: not in training data!! I.e. no consistent behaviour to be learnt!
        # for simulation and for testing splits
        for split in range(1, num_splits+1):
            # paths
            for_simulation_path = sorted_files_basePath + 'Split' + str(split) + '/for_simulation/separated_levels/'
            for_testing_path = sorted_files_basePath + 'Split' + str(split) + '/for_testing/separated_levels/'
            # shuffle and tested_split data
            firstLevel_realData_memory_split = firstLevel_realData_memory.sample(frac=1)  # shuffle
            firstLevel_realData_memory_for_simulation = firstLevel_realData_memory_split[:len(firstLevel_realData_memory_split) - num_test_logs]  # sor simulation tested_split
            firstLevel_realData_memory_for_testing = firstLevel_realData_memory_split[len(firstLevel_realData_memory_split) - num_test_logs:]  # sor testing tested_split

            secondLevel_realData_memory_split = secondLevel_realData_memory.sample(frac=1)  # shuffle
            secondLevel_realData_memory_for_simulation = secondLevel_realData_memory_split[:len(secondLevel_realData_memory_split) - num_test_logs]  # sor simulation tested_split
            secondLevel_realData_memory_for_testing = secondLevel_realData_memory_split[len(secondLevel_realData_memory_split) - num_test_logs:]  # sor testing tested_split

            thirdLevel_realData_memory_split = thirdLevel_realData_memory.sample(frac=1)  # shuffle
            thirdLevel_realData_memory_for_simulation = thirdLevel_realData_memory_split[:len(thirdLevel_realData_memory_split) - num_test_logs]  # sor simulation tested_split
            thirdLevel_realData_memory_for_testing = thirdLevel_realData_memory_split[len(thirdLevel_realData_memory_split) - num_test_logs:]  # sor testing tested_split
            # write data
            firstLevel_realData_memory_for_simulation.to_csv(for_simulation_path + 'firstLevel_realData_memory.log', encoding='utf-8', header=None, index=False)
            firstLevel_realData_memory_for_testing.to_csv(for_testing_path + 'firstLevel_realData_memory.log', encoding='utf-8', header=None, index=False)

            secondLevel_realData_memory_for_simulation.to_csv(for_simulation_path + 'secondLevel_realData_memory.log',encoding='utf-8', header=None, index=False)
            secondLevel_realData_memory_for_testing.to_csv(for_testing_path + 'secondLevel_realData_memory.log',encoding='utf-8', header=None, index=False)

            thirdLevel_realData_memory_for_simulation.to_csv(for_simulation_path + 'thirdLevel_realData_memory.log', encoding='utf-8', header=None, index=False)
            thirdLevel_realData_memory_for_testing.to_csv(for_testing_path + 'thirdLevel_realData_memory.log', encoding='utf-8', header=None, index=False)

    def sort_muBased_memory_levels_data(self,
                                        basePath='/share/documents/msalous/DINCO/Data_Collection/memo/MU_Based_Multiple_Splits/Split1/for_simulation/',
                                        sorted_files_parentPath='/share/documents/msalous/DINCO/Data_Collection/memo/MU_Based_Multiple_Splits/Split1/for_simulation/separated_levels/'):
        """This function reads muBased memory games logs and sorts them into two separated files according to their 
        labels: mu1=0 and mu2=1.
         """
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        col_names = cards + ['card_t' + str(i) for i in
                             range(1, self.sequence_length + 1)]  # card_1...card_40, card_t1...card_t40
        col_names = col_names + ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']

        memo_all_logs = os.listdir(basePath)

        muBased_realData = pd.concat(
            pd.read_csv(basePath + mu_log, header=None, names=col_names)
            for mu_log in memo_all_logs)

        print('Real data: ' + str(muBased_realData.shape))

        # Separate the real data into two files:
        mu1_realData_memory = muBased_realData.loc[muBased_realData['label'] == 0]
        # Ensure integer label
        mu1_realData_memory.loc[:, 'label'] = 0
        mu2_realData_memory = muBased_realData.loc[muBased_realData['label'] == 1]
        # Ensure integer label
        mu2_realData_memory.loc[:, 'label'] = 1

        print('mu1_realData_memory: ' + str(mu1_realData_memory.shape))
        print('mu2_realData_memory: ' + str(mu2_realData_memory.shape))


        # write the sorted files:
        mu1_realData_memory.to_csv(sorted_files_parentPath + 'mu1_realData_memory.log', encoding='utf-8', header=None, index=False)
        mu2_realData_memory.to_csv(sorted_files_parentPath + 'mu2_realData_memory.log', encoding='utf-8', header=None, index=False)


    def crossing_t_test(self,
                        simulated_data_matching_metric_path ='/share/documents/msalous/DINCO/SimulationResults/WinStrategy_TrainingData/RealDataStatisticsBased/20_Rounds/FourLabels_Statistics/MatchingPairs_Statistics/',
                        simulated_data_punishments_metric_path='/share/documents/msalous/DINCO/SimulationResults/WinStrategy_TrainingData/RealDataStatisticsBased/20_Rounds/FourLabels_Statistics/ExploitingScore_Statistics/',
                        real_data_matching_metric_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                        real_data_punishments_metric_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/', ):

        """This functions loads  simulated and real-data based statistics from files and calc t test accordingly"""

        # Simulated data: matching metric
        with open(simulated_data_matching_metric_path + 'standardGame_simulatedData', 'rb') as fp:
            simulated_data_matching_metric_first_level = pickle.load(fp)

        with open(simulated_data_matching_metric_path + 'second_level_simulatedData', 'rb') as fp:
            simulated_data_matching_metric_second_level = pickle.load(fp)

        with open(simulated_data_matching_metric_path + 'visualObstacle_simulatedData', 'rb') as fp:
            simulated_data_matching_metric_third_level = pickle.load(fp)

        with open(simulated_data_matching_metric_path + 'fourth_level_simulatedData', 'rb') as fp:
            simulated_data_matching_metric_fourth_level = pickle.load(fp)

        # Simulated data: punishments metric
        with open(simulated_data_punishments_metric_path + 'standardGame_simulatedData', 'rb') as fp:
            simulated_data_punishments_metric_first_level = pickle.load(fp)

        with open(simulated_data_punishments_metric_path + 'second_level_simulatedData', 'rb') as fp:
            simulated_data_punishments_metric_second_level = pickle.load(fp)

        with open(simulated_data_punishments_metric_path + 'visualObstacle_simulatedData', 'rb') as fp:
            simulated_data_punishments_metric_third_level = pickle.load(fp)

        with open(simulated_data_punishments_metric_path + 'fourth_level_simulatedData', 'rb') as fp:
            simulated_data_punishments_metric_fourth_level = pickle.load(fp)

        # Real data: matching metric
        with open(real_data_matching_metric_path + 'standardGame_realData', 'rb') as fp:
            real_data_matching_metric_first_level = pickle.load(fp)

        with open(real_data_matching_metric_path + 'memoryObstacle_realData', 'rb') as fp:
            real_data_matching_metric_second_level = pickle.load(fp)

        with open(real_data_matching_metric_path + 'visualObstacle_realData', 'rb') as fp:
            real_data_matching_metric_third_level = pickle.load(fp)

        with open(real_data_matching_metric_path + 'fourth_level_realData', 'rb') as fp:
            real_data_matching_metric_fourth_level = pickle.load(fp)

        # Real data: punishments metric
        with open(real_data_punishments_metric_path + 'standardGame_realData', 'rb') as fp:
            real_data_punishments_metric_first_level = pickle.load(fp)

        with open(real_data_punishments_metric_path + 'memoryObstacle_realData', 'rb') as fp:
            real_data_punishments_metric_second_level = pickle.load(fp)

        with open(real_data_punishments_metric_path + 'visualObstacle_realData', 'rb') as fp:
            real_data_punishments_metric_third_level = pickle.load(fp)

        with open(real_data_punishments_metric_path + 'fourth_level_realData', 'rb') as fp:
            real_data_punishments_metric_fourth_level = pickle.load(fp)

        # Calc t tests
        # 1. Matching Statistic: Sim vs. Real
        matching_first_level_sim_vs_real_t, matching_first_level_sim_vs_real_p = ttest_ind(
            simulated_data_matching_metric_first_level,
            real_data_matching_metric_first_level)

        matching_second_level_sim_vs_real_t, matching_second_level_sim_vs_real_p = ttest_ind(
            simulated_data_matching_metric_second_level,
            real_data_matching_metric_second_level)

        matching_third_level_sim_vs_real_t, matching_third_level_sim_vs_real_p = ttest_ind(
            simulated_data_matching_metric_third_level,
            real_data_matching_metric_third_level)

        matching_fourth_level_sim_vs_real_t, matching_fourth_level_sim_vs_real_p = ttest_ind(
            simulated_data_matching_metric_fourth_level,
            real_data_matching_metric_fourth_level)

        # 2. Punishments Statistic: Sim vs. Real
        punishments_first_level_sim_vs_real_t, punishments_first_level_sim_vs_real_p = ttest_ind(
            simulated_data_punishments_metric_first_level,
            real_data_punishments_metric_first_level)

        punishments_second_level_sim_vs_real_t, punishments_second_level_sim_vs_real_p = ttest_ind(
            simulated_data_punishments_metric_second_level,
            real_data_punishments_metric_second_level)

        punishments_third_level_sim_vs_real_t, punishments_third_level_sim_vs_real_p = ttest_ind(
            simulated_data_punishments_metric_third_level,
            real_data_punishments_metric_third_level)

        punishments_fourth_level_sim_vs_real_t, punishments_fourth_level_sim_vs_real_p = ttest_ind(
            simulated_data_punishments_metric_fourth_level,
            real_data_punishments_metric_fourth_level)

        return matching_first_level_sim_vs_real_t, matching_first_level_sim_vs_real_p, \
               matching_second_level_sim_vs_real_t, matching_second_level_sim_vs_real_p,\
               matching_third_level_sim_vs_real_t, matching_third_level_sim_vs_real_p,\
               matching_fourth_level_sim_vs_real_t, matching_fourth_level_sim_vs_real_p,\
               punishments_first_level_sim_vs_real_t, punishments_first_level_sim_vs_real_p,\
               punishments_second_level_sim_vs_real_t, punishments_second_level_sim_vs_real_p, \
               punishments_third_level_sim_vs_real_t, punishments_third_level_sim_vs_real_p,\
               punishments_fourth_level_sim_vs_real_t, punishments_fourth_level_sim_vs_real_p


    def significant_difference_t_test(self,
                                      data_type='realData',
                                      matching_metric_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                                      punishments_metric_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/', ):

        """This functions loads real-data based statistics from files and calculates t test accordingly"""

        # Real data: matching metric
        with open(matching_metric_path + 'first_level_'+data_type, 'rb') as fp:
            real_data_matching_metric_first_level = pickle.load(fp)

        with open(matching_metric_path + 'second_level_'+data_type, 'rb') as fp:
            real_data_matching_metric_second_level = pickle.load(fp)

        with open(matching_metric_path + 'third_level_'+data_type, 'rb') as fp:
            real_data_matching_metric_third_level = pickle.load(fp)

        with open(matching_metric_path + 'fourth_level_'+data_type, 'rb') as fp:
            real_data_matching_metric_fourth_level = pickle.load(fp)

        # Real data: punishments metric
        with open(punishments_metric_path + 'first_level_'+data_type, 'rb') as fp:
            real_data_punishments_metric_first_level = pickle.load(fp)

        with open(punishments_metric_path + 'second_level_'+data_type, 'rb') as fp:
            real_data_punishments_metric_second_level = pickle.load(fp)

        with open(punishments_metric_path + 'third_level_'+data_type, 'rb') as fp:
            real_data_punishments_metric_third_level = pickle.load(fp)

        with open(punishments_metric_path + 'fourth_level_'+data_type, 'rb') as fp:
            real_data_punishments_metric_fourth_level = pickle.load(fp)

        # Calc t tests
        # 1. Matching Statistic:
        matching_first_level_vs_second_level_t, matching_first_level_vs_second_level_p = ttest_ind(
            real_data_matching_metric_first_level,
            real_data_matching_metric_second_level)

        matching_first_level_vs_third_level_t, matching_first_level_vs_third_level_p = ttest_ind(
            real_data_matching_metric_first_level,
            real_data_matching_metric_third_level)

        matching_first_level_vs_fourth_level_t, matching_first_level_vs_fourth_level_p = ttest_ind(
            real_data_matching_metric_first_level,
            real_data_matching_metric_fourth_level)

        matching_second_level_vs_third_level_t, matching_second_level_vs_third_level_p = ttest_ind(
            real_data_matching_metric_second_level,
            real_data_matching_metric_third_level)

        matching_second_level_vs_fourth_level_t, matching_second_level_vs_fourth_level_p = ttest_ind(
            real_data_matching_metric_second_level,
            real_data_matching_metric_fourth_level)

        matching_third_level_vs_fourth_level_t, matching_third_level_vs_fourth_level_p = ttest_ind(
            real_data_matching_metric_third_level,
            real_data_matching_metric_fourth_level)

        # 2. Punishments Statistic:
        punishments_first_level_vs_second_level_t, punishments_first_level_vs_second_level_p = ttest_ind(
            real_data_punishments_metric_first_level,
            real_data_punishments_metric_second_level)

        punishments_first_level_vs_third_level_t, punishments_first_level_vs_third_level_p = ttest_ind(
            real_data_punishments_metric_first_level,
            real_data_punishments_metric_third_level)

        punishments_first_level_vs_fourth_level_t, punishments_first_level_vs_fourth_level_p = ttest_ind(
            real_data_punishments_metric_first_level,
            real_data_punishments_metric_fourth_level)

        punishments_second_level_vs_third_level_t, punishments_second_level_vs_third_level_p = ttest_ind(
            real_data_punishments_metric_second_level,
            real_data_punishments_metric_third_level)

        punishments_second_level_vs_fourth_level_t, punishments_second_level_vs_fourth_level_p = ttest_ind(
            real_data_punishments_metric_second_level,
            real_data_punishments_metric_fourth_level)

        punishments_third_level_vs_fourth_level_t, punishments_third_level_vs_fourth_level_p = ttest_ind(
            real_data_punishments_metric_third_level,
            real_data_punishments_metric_fourth_level)

        return matching_first_level_vs_second_level_t, matching_first_level_vs_second_level_p, \
               matching_first_level_vs_third_level_t, matching_first_level_vs_third_level_p,\
               matching_first_level_vs_fourth_level_t, matching_first_level_vs_fourth_level_p,\
               matching_second_level_vs_third_level_t, matching_second_level_vs_third_level_p, \
               matching_second_level_vs_fourth_level_t, matching_second_level_vs_fourth_level_p, \
               matching_third_level_vs_fourth_level_t, matching_third_level_vs_fourth_level_p, \
               punishments_first_level_vs_second_level_t, punishments_first_level_vs_second_level_p, \
               punishments_first_level_vs_third_level_t, punishments_first_level_vs_third_level_p, \
               punishments_first_level_vs_fourth_level_t, punishments_first_level_vs_fourth_level_p, \
               punishments_second_level_vs_third_level_t, punishments_second_level_vs_third_level_p, \
               punishments_second_level_vs_fourth_level_t, punishments_second_level_vs_fourth_level_p, \
               punishments_third_level_vs_fourth_level_t, punishments_third_level_vs_fourth_level_p


    def compare_data(self,
                             real_data_matching_path ='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/MatchingPairs_Statistics/',
                             simulated_data_matching_path ='/share/documents/msalous/DINCO/SimulationResults/WinStrategy_TrainingData/RealDataStatisticsBased/20_Rounds/FourLabels_Statistics/MatchingPairs_Statistics/',
                             real_data_punishments_path='/share/documents/msalous/DINCO/Memo_App_RealData/FourLabels/Statistics/ExploitingScore_Statistics/',
                             simulated_data_punishments_path='/share/documents/msalous/DINCO/SimulationResults/WinStrategy_TrainingData/RealDataStatisticsBased/20_Rounds/FourLabels_Statistics/ExploitingScore_Statistics/'):

        """This functions loads statistics from files and compares them using RMSE Matrix"""

        # Real data matching statistic path
        with open(real_data_matching_path + 'standardGame_realData', 'rb') as fp:
            first_level_matching_realData = pickle.load(fp)

        with open(real_data_matching_path + 'memoryObstacle_realData', 'rb') as fp:
            second_level_matching_realData = pickle.load(fp)

        with open(real_data_matching_path + 'visualObstacle_realData', 'rb') as fp:
            third_level_matching_realData = pickle.load(fp)

        with open(real_data_matching_path + 'fourth_level_realData', 'rb') as fp:
            fourth_level_matching_realData = pickle.load(fp)

        realData_matching_levels_list = [first_level_matching_realData.tolist(), second_level_matching_realData.tolist(),
                                         third_level_matching_realData.tolist(), fourth_level_matching_realData.tolist()]

        # simulated data matching statistic path
        with open(simulated_data_matching_path + 'standardGame_simulatedData', 'rb') as fp:
            first_level_matching_simulatedData = pickle.load(fp)

        with open(simulated_data_matching_path + 'second_level_simulatedData', 'rb') as fp:
            second_level_matching_simulatedData = pickle.load(fp)

        with open(simulated_data_matching_path + 'visualObstacle_simulatedData', 'rb') as fp:
            third_level_matching_simulatedData = pickle.load(fp)

        with open(simulated_data_matching_path + 'fourth_level_simulatedData', 'rb') as fp:
            fourth_level_matching_simulatedData = pickle.load(fp)

        simulated_data_matching_levels_list = [first_level_matching_simulatedData.tolist(), second_level_matching_simulatedData.tolist(),
                                               third_level_matching_simulatedData.tolist(), fourth_level_matching_simulatedData.tolist()]

        matching_rmse_matrix = self.calc_rmse_matrix(realData_levels_list=realData_matching_levels_list,
                                                     simulatedData_levels_list=simulated_data_matching_levels_list)

        # Real data punishments statistic path
        with open(real_data_punishments_path + 'standardGame_realData', 'rb') as fp:
            first_level_punishments_realData = pickle.load(fp)

        with open(real_data_punishments_path + 'memoryObstacle_realData', 'rb') as fp:
            second_level_punishments_realData = pickle.load(fp)

        with open(real_data_punishments_path + 'visualObstacle_realData', 'rb') as fp:
            third_level_punishments_realData = pickle.load(fp)

        with open(real_data_punishments_path + 'fourth_level_realData', 'rb') as fp:
            fourth_level_punishments_realData = pickle.load(fp)

        realData_punishments_levels_list = [first_level_punishments_realData.tolist(), second_level_punishments_realData.tolist(),
                                         third_level_punishments_realData.tolist(), fourth_level_punishments_realData.tolist()]

        # simulated data punishments statistic path
        with open(simulated_data_punishments_path + 'standardGame_simulatedData', 'rb') as fp:
            first_level_punishments_simulatedData = pickle.load(fp)

        with open(simulated_data_punishments_path + 'second_level_simulatedData', 'rb') as fp:
            second_level_punishments_simulatedData = pickle.load(fp)

        with open(simulated_data_punishments_path + 'visualObstacle_simulatedData', 'rb') as fp:
            third_level_punishments_simulatedData = pickle.load(fp)

        with open(simulated_data_punishments_path + 'fourth_level_simulatedData', 'rb') as fp:
            fourth_level_punishments_simulatedData = pickle.load(fp)

        simulated_data_punishments_levels_list = [first_level_punishments_simulatedData.tolist(), second_level_punishments_simulatedData.tolist(),
                                               third_level_punishments_simulatedData.tolist(), fourth_level_punishments_simulatedData.tolist()]

        punishments_rmse_matrix = self.calc_rmse_matrix(realData_levels_list=realData_punishments_levels_list,
                                                        simulatedData_levels_list=simulated_data_punishments_levels_list)

        return matching_rmse_matrix, punishments_rmse_matrix
    
    
    def calc_rmse_matrix(self,
                         realData_levels_list,
                         simulatedData_levels_list):
        """This function calculates an RMSE matrix between real and simulated memory levels data"""

        rmse_matrix = np.zeros((4, 4))
        for realData_level, simulatedData_level in list(product(realData_levels_list, simulatedData_levels_list)):
            real_data_index = realData_levels_list.index(realData_level)
            simulated_data_index = simulatedData_levels_list.index(simulatedData_level)
            rmse_matrix[real_data_index, simulated_data_index] = sqrt(mean_squared_error(realData_level,
                                                                                         simulatedData_level))
        return rmse_matrix


    def analyze_memo_data(self,
                          memo_data_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Logs/',
                          metric_1_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics/MatchingPairs_Statistics/',
                          metric_2_path='/share/documents/msalous/DINCO/Data_Collection/memo/Statistics/ExploitingScore_Statistics/',
                          is_real_data = True):

        """
        This function analyzes memo data by: 
         1. separating them into the 4 different emulated-memory-levels: MG, MG_Noise, MG_CS, and MG_CS_ST
         2. applying and plotting the matching-paris and punishments statistics
         
        :param memo_data_basePath: 
        :param metric_1_path: 
        :param metric_2_path: 
        :return: 
        """
        mg_cs_st_data, mg_cs_data, mg_data, mg_noise_data = self.read_memo_data(memo_data_basePath,
                                                                                is_real_data=is_real_data)

        plt_prefix = 'simulatedData'
        if is_real_data:
            plt_prefix = 'realData'

        self.analyzeData(data_1=mg_data,
                         data_1_title='mg_data',
                         data_2=mg_noise_data,
                         data_2_title='mg_noise_data',
                         data_3=mg_cs_data,
                         data_3_title='mg_cs_data',
                         data_4=mg_cs_st_data,
                         data_4_title='mg_cs_st_data',
                         metric_1_path=metric_1_path,
                         metric_2_path=metric_2_path,
                         plt_prefix=plt_prefix)


    def muBased_realMemo_for_simulation_testing_split(self,
                                                      mu1_memo_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/MU_Based_Level1/',
                                                      mu2_memo_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/MU_Based_Level2/',
                                                      for_simulation=0.5,
                                                      metric_1_path='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Splits/Statistics/MatchingPairs_Statistics/MuBased_Memo/',
                                                      metric_2_path='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Splits/Statistics/ExploitingScore_Statistics/MuBased_Memo/',
                                                      data_splits_path='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Splits/MuBased_Memo/'):

        """
        This function separates the given memo real data into simulation basis and testing data.
        The given memo real data are mu_based data. They are organized in two levels as mu1 and mu2.
        Both the simulation basis and testing data will be extracted equally from mu1 amd mu2 levels.
        It plots then the matching-pairs and punishments statistics on both the simulation basis and testing data to 
        show how similar they are.
        
        :param mu1_memo_basePath: 
        :param for_simulation: 
        :return: 
        """

        # Preparation
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40
        col_names = cards + ['card_t' + str(i) for i in
                             range(1, self.sequence_length + 1)]  # card_1...card_40, card_t1...card_t40
        col_names = col_names + ['totalTime', 'actualSum', 'userSum', 'shortStoryScore', 'label']
        
        # First, read mu based data
        mg_cs_st_realData_mu1, mg_cs_realData_mu1, mg_realData_mu1, mg_noise_realData_mu1 = \
            self.read_memo_data(mu1_memo_basePath)
        mg_cs_st_realData_mu2, mg_cs_realData_mu2, mg_realData_mu2, mg_noise_realData_mu2 = \
            self.read_memo_data(mu2_memo_basePath)

        # Then, separate them into simulation basis and testing data organized in the four emulated memory levels
        # 1. MG_CS_ST Level:
        mg_cs_st_realData_for_simulation_mu1, mg_cs_st_realData_for_testing_mu1 = train_test_split(mg_cs_st_realData_mu1,
                                                                                                   test_size=1-for_simulation)
        #mg_cs_st_realData_for_simulation_mu1 = mg_cs_st_realData_for_simulation_mu1.reset_index()
        #mg_cs_st_realData_for_testing_mu1 = mg_cs_st_realData_for_testing_mu1.reset_index()

        mg_cs_st_realData_for_simulation_mu2, mg_cs_st_realData_for_testing_mu2 = train_test_split(mg_cs_st_realData_mu2,
                                                                                                   test_size=1 - for_simulation)
        #mg_cs_st_realData_for_simulation_mu2 = mg_cs_st_realData_for_simulation_mu2.reset_index()
        #mg_cs_st_realData_for_testing_mu2 = mg_cs_st_realData_for_testing_mu2.reset_index()

        mg_cs_st_realData_for_simulation = pd.concat((mu_based for mu_based in [mg_cs_st_realData_for_simulation_mu1,
                                                                               mg_cs_st_realData_for_simulation_mu2]),
                                                     names=col_names, ignore_index=True)
        mg_cs_st_realData_for_testing = pd.concat((mu_based for mu_based in [mg_cs_st_realData_for_testing_mu1,
                                                                            mg_cs_st_realData_for_testing_mu2]),
                                                  names=col_names, ignore_index=True)

        # 2. MG_CS Level:
        mg_cs_realData_for_simulation_mu1, mg_cs_realData_for_testing_mu1 = train_test_split(mg_cs_realData_mu1,
                                                                                             test_size=1-for_simulation)
        #mg_cs_realData_for_simulation_mu1 = mg_cs_realData_for_simulation_mu1.reset_index()
        #mg_cs_realData_for_testing_mu1 = mg_cs_realData_for_testing_mu1.reset_index()

        mg_cs_realData_for_simulation_mu2, mg_cs_realData_for_testing_mu2 = train_test_split(mg_cs_realData_mu2,
                                                                                             test_size=1-for_simulation)
        #mg_cs_realData_for_simulation_mu2 = mg_cs_realData_for_simulation_mu2.reset_index()
        #mg_cs_realData_for_testing_mu2 = mg_cs_realData_for_testing_mu2.reset_index()

        mg_cs_realData_for_simulation = pd.concat((mu_based for mu_based in [mg_cs_realData_for_simulation_mu1,
                                                                            mg_cs_realData_for_simulation_mu2]),
                                                  names=col_names, ignore_index=True)
        mg_cs_realData_for_testing = pd.concat((mu_based for mu_based in [mg_cs_realData_for_testing_mu1,
                                                                         mg_cs_realData_for_testing_mu2]),
                                               names=col_names, ignore_index=True)

        # 3. MG Level:
        mg_realData_for_simulation_mu1, mg_realData_for_testing_mu1 = train_test_split(mg_realData_mu1,
                                                                                       test_size=1-for_simulation)
        #mg_realData_for_simulation_mu1 = mg_realData_for_simulation_mu1.reset_index()
        #mg_realData_for_testing_mu1 = mg_realData_for_testing_mu1.reset_index()

        mg_realData_for_simulation_mu2, mg_realData_for_testing_mu2 = train_test_split(mg_realData_mu2,
                                                                                       test_size=1 - for_simulation)
        #mg_realData_for_simulation_mu2 = mg_realData_for_simulation_mu2.reset_index()
        #mg_realData_for_testing_mu2 = mg_realData_for_testing_mu2.reset_index()

        mg_realData_for_simulation = pd.concat((mu_based for mu_based in [mg_realData_for_simulation_mu1,
                                                                         mg_realData_for_simulation_mu2]),
                                               names=col_names, ignore_index=True)
        mg_realData_for_testing = pd.concat((mu_based for mu_based in [mg_realData_for_testing_mu1,
                                                                      mg_realData_for_testing_mu2]),
                                            names=col_names, ignore_index=True)

        # 4. MG_Noise Level:
        mg_noise_realData_for_simulation_mu1, mg_noise_realData_for_testing_mu1 = train_test_split(mg_noise_realData_mu1,
                                                                                                   test_size=1-for_simulation)
        #mg_noise_realData_for_simulation_mu1 = mg_noise_realData_for_simulation_mu1.reset_index()
        #mg_noise_realData_for_testing_mu1 = mg_noise_realData_for_testing_mu1.reset_index()

        mg_noise_realData_for_simulation_mu2, mg_noise_realData_for_testing_mu2 = train_test_split(mg_noise_realData_mu2,
                                                                                                   test_size=1-for_simulation)
        #mg_noise_realData_for_simulation_mu2 = mg_noise_realData_for_simulation_mu2.reset_index()
        #mg_noise_realData_for_testing_mu2 = mg_noise_realData_for_testing_mu2.reset_index()

        mg_noise_realData_for_simulation = pd.concat((mu_based for mu_based in [mg_noise_realData_for_simulation_mu1,
                                                                               mg_noise_realData_for_simulation_mu2]),
                                                     names=col_names, ignore_index=True)
        mg_noise_realData_for_testing = pd.concat((mu_based for mu_based in [mg_noise_realData_for_testing_mu1,
                                                                            mg_noise_realData_for_testing_mu2]),
                                                  names=col_names, ignore_index=True)

        # Write the simulation basis data
        mg_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/firstLevel_realData_for_simulation.log', encoding='utf-8',
                                          header=None, index=False)
        mg_noise_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/secondLevel_realData_for_simulation.log',
                                                encoding='utf-8', header=None, index=False)
        mg_cs_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/thirdLevel_realData_for_simulation.log',
                                             encoding='utf-8', header=None, index=False)
        mg_cs_st_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/fourthLevel_realData_for_simulation.log',
                                                encoding='utf-8', header=None, index=False)

        # Reindexing:
        mg_realData_for_simulation.index = np.arange(len(mg_realData_for_simulation.index))
        mg_noise_realData_for_simulation.index = np.arange(len(mg_noise_realData_for_simulation))
        mg_cs_realData_for_simulation.index = np.arange(len(mg_cs_realData_for_simulation))
        mg_cs_st_realData_for_simulation.index = np.arange(len(mg_cs_st_realData_for_simulation))

        # Analyzing and plotting the simulation basis data
        self.analyzeData(data_1=mg_realData_for_simulation,
                         data_1_title='mg_realData_for_simulation',
                         data_2=mg_noise_realData_for_simulation,
                         data_2_title='mg_noise_realData_for_simulation',
                         data_3=mg_cs_realData_for_simulation,
                         data_3_title='mg_cs_realData_for_simulation',
                         data_4=mg_cs_st_realData_for_simulation,
                         data_4_title='mg_cs_st_realData_for_simulation',
                         metric_1_path=metric_1_path,
                         metric_2_path=metric_2_path,
                         plt_prefix='for_simulation')

        # Save the testing data
        mg_realData_for_testing.to_csv(data_splits_path + 'for_testing/firstLevel_realData_for_testing.log', encoding='utf-8',
                                          header=None, index=False)
        mg_noise_realData_for_testing.to_csv(data_splits_path + 'for_testing/secondLevel_realData_for_testing.log', encoding='utf-8',
                                             header=None, index=False)
        mg_cs_realData_for_testing.to_csv(data_splits_path + 'for_testing/thirdLevel_realData_for_testing.log', encoding='utf-8',
                                          header=None, index=False)
        mg_cs_st_realData_for_testing.to_csv(data_splits_path + 'for_testing/fourthLevel_realData_for_testing.log', encoding='utf-8',
                                             header=None, index=False)

        # Reindexing:
        mg_realData_for_testing.index = np.arange(len(mg_realData_for_testing.index))
        mg_noise_realData_for_testing.index = np.arange(len(mg_noise_realData_for_testing))
        mg_cs_realData_for_testing.index = np.arange(len(mg_cs_realData_for_testing))
        mg_cs_st_realData_for_testing.index = np.arange(len(mg_cs_st_realData_for_testing))
        
        # Analyzing and plotting the testing basis data
        self.analyzeData(data_1=mg_realData_for_testing,
                         data_1_title='mg_realData_for_testing',
                         data_2=mg_noise_realData_for_testing,
                         data_2_title='mg_noise_realData_for_testing',
                         data_3=mg_cs_realData_for_testing,
                         data_3_title='mg_cs_realData_for_testing',
                         data_4=mg_cs_st_realData_for_testing,
                         data_4_title='mg_cs_st_realData_for_testing',
                         metric_1_path=metric_1_path,
                         metric_2_path=metric_2_path,
                         plt_prefix='for_testing')
        
        return mg_realData_for_simulation, mg_noise_realData_for_simulation, mg_cs_realData_for_simulation, \
               mg_cs_st_realData_for_simulation, mg_realData_for_testing, mg_noise_realData_for_testing, \
               mg_cs_realData_for_testing, mg_cs_st_realData_for_testing

    def all_realMemo_for_simulation_testing_split(self,
                                                  memo_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Logs/',
                                                  for_simulation=0.5,
                                                  metric_1_path='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Splits/Statistics/MatchingPairs_Statistics/All_Memo/',
                                                  metric_2_path='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Splits/Statistics/ExploitingScore_Statistics/All_Memo/',
                                                  data_splits_path='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Splits/All_Memo/'):

        """
        This function separates the given memo real data into simulation basis and testing data.
        Both the simulation basis and testing data will be extracted equally from all memo data path.
        It plots then the matching-pairs and punishments statistics on both the simulation basis and testing data to 
        show how similar they are.

        :param memo_basePath: 
        :param for_simulation: 
        :return: 
        """
        # First, read all memo data
        mg_cs_st_realData, mg_cs_realData, mg_realData, mg_noise_realData = self.read_memo_data(memo_basePath)

        # Then, separate them into simulation basis and testing data organized in the four emulated memory levels
        # 1. MG_CS_ST Level:
        mg_cs_st_realData_for_simulation, mg_cs_st_realData_for_testing = train_test_split(mg_cs_st_realData,
                                                                                           test_size=1-for_simulation)
        #mg_cs_st_realData_for_simulation = mg_cs_st_realData_for_simulation.reset_index()
        #mg_cs_st_realData_for_testing = mg_cs_st_realData_for_testing.reset_index()

        # 2. MG_CS Level:
        mg_cs_realData_for_simulation, mg_cs_realData_for_testing = train_test_split(mg_cs_realData,
                                                                                     test_size=1-for_simulation)
        #mg_cs_realData_for_simulation = mg_cs_realData_for_simulation.reset_index()
        #mg_cs_realData_for_testing = mg_cs_realData_for_testing.reset_index()

        # 3. MG Level:
        mg_realData_for_simulation, mg_realData_for_testing = train_test_split(mg_realData,
                                                                               test_size=1-for_simulation)
        #mg_realData_for_simulation = mg_realData_for_simulation.reset_index()
        #mg_realData_for_testing = mg_realData_for_testing.reset_index()

        # 4. MG_Noise Level:
        mg_noise_realData_for_simulation, mg_noise_realData_for_testing = train_test_split(mg_noise_realData,
                                                                                           test_size=1-for_simulation)
        #mg_noise_realData_for_simulation = mg_noise_realData_for_simulation.reset_index()
        #mg_noise_realData_for_testing = mg_noise_realData_for_testing.reset_index()

        # Write the simulation basis data
        mg_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/firstLevel_realData_for_simulation.log', encoding='utf-8',
                                          header=None, index=False)
        mg_noise_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/secondLevel_realData_for_simulation.log',
                                                encoding='utf-8', header=None, index=False)
        mg_cs_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/thirdLevel_realData_for_simulation.log',
                                             encoding='utf-8', header=None, index=False)
        mg_cs_st_realData_for_simulation.to_csv(data_splits_path + 'for_simulation/fourthLevel_realData_for_simulation.log',
                                                encoding='utf-8', header=None, index=False)

        # Reindexing:
        mg_realData_for_simulation.index = np.arange(len(mg_realData_for_simulation.index))
        mg_noise_realData_for_simulation.index = np.arange(len(mg_noise_realData_for_simulation))
        mg_cs_realData_for_simulation.index = np.arange(len(mg_cs_realData_for_simulation))
        mg_cs_st_realData_for_simulation.index = np.arange(len(mg_cs_st_realData_for_simulation))

        # Analyzing and plotting the simulation basis data
        self.analyzeData(data_1=mg_realData_for_simulation,
                         data_1_title='mg_realData_for_simulation',
                         data_2=mg_noise_realData_for_simulation,
                         data_2_title='mg_noise_realData_for_simulation',
                         data_3=mg_cs_realData_for_simulation,
                         data_3_title='mg_cs_realData_for_simulation',
                         data_4=mg_cs_st_realData_for_simulation,
                         data_4_title='mg_cs_st_realData_for_simulation',
                         metric_1_path=metric_1_path,
                         metric_2_path=metric_2_path,
                         plt_prefix='for_simulation')

        # Save the testing data
        mg_realData_for_testing.to_csv(data_splits_path + 'for_testing/firstLevel_realData_for_testing.log', encoding='utf-8',
                                       header=None, index=False)
        mg_noise_realData_for_testing.to_csv(data_splits_path + 'for_testing/secondLevel_realData_for_testing.log',
                                             encoding='utf-8',
                                             header=None, index=False)
        mg_cs_realData_for_testing.to_csv(data_splits_path + 'for_testing/thirdLevel_realData_for_testing.log', encoding='utf-8',
                                          header=None, index=False)
        mg_cs_st_realData_for_testing.to_csv(data_splits_path + 'for_testing/fourthLevel_realData_for_testing.log',
                                             encoding='utf-8',
                                             header=None, index=False)

        # Reindexing:
        mg_realData_for_testing.index = np.arange(len(mg_realData_for_testing.index))
        mg_noise_realData_for_testing.index = np.arange(len(mg_noise_realData_for_testing))
        mg_cs_realData_for_testing.index = np.arange(len(mg_cs_realData_for_testing))
        mg_cs_st_realData_for_testing.index = np.arange(len(mg_cs_st_realData_for_testing))

        # Analyzing and plotting the testing basis data
        self.analyzeData(data_1=mg_realData_for_testing,
                         data_1_title='mg_realData_for_testing',
                         data_2=mg_noise_realData_for_testing,
                         data_2_title='mg_noise_realData_for_testing',
                         data_3=mg_cs_realData_for_testing,
                         data_3_title='mg_cs_realData_for_testing',
                         data_4=mg_cs_st_realData_for_testing,
                         data_4_title='mg_cs_st_realData_for_testing',
                         metric_1_path=metric_1_path,
                         metric_2_path=metric_2_path,
                         plt_prefix='for_testing')

        return mg_realData_for_simulation, mg_noise_realData_for_simulation, mg_cs_realData_for_simulation, \
               mg_cs_st_realData_for_simulation, mg_realData_for_testing, mg_noise_realData_for_testing, \
               mg_cs_realData_for_testing, mg_cs_st_realData_for_testing


    def read_memo_data(self,
                       memo_data_basePath='/share/documents/msalous/DINCO/Data_Collection/memo/Memo_Logs/',
                       is_real_data = True):

        """
        This function reads memo data from the given path and returns them as four separated memory levels.
        
        :param memo_data_basePath: 
        :return: 
        """

        memo_all_logs = os.listdir(memo_data_basePath)

        # Read real data into pandas dataFrame
        cards = ['card_' + str(i) for i in range(1, self.sequence_length + 1)]  # card_1...card_40

        if is_real_data:
            col_names = cards + ['card_t' + str(i) for i in
                                 range(1, self.sequence_length + 1)]  # card_1...card_40, card_t1...card_t40
            col_names = col_names + ['totalTime', 'actualSum', 'userSum', 'shortStoryScore', 'label']
        else:
            col_names = cards + ['lda_feature_' + str(i) for i in range(1, 5)] + ['label']  # lda_features & label

        fourLabels_data = pd.concat(
            pd.read_csv(memo_data_basePath + memo_log, header=None, names=col_names)
            for memo_log in memo_all_logs)

        # Temp: get smaller set from data
        # fourLabels_data = fourLabels_data.sample(frac=0.1) # Shuffle
        # fourLabels_data.index = np.arange(len(fourLabels_data.index))

        print('Memo data: ' + str(fourLabels_data.shape))

        # Separate the real data into four labels:
        # Abbreviations meanings:
        # MG: Memory Game
        # CS: Cumulative Sum
        # ST: Short Story
        # N: Noise

        # It's important to distinguish between simulated and real data labels
        if is_real_data:
            mg_cs_st_data_label = 0
            mg_cs_data_label = 1
            mg_data_label = 2
            mg_noise_data_label = 3
        else:
            mg_data_label = 0
            mg_noise_data_label = 1
            mg_cs_data_label = 2
            mg_cs_st_data_label = 3

        mg_cs_st_data = fourLabels_data.loc[fourLabels_data['label'] == mg_cs_st_data_label]#.reset_index()
        mg_cs_data = fourLabels_data.loc[fourLabels_data['label'] == mg_cs_data_label]#.reset_index()
        mg_data = fourLabels_data.loc[fourLabels_data['label'] == mg_data_label]#.reset_index()
        mg_noise_data = fourLabels_data.loc[fourLabels_data['label'] == mg_noise_data_label]#.reset_index()

        # Temp: in case of using smaller set from data, and ensure equal size lists
        # mg_cs_st_data = mg_cs_st_data[:500]
        # mg_cs_data = mg_cs_data[:500]
        # mg_data = mg_data[:500]
        # mg_noise_data = mg_noise_data[:500]

        # Reindexing:
        mg_data.index = np.arange(len(mg_data.index))
        mg_noise_data.index = np.arange(len(mg_noise_data))
        mg_cs_data.index = np.arange(len(mg_cs_data))
        mg_cs_st_data.index = np.arange(len(mg_cs_st_data))

        print('mg_cs_st data: ' + str(mg_cs_st_data.shape))
        print('mg_cs data: ' + str(mg_cs_data.shape))
        print('mg data: ' + str(mg_data.shape))
        print('mg_noise data: ' + str(mg_noise_data.shape))

        return mg_cs_st_data, mg_cs_data, mg_data, mg_noise_data