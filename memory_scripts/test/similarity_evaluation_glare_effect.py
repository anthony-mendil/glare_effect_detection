# This script has 2 uses:
# 1. Perform paired-sample t-tests or various comparissons 
# between real and simulated no obstacle and glare effect games. 
# 2. Plott the average penalties and matching pairs per round 
# in real and simulated no obsatcle and glare effect games.
# By Anthony Mendil in 2020.

import pandas as pd
import sys
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
sys.path.append('..')
from src.memory_data_analysis_glare_effect import MemoryDataAnalyzer

# The number of rounds recorded for each game. 
sequence_length = 40  # first 20 rounds
# Base path of the data used. 
prefix = 'C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\'

# Components of the data. 
similarities_assignments_noObst = ['Assign_' + str(i) for i in range(1, 22)]
similarities_assignments_glare = ['Assign_' + str(i) for i in range(1, 29)]

cards = ['card_' + str(i) for i in range(1, sequence_length+1)] # card_1...card_40
cards_ts = ['card_t' + str(i) for i in range(1, sequence_length+1)] # card_1...card_40, card_t1...card_t40
statistics_and_label = ['totalTime','actualSum','userSum','userScore','label']
rmse = ['rmse']

# Columns of the dataframes. 
min_cols = cards + cards_ts + statistics_and_label
real_data_all_cols_noObst = similarities_assignments_noObst + cards + cards_ts + statistics_and_label
real_data_all_cols_glare = similarities_assignments_glare + cards + cards_ts + statistics_and_label
sim_data_all_cols = cards + statistics_and_label
sim_data_all_cols_rmse = rmse + sim_data_all_cols

# Real no obstacle logs.
allLabels_standardGame_realData_path = prefix + 'mapped_logs\\glare_effect_noObst_valid_logs\\valid_noObst_logs.txt' 
allLabels_standardGame_realData = pd.read_csv(allLabels_standardGame_realData_path, header=None, names=real_data_all_cols_noObst)
allLabels_standardGame_realData = allLabels_standardGame_realData.loc[:, min_cols]
# Real glare effect logs. 
allLabels_glareEffect_realData_path = prefix + 'mapped_logs\\glare_effect_noObst_valid_logs\\valid_glare_effect_logs.txt' 
allLabels_glareEffect_realData = pd.read_csv(allLabels_glareEffect_realData_path, header=None, names=real_data_all_cols_glare)
allLabels_glareEffect_realData = allLabels_glareEffect_realData.loc[:, min_cols]

# Number of simulated games per real game inlcuded. 
simulated_logs_per_participant = 15 # 1000 

# No obstacle simulations.
standardGame_simulatedData_path = prefix + 'simulation\\best_simulations\\simulated_noObst_sorted.log'
standardGame_simulatedData = pd.read_csv(standardGame_simulatedData_path, header=None, names=sim_data_all_cols_rmse)
standardGame_simulatedData = standardGame_simulatedData.loc[:, sim_data_all_cols]
# Glare effect simulations. 
glareEffect_simulatedData_path = prefix + 'simulation\\best_simulations\\simulated_glare_effect_sorted.log'
glareEffect_simulatedData = pd.read_csv(glareEffect_simulatedData_path, header=None, names=sim_data_all_cols_rmse)
glareEffect_simulatedData = glareEffect_simulatedData.loc[:, sim_data_all_cols]

###################

# Only using a specified number of the best simulations. 
index_list = []
for i in range(20):
    for l in range(simulated_logs_per_participant):
        index_list.append((i * 1000) + l)

standardGame_simulatedData = standardGame_simulatedData.iloc[index_list, :]
standardGame_simulatedData = standardGame_simulatedData.reset_index(drop=True)
glareEffect_simulatedData = glareEffect_simulatedData.iloc[index_list, :]
glareEffect_simulatedData = glareEffect_simulatedData.reset_index(drop=True)

###################

print("#########################")

# Starting index of included rounds.
start = 0
# End index of included rounds.
end = 10

# Preparing the data for the statistical tests and the plotting of the performance measurements. 
analyzer = MemoryDataAnalyzer()

data_1_numofMatchingCards_allGames, data_1_ExploitingScore_allGames, data_2_numofMatchingCards_allGames, data_2_ExploitingScore_allGames = \
    analyzer.doStatistics(allLabels_standardGame_realData, allLabels_glareEffect_realData, max_num_rounds=20)
data_3_numofMatchingCards_allGames, data_3_ExploitingScore_allGames, data_4_numofMatchingCards_allGames, data_4_ExploitingScore_allGames = \
    analyzer.doStatistics(standardGame_simulatedData, glareEffect_simulatedData, max_num_rounds=20)

standardGame_realData_numofMatchingCards_allGames = np.array(data_1_numofMatchingCards_allGames)
standardGame_realData_numofMatchingCards_mean = np.mean(standardGame_realData_numofMatchingCards_allGames, axis=0)
standardGame_realData_numofMatchingCards_mean = standardGame_realData_numofMatchingCards_mean[start:end]

standardGame_realData_ExploitingScore_allGames = np.array(data_1_ExploitingScore_allGames)
standardGame_realData_ExploitingScore_mean = np.mean(standardGame_realData_ExploitingScore_allGames, axis=0)
standardGame_realData_ExploitingScore_mean = standardGame_realData_ExploitingScore_mean[start:end]

glareEffect_realData_numofMatchingCards_allGames = np.array(data_2_numofMatchingCards_allGames)
glareEffect_realData_numofMatchingCards_mean = np.mean(glareEffect_realData_numofMatchingCards_allGames, axis=0)
glareEffect_realData_numofMatchingCards_mean = glareEffect_realData_numofMatchingCards_mean[start:end]

glareEffect_realData_ExploitingScore_allGames = np.array(data_2_ExploitingScore_allGames)
glareEffect_realData_ExploitingScore_mean = np.mean(glareEffect_realData_ExploitingScore_allGames, axis=0)
glareEffect_realData_ExploitingScore_mean = glareEffect_realData_ExploitingScore_mean[start:end]

standardGame_simulatedData_numofMatchingCards_allGames = np.array(data_3_numofMatchingCards_allGames)
standardGame_simulatedData_numofMatchingCards_mean = np.mean(standardGame_simulatedData_numofMatchingCards_allGames, axis=0)
standardGame_simulatedData_numofMatchingCards_mean = standardGame_simulatedData_numofMatchingCards_mean[start:end]

standardGame_simulatedData_ExploitingScore_allGames = np.array(data_3_ExploitingScore_allGames)
standardGame_simulatedData_ExploitingScore_mean = np.mean(standardGame_simulatedData_ExploitingScore_allGames, axis=0)
standardGame_simulatedData_ExploitingScore_mean = standardGame_simulatedData_ExploitingScore_mean[start:end]

glareEffect_simulatedData_numofMatchingCards_allGames = np.array(data_4_numofMatchingCards_allGames)
glareEffect_simulatedData_numofMatchingCards_mean = np.mean(glareEffect_simulatedData_numofMatchingCards_allGames, axis=0)
glareEffect_simulatedData_numofMatchingCards_mean = glareEffect_simulatedData_numofMatchingCards_mean[start:end]

glareEffect_simulatedData_ExploitingScore_allGames = np.array(data_4_ExploitingScore_allGames)
glareEffect_simulatedData_ExploitingScore_mean = np.mean(glareEffect_simulatedData_ExploitingScore_allGames, axis=0)
glareEffect_simulatedData_ExploitingScore_mean = glareEffect_simulatedData_ExploitingScore_mean[start:end]


print(allLabels_standardGame_realData)
print(allLabels_glareEffect_realData)
print(standardGame_simulatedData)
print(glareEffect_simulatedData)


def calc(data_1_numofMatchingCards_mean, data_1_ExploitingScore_mean, data_2_numofMatchingCards_mean, data_2_ExploitingScore_mean):
    '''
    Perfroms multiple paired-sample t-tests for the passed lists of mean values. 

    :param data_1_numofMatchingCards_mean: Mean numbers of matchings pairs in each round of the first data.
    :param data_1_ExploitingScore_mean: Mean numbers of penalties in each round of the first data.
    :param data_2_numofMatchingCards_mean: Mean numbers of matchings pairs in each round of the second data.
    :param data_2_ExploitingScore_mean: Mean numbers of penalties in each round of the second data.
    '''

    print("##########")

    matching_data1_vs_data2_t, matching_data1_vs_data2_p = ttest_rel(data_1_numofMatchingCards_mean,
                                                                         data_2_numofMatchingCards_mean)
    print("mp_t = " + str(matching_data1_vs_data2_t))
    print("mp_p = " + str(matching_data1_vs_data2_p))

    penalties_data1_vs_data2_t, penalties_data1_vs_data2_p = ttest_rel(data_1_ExploitingScore_mean,
                                                                           data_2_ExploitingScore_mean)
    print("p_t = " + str(penalties_data1_vs_data2_t))
    print("p_p = " + str(penalties_data1_vs_data2_p))

    print("##########")

# Performing the paired-sample t-tests for the different comparissons.
print("\nreal_no_obst_comparisons: \n")
print("r_n vs. r_g")
calc(standardGame_realData_numofMatchingCards_mean, standardGame_realData_ExploitingScore_mean, glareEffect_realData_numofMatchingCards_mean, glareEffect_realData_ExploitingScore_mean)

print("------")
print(standardGame_realData_numofMatchingCards_mean)
print(glareEffect_realData_numofMatchingCards_mean)
print("--")
print(standardGame_realData_ExploitingScore_mean)
print(glareEffect_realData_ExploitingScore_mean)
print("------")

print("r_n vs. s_n")
calc(standardGame_realData_numofMatchingCards_mean, standardGame_realData_ExploitingScore_mean, standardGame_simulatedData_numofMatchingCards_mean, standardGame_simulatedData_ExploitingScore_mean)

print("------")
print(standardGame_realData_numofMatchingCards_mean)
print(standardGame_simulatedData_numofMatchingCards_mean)
print("--")
print(standardGame_realData_ExploitingScore_mean)
print(standardGame_simulatedData_ExploitingScore_mean)
print("------")

print("r_n vs. s_g")
calc(standardGame_realData_numofMatchingCards_mean, standardGame_realData_ExploitingScore_mean, glareEffect_simulatedData_numofMatchingCards_mean, glareEffect_simulatedData_ExploitingScore_mean)

print("\nreal_glare_comparisons: \n")
print("r_g vs. s_n")
calc(glareEffect_realData_numofMatchingCards_mean, glareEffect_realData_ExploitingScore_mean, standardGame_simulatedData_numofMatchingCards_mean, standardGame_simulatedData_ExploitingScore_mean)
print("r_g vs. s_g")
calc(glareEffect_realData_numofMatchingCards_mean, glareEffect_realData_ExploitingScore_mean, glareEffect_simulatedData_numofMatchingCards_mean, glareEffect_simulatedData_ExploitingScore_mean)

print("------")
print(glareEffect_realData_numofMatchingCards_mean)
print(glareEffect_simulatedData_numofMatchingCards_mean)
print("--")
print(glareEffect_realData_ExploitingScore_mean)
print(glareEffect_simulatedData_ExploitingScore_mean)
print("------")

print("\nsim_no_obst_comparisons: \n")
print("s_n vs. s_g")
calc(standardGame_simulatedData_numofMatchingCards_mean, standardGame_simulatedData_ExploitingScore_mean, glareEffect_simulatedData_numofMatchingCards_mean, glareEffect_simulatedData_ExploitingScore_mean)


# Plotting the mean matching pairs and penalties in real and simulated no obstacle and glare effect games. 
data_1_numofMatchingCards_mean, \
data_2_numofMatchingCards_mean, \
data_3_numofMatchingCards_mean, \
data_4_numofMatchingCards_mean, \
 \
data_1_ExploitingScore_mean, \
data_2_ExploitingScore_mean, \
data_3_ExploitingScore_mean, \
data_4_ExploitingScore_mean, \
_, _, _, _, _, _, _, _, = \
    analyzer.analyzeData_visualObstacle(data_1=allLabels_standardGame_realData,  
                                        data_1_title='real no obstacle', 
                                        data_2=allLabels_glareEffect_realData, 
                                        data_2_title='real glare effect', 
                                        data_3=standardGame_simulatedData,  
                                        data_3_title='simulated no obstacle', 
                                        data_4=glareEffect_simulatedData, 
                                        data_4_title='simulated glare effect', 
                                        metric_1_path=prefix + 'simulation\\simulation_statistics\\dumps\\',
                                        metric_2_path=prefix + 'simulation\\simulation_statistics\\dumps\\',

                                        plotted_levels=[True, True, True, True],
                                        max_num_rounds=20
                                        )