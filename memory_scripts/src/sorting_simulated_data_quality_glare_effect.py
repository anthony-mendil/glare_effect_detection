# This script sorts the simulated data according to their similarity to their real data references:
# Input: real data references and their simulated data (typically 1000 simulated episode per real data episode)
# Output: quality-based (lowest RMSE) sorted simulated data written on files, file per memory condition e.g. noObs_noAdapt.
# By Anthony Mendil in 2020. (big parts were copied and adjusted in ordedr to be utilized) 

import pandas as pd
import sys
sys.path.append('..')
from src.memory_data_analysis import MemoryDataAnalyzer

# The number of simulated games per real one.
num_training_pro_individual = 1000 
# Number of subjects. 
num_individuals = 20
# Number of recorded steps (two steps are one round). 
sequence_length = 40

naming_map = {0: 'simulated_noObst_sorted_rd_after20_02.log',
              1: 'simulated_glare_effect_sorted.log'}

sim_data_sorter = MemoryDataAnalyzer()

# The number of simulated games per real one used in the sorting.
num_sim_data_for_sorting = 1000  # 10  # all the 1000, or random set e.g. 3, 10 etc.

# Input path.
prefix = 'C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\'

# Output path. 
best_simulated_data_base_path = prefix + 'simulation\\best_simulations\\'

# Components of the data.
similarities_assignments_noObst = ['Assign_' + str(i) for i in range(1, 22)]
similarities_assignments_glare = ['Assign_' + str(i) for i in range(1, 29)]

cards = ['card_' + str(i) for i in range(1, sequence_length+1)] # card_1...card_40
cards_ts = ['card_t' + str(i) for i in range(1, sequence_length+1)] # card_1...card_40, card_t1...card_t40
statistics_and_label = ['totalTime','actualSum','userSum','userScore','label']

# Columns of the dataframes. 
min_cols = cards + cards_ts + statistics_and_label
real_data_all_cols_noObst = similarities_assignments_noObst + cards + cards_ts + statistics_and_label
real_data_all_cols_glare = similarities_assignments_glare + cards + cards_ts + statistics_and_label
sim_data_all_cols = cards + statistics_and_label

# Real no obstacle logs. 
first_level_rd = pd.read_csv(prefix + 'mapped_logs\\glare_effect_noObst_valid_logs\\valid_noObst_logs.txt', names=real_data_all_cols_noObst, header=None)
first_level_rd = first_level_rd.loc[:, min_cols]

# Real glare effect logs. 
second_level_rd = pd.read_csv(prefix + 'mapped_logs\\glare_effect_noObst_valid_logs\\valid_glare_effect_logs.txt', names=real_data_all_cols_glare, header=None)
second_level_rd = second_level_rd.loc[:, min_cols]

print('Real data loaded!!\n')

print('first_level_rd: ' + str(first_level_rd.shape))
print('second_level_rd: ' + str(second_level_rd.shape))

# Loading the simulated data. 
first_level_sd = pd.read_csv(prefix + 'simulation\\simulated_logs\\noObst\\simulated_logs_noObst_rd_after20_02.log', names=sim_data_all_cols, header=None)
second_level_sd = pd.read_csv(prefix + 'simulation\\simulated_logs\\glare_effect\\simulated_logs_glare_effect.log', names=sim_data_all_cols, header=None)

print('\nSimulated data loaded!!\n')

print('first_level_sd: ' + str(first_level_sd.shape))
print('second_level_sd: ' + str(second_level_sd.shape))

# tuple all data levels
sd_all_levels = (first_level_sd, second_level_sd) 
rd_all_levels = (first_level_rd, second_level_rd) 

# Sorting the simulations according to how close their performance is
# to that of the real game used to simulate them. 
for index, (sd_data, rd_data) in enumerate(zip(sd_all_levels, rd_all_levels)):
    # packed sim data[[subject1_1000Episodes]...[subject49_1000Episodes]]
    sd_packed_list = [sd_data[i * num_training_pro_individual:
                              (i * num_training_pro_individual) + num_training_pro_individual]
                      for i in range(0, num_individuals)]

    # shuffled packed sim data: safe, shuffle data for each subject
    sd_packed_list_shuffled = []
    for i in sd_packed_list:
        sd_packed_list_shuffled.append(i.sample(frac=1))

    # sim data to be sorted
    sd_packed_list_for_sorting = [i[0: num_sim_data_for_sorting] for i in sd_packed_list_shuffled]

    # sort sim data: result is list((RMSE, sim_game_df))
    sorted_sd_list = [sim_data_sorter.sort_simulated_data_quality(pd.DataFrame([rd_game], columns=min_cols),
                                                                  sd_games)  # [0][1] not only best[0], but also all,
                                                                             # not only df [][1], but also with RMSE
                      for rd_game, sd_games in zip(rd_data.to_numpy(), sd_packed_list_for_sorting)]

    sorted_sd_df = pd.concat([individual_rmse_df_tuple[1] for individual_sorted_list in sorted_sd_list
                              for individual_rmse_df_tuple in individual_sorted_list], axis=0, ignore_index=True)

    sorted_rmse_df = pd.DataFrame({'rmse': ['rmse=' + str(individual_rmse_df_tuple[0])
                                            for individual_sorted_list in sorted_sd_list
                                            for individual_rmse_df_tuple in individual_sorted_list]},
                                  columns=['rmse'])

    sorted_rmse_sd_df = pd.concat([sorted_rmse_df, sorted_sd_df], axis=1)  # df (rmse=xx, 1.1, 2.1..) to be written

    # write resultant data frame
    sorted_rmse_sd_df.to_csv(best_simulated_data_base_path + naming_map[index],
                             encoding='utf-8', header=False, index=False)
    print('Level ' + str(index+1) + ': sorted and written!!')

print('\nAll simulated data levels have been packed, shuffled, sorted and written!!')