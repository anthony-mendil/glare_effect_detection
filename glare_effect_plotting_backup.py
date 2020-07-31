# Required libraries
import pandas as pd
import sys
from scipy.stats import ttest_ind

sys.path.append('..')

from src.memory_data_analysis import MemoryDataAnalyzer

sequence_length = 40  # first 20 rounds, i.e. first 40 chosen cards
# Paths
prefix = 'C:\\Users\\dylin\\Documents\\BA_Glare_Effect\\'
# prefix = '/run/user/1000/gvfs/smb-share:server=file.csl.uni-bremen.de,share='

# Components of the data
similarities_assignments_noObst = ['Assign_' + str(i) for i in range(1, 22)]
similarities_assignments_glare = ['Assign_' + str(i) for i in range(1, 29)]

cards = ['card_' + str(i) for i in range(1, sequence_length+1)] # card_1...card_40
cards_ts = ['card_t' + str(i) for i in range(1, sequence_length+1)] # card_1...card_40, card_t1...card_t40
statistics_and_label = ['totalTime','actualSum','userSum','userScore','label']
rmse = ['rmse']
#col_names = similarities_assignments + cards + cards_ts + statistics_and_label

# Columns 
min_cols = cards + cards_ts + statistics_and_label
real_data_all_cols_noObst = similarities_assignments_noObst + cards + cards_ts + statistics_and_label
real_data_all_cols_glare = similarities_assignments_glare + cards + cards_ts + statistics_and_label
sim_data_all_cols = cards + statistics_and_label
sim_data_all_cols_rmse = rmse + sim_data_all_cols

# No obstacle real
allLabels_standardGame_realData_path = prefix + 'mapped_logs\\glare_effect_noObst_valid_logs\\valid_noObst_logs.txt' 
allLabels_standardGame_realData = pd.read_csv(allLabels_standardGame_realData_path, header=None, names=real_data_all_cols_noObst)
allLabels_standardGame_realData = allLabels_standardGame_realData.loc[:, min_cols]
# Glare effect real
allLabels_glareEffect_realData_path = prefix + 'mapped_logs\\glare_effect_noObst_valid_logs\\valid_glare_effect_logs.txt' 
allLabels_glareEffect_realData = pd.read_csv(allLabels_glareEffect_realData_path, header=None, names=real_data_all_cols_glare)
allLabels_glareEffect_realData = allLabels_glareEffect_realData.loc[:, min_cols]

simulated_logs_per_participant = 1
# TODO: Sortiere nehen statt normale und RMSE column auslassen. Und nr_simulated_data einbauen sodass nur so viele genommen werden. Jeweils die ersten bei den einzelnen Personen!

# No obstacle simulated 
standardGame_simulatedData_path = prefix + 'simulation\\simulated_logs\\noObst\\simulated_logs_noObst_optimized.log'
standardGame_simulatedData = pd.read_csv(standardGame_simulatedData_path, header=None, names=sim_data_all_cols)
standardGame_simulatedData = standardGame_simulatedData.loc[:, sim_data_all_cols]
# Glare effect simulated 
glareEffect_simulatedData_path = prefix + 'simulation\\simulated_logs\\glare_effect\\simulated_logs_glare_effect.log'
glareEffect_simulatedData = pd.read_csv(glareEffect_simulatedData_path, header=None, names=sim_data_all_cols)

print("#########################")

print(allLabels_standardGame_realData)
print(allLabels_glareEffect_realData)

print(standardGame_simulatedData)
print(glareEffect_simulatedData)

analyzer = MemoryDataAnalyzer()

# real data vs simulated data: standardGame vs visualObstacle
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

noObs_real_sim_t, noObs_real_sim_p = ttest_ind(data_1_numofMatchingCards_mean[:10], data_3_numofMatchingCards_mean[:10])
print('noObs_real_sim_t = ' + str(noObs_real_sim_t) + ', noObs_real_sim_p = ' + str(noObs_real_sim_p))

visObs_real_sim_t, visObs_real_sim_p = ttest_ind(data_2_numofMatchingCards_mean[:10], data_4_numofMatchingCards_mean[:10])
print('visObs_real_sim_t = ' + str(visObs_real_sim_t) + ', visObs_real_sim_p = ' + str(visObs_real_sim_p))