
# This script splits simulated and real data according to leave_one_subject_out strategy.
# Thus, cross validation is applied where each fold has one test person.
# By Anthony Mendil in 2020.

import sys
sys.path.append('..')
from src.memory_data_analysis_glare_effect import MemoryDataAnalyzer

splitter = MemoryDataAnalyzer()

# Creating and saving the raw data splits for no obstacle and glare effect games. 
splitter.leave_one_subject_out_cv(num_of_subjects=20,
                                  num_subject_game_trials=1,
                                  simulated_real=[True, True],
                                  rmse_col=True
                                  )
