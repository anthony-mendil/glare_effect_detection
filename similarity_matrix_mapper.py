#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
from io import StringIO

global counter 
counter = 0

# Similarity matrix for glare effect. Created with 300 screenshots.
# With 0.9957 coverage of all combinations.
similarity_matrix = ['1.1=0.1358699983300959', '1.2=0.2159772743951057','1.3=0.30618357051356393',
	'1.4=0.26688764647597807','1.5=0.45477804580967557','1.6=0.27274591749087007',
	'1.7=0.35545683642475645','2.2=0.12561928232079897','2.3=0.3067811561649673',
	'2.4=0.20757827390702924','2.5=0.4876981720676538','2.6=0.19384665346434374',
	'2.7=0.3193321213727848','3.3=0.12303171290262678','3.4=0.22140280963975914',
	'3.5=0.2835260086763833','3.6=0.3905695697507862','3.7=0.5285013450399861',
	'4.4=0.11233870044589911','4.5=0.40628158549784504','4.6=0.268847547702396',
	'4.7=0.41256154291090963','5.5=0.1012097701465351','5.6=0.5834785652275842',
	'5.7=0.7209013937882208','6.6=0.11804123010975442','6.7=0.24761046897549838',
	'7.7=0.13173179920287795']


def map_similarity_matrix(original_log, mapped_cards_original_sim_matrix_log):
    '''
    this function maps the similarity matrix in :param mapped_cards_original_sim_matrix_log to use dynamic card coding 
    in sim matrix entries using the dynamic (mapped) card coding in this param and the original card coding from the 
    original log.
    
    :param original_log: contains non mapped motives, i.e. static motives (original motives)
    :param mapped_cards_original_sim_matrix_log: it has dynamic card coding but its sim matrix entries 
    use original staic coding (non mapped)!
    :return: mapped_motives_mapped_sim_matrix which is the mapped log with corresponding mapped similarity matrix
    '''
    # logs schema
    similarities_assignments = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7',
                                '2.2', '2.3', '2.4', '2.5', '2.6', '2.7',
                                '3.3', '3.4', '3.5', '3.6', '3.7',
                                '4.4', '4.5', '4.6', '4.7',
                                '5.5', '5.6', '5.7',
                                '6.6', '6.7',
                                '7.7']

    cards = ['card_' + str(i) for i in range(1, 41)]  # card_1...card_40
    cards_ts = ['card_t' + str(i) for i in range(1, 41)]  # card_t1...card_t40
    statistics_and_label = ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']

    # col_names
    original_col_names = cards + cards_ts
    mapped_col_names = similarities_assignments + cards + cards_ts + statistics_and_label
    
    # Dataframe for original log.
    original_log_df = pd.read_csv(original_log, header=None, names=original_col_names, index_col=False)

    with open(mapped_cards_original_sim_matrix_log, "r") as log_file:
        old_log = log_file.read().split(',')

    new_log = []
    new_log.extend(similarity_matrix)
    new_log.extend(old_log[21:])

    log = ''
    for entry in new_log:
        log += '%s,' %entry
    log = log[:-1]

    log = StringIO('''%s''' %log)
    mapped_log_df = pd.read_csv(log, sep=',', header=None, names=mapped_col_names, index_col=False)
    
    
    # Resultant data frame of mapped motives with corresponding mapped similarity matrix to be prepared for all games
    mapped_motives_mapped_sim_matrix = pd.DataFrame(columns=mapped_col_names)
    # force ts columns as int64 data type (to avoid automatic non-precise float data type)
    for ts in cards_ts:
        mapped_motives_mapped_sim_matrix = mapped_motives_mapped_sim_matrix.astype({ts: int})

    # map sim matrix
    # mapper
    motives_mapper = {}

    for card_num in range(1, 41):
        mapped_card = mapped_log_df.loc[0, 'card_' + str(card_num)]
        mapped_motive = int(mapped_card)
        if mapped_motive == 0:
            break

        original_card = original_log_df.loc[0, 'card_'+str(card_num)]
        original_motive = int(original_card)

        if mapped_motive not in motives_mapper:
            motives_mapper[mapped_motive] = original_motive

    global counter 
    # mapper is ready for this game
    print('Motives for game ' + str(counter) + ' are mapped!')
    counter += 1

    print(motives_mapper)

    # do map the similarity matrix entries
    for assignment in similarities_assignments:
        # map the assignment key
        from_motive = str(motives_mapper[int(assignment[0])])  # X in the mapped X.Y
        to_motive = str(motives_mapper[int(assignment[2])])  # Y in the mapped X.Y

        if (from_motive + '.' + to_motive) in mapped_log_df.columns:
            assignment_mapped_key = from_motive + '.' + to_motive
        else:
            assignment_mapped_key = to_motive + '.' + from_motive

        # get the assignment value
        # Before: ... = str(mapped_log_df.get_value(index=game_index, column=assignment_mapped_key))
        assignment_mapped_value = str(mapped_log_df.at[0, assignment_mapped_key])

        # mapped assignment
        mapped_assignment = assignment + '=' + assignment_mapped_value[4:]  # just the value, drop the 'X.Y='

        print('assignment=' + assignment)
        print('assignment_mapped_key=' + assignment_mapped_key)
        print('mapped assignment: ' + mapped_assignment)

        # add the mapped assignment
        mapped_motives_mapped_sim_matrix.at[0, assignment] = mapped_assignment

    # set the mapped cards, ts , statistics and labels of this game: game_index
    mapped_motives_mapped_sim_matrix.loc[[0], cards + cards_ts + statistics_and_label] =\
        mapped_log_df.loc[[0], cards + cards_ts + statistics_and_label]

    # End games loop

    print('Done!')
    print(mapped_motives_mapped_sim_matrix)

    return mapped_motives_mapped_sim_matrix

def load_logs(log_dir):
    '''
    Loads the original logs and the wrongly mapped logs.

    :param log_dir: The directory the logs are stored in.
    :return: The original logs and the wrongly mapped logs.
    '''
    original_logs = []
    mapped_wrong_logs = []
    for probant in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, probant)):
            full__base_path = os.path.join(log_dir, probant, 'behavioural', \
                probant, 'glare')
            for log_file in os.listdir(full__base_path):
                if '_original' in log_file:
                    full_path_original = os.path.join(full__base_path, log_file)
                    original_logs.append((full_path_original, probant))
                elif '_mapped' in log_file:
                    full_path_mapped_wrong = os.path.join(full__base_path, log_file)
                    mapped_wrong_logs.append(full_path_mapped_wrong)
    return (original_logs, mapped_wrong_logs)

def create_log_text(log_dataframe):
    '''
    Creates a log string out of the dataframe in the format 
    in that it is saved in the file. 

    :param log_dataframe: The log dataframe.
    :return: The log string.
    '''
    values = log_dataframe.values.tolist()[0]
    values = ','.join(['%s' %value for value in values])
    return values

if __name__ == "__main__":

    wd = os.getcwd()

    # Argument handling.
    parser = argparse.ArgumentParser(description='Used to create a correcly mapped similarity matrix.')
    parser.add_argument("--l", default=("%s\\DBN_Data_Collection" %wd), \
        help='The directory the logs are stored in.')

    args = parser.parse_args()
    log_dir = args.l

    if not os.path.exists(log_dir):
        print('The path for the logs does not exist. Exiting...')
        exit()

    logs = load_logs(log_dir)
    original_logs = logs[0]
    mapped_wrong_logs = logs[1]

    if len(original_logs) != len(mapped_wrong_logs):
        exit()

    logs_for_mapper = list(zip(original_logs, mapped_wrong_logs))

    mapped_correct_logs = ''
    for entry in logs_for_mapper:
        probant = entry[0][1]
        original_log = entry[0][0]
        mapped_wrong_log = entry[1]

        if (probant not in original_log) or (probant not in mapped_wrong_log):
            print('The probant was not correcly allocated.')
            exit()

        mapped_correct_log = map_similarity_matrix(original_log, \
        mapped_wrong_log)
        correct_log_text = create_log_text(mapped_correct_log)
        mapped_correct_logs += '%s:' %probant + correct_log_text + '\n'

    with open('%s\\mapped_logs\\unchecked\\glare_effect\\mapped_logs_glare_effect_with_probants.txt' %wd, 'w') as mapped_correct:
        mapped_correct.write(mapped_correct_logs)