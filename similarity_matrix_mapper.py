#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


def map_similarity_matrix(original_log, mapped_cards_original_sim_matrix_log):
    '''
    this function maps the similarity matrix in :param mapped_cards_original_sim_matrix_log to use dynamic card coding 
    in sim matrix entries using the dynamic (mapped) card coding in this param and the original card coding from the 
    original log :param original_log
    :param original_log: contains non mapped motives, i.e. static motives (original motives)
    :param mapped_cards_original_sim_matrix_log: it has dynamic card coding but its sim matrix entries 
    use original staic coding (non mapped)!
    :return: mapped_motives_mapped_sim_matrix which is the mapped log with corresponding mapped similarity matrix
    '''

    # logs schema
    similarities_assignments = ['1.2', '1.3', '1.4', '1.5', '1.6', '1.7',
                                '2.3', '2.4', '2.5', '2.6', '2.7',
                                '3.4', '3.5', '3.6', '3.7',
                                '4.5', '4.6', '4.7',
                                '5.6', '5.7',
                                '6.7']
    cards = ['card_' + str(i) for i in range(1, 41)]  # card_1...card_40
    cards_ts = ['card_t' + str(i) for i in range(1, 41)]  # card_t1...card_t40
    statistics_and_label = ['totalTime', 'actualSum', 'userSum', 'userScore', 'label']

    # col_names
    original_col_names = cards + cards_ts
    mapped_col_names = similarities_assignments + cards + cards_ts + statistics_and_label

    # data frames
    original_log_df = pd.read_csv(original_log, header=None, names=original_col_names, index_col=False)
    mapped_log_df = pd.read_csv(mapped_cards_original_sim_matrix_log, header=None, names=mapped_col_names, index_col=False)

    # Resultant data frame of mapped motives with corresponding mapped similarity matrix to be prepared for all games
    mapped_motives_mapped_sim_matrix = pd.DataFrame(columns=mapped_col_names)
    # force ts columns as int64 data type (to avoid automatic non-precise float data type)
    for ts in cards_ts:
        mapped_motives_mapped_sim_matrix = mapped_motives_mapped_sim_matrix.astype({ts: int})

    # map sim matrices in all games
    for game_index in range(len(mapped_log_df)):
        # mapper
        motives_mapper = {}

        for card_num in range(1, 41):
            mapped_card = mapped_log_df.loc[game_index, 'card_' + str(card_num)]
            mapped_motive = int(mapped_card)
            if mapped_motive == 0:
                break

            original_card = original_log_df.loc[game_index, 'card_'+str(card_num)]
            original_motive = int(original_card)

            if mapped_motive not in motives_mapper:
                motives_mapper[mapped_motive] = original_motive

        # mapper is ready for this game
        print('Motives for game ' + str(game_index) + ' are mapped!')
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
            assignment_mapped_value = str(mapped_log_df.get_value(index=game_index, col=assignment_mapped_key))

            # mapped assignment
            mapped_assignment = assignment + '=' + assignment_mapped_value[4:]  # just the value, drop the 'X.Y='

            print('assignment=' + assignment)
            print('assignment_mapped_key=' + assignment_mapped_key)
            print('mapped assignment: ' + mapped_assignment)

            # add the mapped assignment
            mapped_motives_mapped_sim_matrix.at[game_index, assignment] = mapped_assignment

        # set the mapped cards, ts , statistics and labels of this game: game_index
        mapped_motives_mapped_sim_matrix.loc[[game_index], cards + cards_ts + statistics_and_label] =\
            mapped_log_df.loc[[game_index], cards + cards_ts + statistics_and_label]

        # End games loop

    print('Done!')
    print(mapped_motives_mapped_sim_matrix)

    return mapped_motives_mapped_sim_matrix
