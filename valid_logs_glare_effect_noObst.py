#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

def validate_log(log):
    needed_entries = ['1.1,', '2.1,', '3.1,', '4.1,', '5.1,', '6.1,', '7.1,', \
        '1.2,', '2.2,', '3.2,', '4.2,', '5.2,', '6.2,', '7.2,']
    for needed_entry in needed_entries:
        if needed_entry not in log:
            return False
    return True

def collect_glare_effect_logs(glare_path):
    with open(glare_path, "r") as glare_log_file:
        logs_with_probants = glare_log_file.read().strip().split('\n')
    logs = {}
    for entry in logs_with_probants:
        log_with_probant = entry.split(':')
        probant = log_with_probant[0]
        log = log_with_probant[1]
        logs[probant] = log
    return logs

def collect_noObts_logs(log_dir):
    logs = {}
    for probant in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, probant)):
            full_path = os.path.join(log_dir, probant, 'behavioural', \
                'g', '1' , 'b_mapped_sim_matrix.txt')
            if os.path.isfile(full_path):
                with open(full_path, "r") as log_file:
                    log_text = log_file.read() 

                if validate_log(log_text):
                    logs[probant] = log_text
                else:
                    print('Found invalid log for probant %s in level 1 (recording 1)' %probant)
            else:
                print('Found no log for probant %s in level g (recording 1)' %probant)
        else: 
            print('%s does not exist!' %os.path.join(log_dir, probant))
    return logs

def validate(noObst_logs, glare_effect_logs):
    valid_noObst_logs = ''
    valid_glare_effect_logs = ''
    glare_probants = glare_effect_logs.keys()
    for probant in noObst_logs.keys():
        if probant in glare_probants:
            noObst_log = noObst_logs[probant]
            glare_log = glare_effect_logs[probant]
            if validate_log(noObst_log) and validate_log(glare_log):
                valid_noObst_logs += noObst_log
                valid_glare_effect_logs += glare_log + '\n'
            else:
                print('For probant %s either noObst or glare effect was invalid.' %probant)
        else:
            print('Probant %s has no recording for glare effect and is therefore left out.' %probant)
    return (valid_noObst_logs, valid_glare_effect_logs)

if __name__ == "__main__":

    wd = os.getcwd()

    # Argument handling.
    parser = argparse.ArgumentParser(description='Used to collect the logs of all levels (except glare effect).')
    parser.add_argument('--l', default=('%s\\DBN_Data_Collection' %wd), \
        help='The directory the logs are stored in.')
    parser.add_argument('--g', default=('%s\\mapped_logs\\unchecked\\glare_effect\\mapped_logs_glare_effect_with_probants.txt' %wd), \
        help='The file the glare effecr logs containing the probant ids are stored in.')
    
    args = parser.parse_args()
    log_dir = args.l
    glare_path = args.g
    
    noObst_logs = collect_noObts_logs(log_dir)
    glare_effect_logs = collect_glare_effect_logs(glare_path)

    valid_logs = validate(noObst_logs, glare_effect_logs)
    noObst_logs = valid_logs[0]
    glare_effect_logs = valid_logs[1]

    with open('%s\\mapped_logs\\glare_effect_noObst_valid_logs\\valid_noObst_logs.txt' %wd, 'w') as noObst_logs_file:
            noObst_logs_file.write(noObst_logs)

    with open('%s\\mapped_logs\\glare_effect_noObst_valid_logs\\valid_glare_effect_logs.txt' %wd, 'w') as glare_logs_file:
            glare_logs_file.write(glare_effect_logs)
