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

def collect_logs(log_dir, level, valid):

    file_name = 'b_mapped_sim_matrix.txt'

    logs = ''
    for probant in os.listdir(log_dir):
        for i in range(1, 3):
            if os.path.isdir(os.path.join(log_dir, probant)):
                full_path = os.path.join(log_dir, probant, 'behavioural', \
                    level, '%s' %i , file_name)

                if os.path.isfile(full_path):
                    with open(full_path, "r") as log_file:
                        log_text = log_file.read() 
                    
                    if valid:
                        if validate_log(log_text):
                            logs += log_text
                        else:
                            print('Found invalid log for probant %s in level %s (recording %s)' %(probant, level, i))
                    else:
                        logs += log_text
                else:
                    print('Found no log for probant %s in level %s (recording %s)' %(probant, level, i))
            else: 
                print('%s does not exist!' %os.path.join(log_dir, probant))

    if valid:
        with open('%s\\mapped_logs\\valid\\%s\\combined_logs_%s.txt' %(wd, level, level), 'w') as logs_file:
            logs_file.write(logs)
    else: 
        with open('%s\\mapped_logs\\unchecked\\%s\\combined_logs_%s.txt' %(wd, level, level), 'w') as logs_file:
            logs_file.write(logs)

if __name__ == "__main__":

    wd = os.getcwd()

    # Argument handling.
    parser = argparse.ArgumentParser(description='Used to collect the logs of all levels (except glare effect).')
    parser.add_argument('--l', default=('%s\\DBN_Data_Collection' %wd), \
        help='The directory the logs are stored in.')
    parser.add_argument('--valid', action='store_true', help='Wheter only valid logs should be collected.')

    args = parser.parse_args()
    log_dir = args.l
    valid = args.valid

    levels = ['g', 'g_b', 'g_v', 'gm', 'gmb', 'gmv', 'gv', 'gvb', 'gvv']

    for level in levels:
        collect_logs(log_dir, level, valid)
    