#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
import os 
import json
from datetime import datetime
import pandas as pd

# TODO: 
'''
- models laden aus einen directory (sd-dir)
- load test data for each split (data-dir)
- for each test data for one split do the voting with the according 20 models (print decisions of each repition to check)
    --> result in final labels for the 20 glare effect and 20 no obstacle test data
- use the 40 labels for each split and determine accuracy of each split indipendently
- calculate average over splits
'''


if __name__ == "__main__":

    
