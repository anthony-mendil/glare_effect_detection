#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Used to plot a similarity matrix. 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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

# Creating an array containing the values in the correct position for the plot. 
entries_list = np.zeros((7, 7))
for entry in similarity_matrix:
    value = (entry.split('='))[1]
    colour_codes = (entry.split('='))[0].split('.')
    first_index = int(colour_codes[0]) - 1
    second_index = int(colour_codes[1]) - 1
    if first_index == second_index:
        entries_list[first_index][first_index] = value
    else:
        entries_list[first_index][second_index] = value
        entries_list[second_index][first_index] = value

# The labels for the axes.
df = pd.DataFrame(entries_list, 
    columns = ['orange', 'brown', 'green', 'dark green', 'light green', 'dark red', 'red'],
    index = ['orange', 'brown', 'green', 'dark green', 'light green', 'dark red', 'red'])

print(df)

# Only showing the necessary values. 
mask = np.tril(np.ones(df.shape)).astype(np.bool)[0:8,0:8]
mask = np.asarray([[not entry for entry in row] for row in mask])

# Plotting the similarity matrix. 
plt.figure(figsize=(13, 13))
ax = sns.heatmap(
    df,
    cmap='binary',
    square = True,
    linewidth=1, 
    cbar_kws={'label': 'downscaled Delta E value'},
    mask=mask,
    annot=True
)
ax.set(xlabel="Colours",
      ylabel="Colours")

plt.yticks(np.arange(7) + 0.5, \
    ('orange', 'brown', 'green', 'dark green', 'light green', 'dark red', 'red'), \
    rotation=0, fontsize="10", va="center")
plt.show()