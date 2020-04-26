#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from PIL import Image 
import numpy as np
import itertools
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

# 150 * 150 pixels are taken for each card. Not whole card because 
# small differneces may appear between pictures in terms of the pixel. 
# The tuple value represents the koordinates of the top left corner for each card. 
# cards are about 250x250 and have 30 pixels between them -> 280 pixels until next one.
# Manually checked. Worls only if the resoulktion of the screenshots is 1080p.
# Otherwise adjust values. 
card_corners = [
    (320, 270),
    (600, 270),
    (880, 270),
    (1160, 270),
    (1440, 270),
    (320, 550),
    (600, 550),
    (880, 550),
    (1160, 550),
    (1440, 550),
    (320, 830),
    (600, 830),
    (880, 830),
    (1160, 830)
]

def load_images(image_dir):
    images = []
    for r, _, f in os.walk(image_dir):
        for file_name in f:
            if '.PNG' or '.png' in file_name:
                full_path = os.path.join(r, file_name)
                image = Image.open(full_path)
                images.append(image)
    return images 

# Extracts the rgb values for the image.
# Calculates average for the area.
# Returns list of average colors for all 14 cards. 
# Determining of color is correct: checked values.
def determine_glare_rgb_values(image):
    glare_rgb_values = []
    for corner in card_corners:
        x_border = corner[0] + 150
        y_border = corner[1] + 150
        card_values = []
        for x in range(corner[0], x_border, 1):
            for y in range(corner[1], y_border, 1):
                coordinates = x, y
                pixel_values = image.getpixel(coordinates)
                card_values.append(pixel_values[:-1])
        card_r = int(round(np.mean([color[0] for color in card_values])))
        card_g = int(round(np.mean([color[1] for color in card_values])))
        card_b = int(round(np.mean([color[2] for color in card_values])))
        glare_rgb_values.append((card_r, card_g, card_b))
    return glare_rgb_values

def load_original_colors(colors_path):
    if os.path.exists(colors_path):
            with open(colors_path, "r") as colors_file:
                return colors_file.read()

# Determines dinstance between colors with delta e fomula. 
def determine_distance(color_1_rgb, color_2_rgb):
    lab_1 = rgb2lab(color_1_rgb)
    lab_1 = LabColor(lab_1[0], lab_1[1], lab_1[2])
    lab_2 = rgb2lab(color_2_rgb)
    lab_2 = LabColor(lab_2[0], lab_2[1], lab_2[2])
    detlta_e = delta_e_cie1976(lab_1, lab_2)
    # Das ist erweiterung die besser sein soll und als standard festgelegt wurde. (wikpedia)
    # detlta_e = delta_e_cie2000(lab_1, lab_2, Kl=1, Kc=1, Kh=1)
    return detlta_e

def rgb2lab(inputColor):
    rgb = []
    for rgb_value in inputColor:
        rgb_value = float(rgb_value) / 255
        if rgb_value > 0.04045:
            rgb_value = ((rgb_value + 0.055) / 1.055) ** 2.4
        else:
            rgb_value = rgb_value / 12.92
        rgb.append(rgb_value * 100)

    values = []
    values.append(float(rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805) / 95.047)
    values.append(float(rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722) / 100.0)
    values.append(float(rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9505) / 108.883)
   
    for value_idx in range(len(values)):
        if values[value_idx] > 0.008856:
            values[value_idx] = values[value_idx] ** (0.3333333333333333)
        else:
            values[value_idx] = (7.787 * values[value_idx]) + (16 / 116)

    Lab = []
    Lab.append((116 * values[1]) - 16)
    Lab.append(500 * (values[0] - values[1]))
    Lab.append(200 * (values[1] - values[2]))
    return Lab

# Returns a matrix for a screenshot.
def create_similarity_matrix(rgb_values, original_colors):
    combinations_for_calculation = []
    matrix_values = []
    for i in range(len(rgb_values)):
        original_color_1 = original_colors[i]
        glare_rgb_1 = rgb_values[i]

        for l in range(len(rgb_values)):
            original_color_2 = original_colors[l]
            glare_rgb_2 = rgb_values[l]
            if original_color_1[2] != original_color_2[2] and \
                    (original_color_2, original_color_1) not in combinations_for_calculation:
                combinations_for_calculation.append((original_color_1, original_color_2))
                position = (original_color_1[1], original_color_2[1])
                matrix_values.append((determine_distance(glare_rgb_1, glare_rgb_2), position, \
                    (original_color_1[2], original_color_2[2])))
    
    values = {'1.1': [], '1.2': [], '1.3': [], '1.4': [], '1.5': [], '1.6': [], \
        '1.7': [], '2.2': [], '2.3': [], '2.4': [], '2.5': [], '2.6': [], '2.7': [], \
        '3.3': [], '3.4': [], '3.5': [], '3.6': [], '3.7': [], '4.4': [], '4.5': [], \
        '4.6': [], '4.7': [], '5.5': [], '5.6': [], '5.7': [], '6.6': [], '6.7': [], \
        '7.7': []}

    for value in matrix_values:
        position = value[1]
        if position[0] < position[1]:
            matrix_pos = '%s.%s' %(position[0], position[1])
        else:
            matrix_pos = '%s.%s' %(position[1], position[0])
        values[matrix_pos].append(value[0])

    # Result: For cells comparing the same color there is only one value (comparing the two cards)
    # and for cells comparing different colors there are four values (2 * 2).
    # Values with 1 - (value / 100)
    values = {key: 1 - (np.mean(values[key]) / 100.0) for key in values.keys()}
    # Values unchanged
    #values = {key: np.mean(values[key]) for key in values.keys()}
    return values

# Determines average of all matrices. 
def create_similarity_matrix_average(matrices_list):
    if not matrices_list:
        exit()
    
    similarity_matrix = ''
    for key in matrices_list[0].keys():
        cell_values = []
        for matrix in matrices_list:
            cell_values.append(matrix[key])
        similarity_matrix += '%s=%s,' %(key, np.mean(cell_values))
    return similarity_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Used to create a static similarity matrix \
        for the simulating games with the glare effect.')
    parser.add_argument("--i", default=(r"C:\Users\dylin\Documents\BA_Glare_Effect\Original_colors_screenshots"), \
        help='The directory the glare effect screenshots are stored in.')
    parser.add_argument("--c", default=(r"C:\Users\dylin\Documents\BA_Glare_Effect\colors_of_cards_original.txt"), \
        help='The file containing the original colors of the crads in the screenshots.')

    args = parser.parse_args()
    image_dir = args.i
    colors_path = args.c

    # Mapping for glare effect and original colors.
    # Usually 0-based but increased by one.
    
    color_mapping = {'orange': 1, 'brown': 2, 'green': 3, 'dark green': 4, 
        'light green': 5, 'dark red': 6, 'red': 7}
    

    # Mapping for color blindness. (Markus)
    # Usually 0-based but increased by one.
    '''
    color_mapping = {'dark green': 1, 'brown': 2, 'red': 3, 'light green': 4, 
        'green': 5, 'orange': 6, 'dark red': 7}
    '''

    images = load_images(image_dir)
    original_colors = load_original_colors(colors_path)
    original_colors = original_colors.split(';\n')
    original_colors = [[(color, color_mapping[color]) for color in colors.split(',')] for colors in original_colors]
    for colors_idx in range(len(original_colors)):
        for color_idx in range(len(original_colors[colors_idx])):
            current = original_colors[colors_idx][color_idx]
            original_colors[colors_idx][color_idx] = (current[0], current[1], color_idx)

    print(len(images), len(original_colors))
    if len(images) != len(original_colors): exit()
    

    matrices = []
    for i in range(len(images)):
        image = images[i]
        original_colors_image = original_colors[i]
        rgb_values = determine_glare_rgb_values(image)
        matrix = create_similarity_matrix(rgb_values, original_colors_image)
        matrices.append(matrix)
    
    similarity_matrix = create_similarity_matrix_average(matrices)
    print(similarity_matrix)

    # TODO: Problem: Some numbers are way over 100. So 100 is not right for normalizing. 
    # Find out maximum difference!

    '''
    Similarity matrix for original colors, original delta e values: (0 means colors are the same)
    (One screenshot)
    1.1=0.0,1.2=45.88020192130337,1.3=87.63889833094898,1.4=90.70742471085778,
    1.5=113.30992788739906,1.6=51.862180706594536,1.7=40.33367951509932,2.2=0.0,
    2.3=66.32289607228708,2.4=51.691944268794856,2.5=113.24204675286033,
    2.6=29.574531519406307,2.7=62.89410952235542,3.3=0.0,3.4=40.62647540540791,
    3.5=59.14027694122307,3.6=94.12780926821966,3.7=121.33131219183188,4.4=0.0,
    4.5=99.76072401547708,4.6=73.55644926181418,4.7=113.16585058997806,5.5=0.0,
    5.6=140.80037175459495,5.7=153.251618780184,6.6=0.0,6.7=46.9793450950003,7.7=0.0,

    Similarity matrix for original colors, values with 1 - (value / 100): (1 means colors are the same)
    (One screenshot)
    1.1=1.0,1.2=0.5411979807869663,1.3=0.12361101669051022,1.4=0.09292575289142224,
    1.5=-0.13309927887399065,1.6=0.4813781929340546,1.7=0.5966632048490068,
    2.2=1.0,2.3=0.33677103927712926,2.4=0.48308055731205146,2.5=-0.13242046752860337,
    2.6=0.704254684805937,2.7=0.37105890477644576,3.3=1.0,3.4=0.5937352459459209,
    3.5=0.4085972305877693,3.6=0.05872190731780336,3.7=-0.21331312191831886,4.4=1.0,
    4.5=0.002392759845229242,4.6=0.26443550738185817,4.7=-0.13165850589978056,5.5=1.0,
    5.6=-0.40800371754594944,5.7=-0.5325161878018401,6.6=1.0,6.7=0.530206549049997,7.7=1.0,

    Similarity matrix for color blindness, original delta e values: (0 means colors are the same)
    (One screenshot)
    1.1=0.0,1.2=26.582540732771232,1.3=47.52212704528696,1.4=33.18960760526932,
    1.5=18.493888873556916,1.6=36.52304106753959,1.7=23.78652925062888,2.2=0.0,
    2.3=21.5101187827116,2.4=25.508635800441034,2.5=26.256681225008673,
    2.6=13.356568589430752,2.7=17.377098502448394,3.3=0.0,3.4=38.17523290865369,
    3.5=46.38741668336888,3.6=17.86706005934495,3.7=31.329656953512146,4.4=0.0,
    4.5=19.97096151002481,4.6=20.315614972133346,4.7=40.253326090901076,5.5=0.0,
    5.6=30.893470743450504,5.7=34.08022166235096,6.6=0.0,6.7=29.98784173181083,7.7=0.0,

    Similarity matrix for color blindness, values with 1 - (value / 100): (1 means colors are the same)
    (One screenshot)
    1.1=1.0,1.2=0.7341745926722877,1.3=0.5247787295471305,1.4=0.6681039239473068,
    1.5=0.8150611112644308,1.6=0.6347695893246041,1.7=0.7621347074937111,
    2.2=1.0,2.3=0.7848988121728839,2.4=0.7449136419955897,2.5=0.7374331877499133,
    2.6=0.8664343141056925,2.7=0.826229014975516,3.3=1.0,3.4=0.6182476709134631,
    3.5=0.5361258331663112,3.6=0.8213293994065505,3.7=0.6867034304648785,4.4=1.0,
    4.5=0.8002903848997519,4.6=0.7968438502786666,4.7=0.5974667390909892,5.5=1.0,
    5.6=0.691065292565495,5.7=0.6591977833764904,6.6=1.0,6.7=0.7001215826818917,7.7=1.0,

    Similarity matrix for glare effect, original delta e values: (0 means colors are the same)
    (100 screenshots)
    1.1=1.0655059768862616,1.2=2.1033477172793793,1.3=2.40607096326105,
    1.4=2.3454120380778942,1.5=3.230645943244361,1.6=2.4034607167961646,
    1.7=2.771497109247035,2.2=1.5573933590440925,2.3=2.644273667937867,
    2.4=1.8615846342661035,2.5=3.8369900046563066,2.6=1.8271657068017348,
    2.7=2.647358432447738,3.3=1.6033740153195593,3.4=2.1642063387102133,
    3.5=2.682343819608548,3.6=3.0107563575329506,3.7=3.8462754739981357,
    4.4=0.909767473312609,4.5=3.587028996948641,4.6=1.8080276834470603,
    4.7=2.932199524408625,5.5=1.029177494318413,5.6=4.438560257586556,
    5.7=5.044602030509317,6.6=0.8481936793805458,6.7=1.9579847432371604,
    7.7=1.4301216683858966,

    Similarity matrix for glare effect, values with 1 - (value / 100): (1 means colors are the same)
    (100 screenshots)
    1.1=0.9893449402311373,1.2=0.9789665228272064,1.3=0.9759392903673892,
    1.4=0.9765458796192212,1.5=0.9676935405675562,1.6=0.9759653928320382,
    1.7=0.9722850289075297,2.2=0.9844260664095592,2.3=0.9735572633206213,
    2.4=0.9813841536573392,2.5=0.961630099953437,2.6=0.9817283429319827,
    2.7=0.9735264156755226,3.3=0.9839662598468043,3.4=0.9783579366128979,
    3.5=0.9731765618039145,3.6=0.9698924364246703,3.7=0.9615372452600187,
    4.4=0.9909023252668738,4.5=0.9641297100305134,4.6=0.9819197231655292,
    4.7=0.9706780047559138,5.5=0.989708225056816,5.6=0.9556143974241342,
    5.7=0.9495539796949067,6.6=0.9915180632061946,6.7=0.9804201525676285,
    7.7=0.985698783316141,
    '''
