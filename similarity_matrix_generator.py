#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script is used to create a similarity matrix
# out of screenshots taken in the memory game app.
# Watch out to use the correct mapping of colors. 

import os
import argparse
from PIL import Image 
import numpy as np
import itertools
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000
from colormath.color_conversions import convert_color
import scipy

# TODO: Add description of return values.

# 150 x 150 pixels are taken for each card.
# The tuple value represents the koordinates of the top left corner for each card. 
# cards are about 250 x 250 and have 30 pixels between them, which results in 
# 280 pixels on a axis until the next corner. The values were manually checked. 
# Worls only if the resoulution of the screenshots is 1920 x 1080. 
# In the Emulator a Pixel 2 was used to play the game and create the screenshots.
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

# Loading the screenshots from which the rgb values will be extracted.
def load_images(image_dir):
    '''
    :param image_dir: The full path to the directory the images are stored in. 
    '''
    images = []
    for r, _, f in os.walk(image_dir):
        for file_name in f:
            if '.PNG' or '.png' in file_name:
                full_path = os.path.join(r, file_name)
                image = Image.open(full_path)
                images.append(image)
    return images 

# Calculates the rgb average rbg value for each 
# card in an image.
def determine_glare_rgb_values(image):
    '''
    :param image: The screenshot of the memory game with all cards turned.
    '''
    glare_rgb_values = []
    for corner in card_corners:
        # With the corner koordinates these borders 
        # create the are in which to extract all pixel values.
        x_border = corner[0] + 150
        y_border = corner[1] + 150
        card_values = []
        for x in range(corner[0], x_border, 1):
            for y in range(corner[1], y_border, 1):
                coordinates = x, y
                pixel_values = image.getpixel(coordinates)
                # Ignoring alpha values. They are always 255 on the screenshots. 
                card_values.append(pixel_values[:-1])
        # Calculating the average r, g and b values in the are of the card. 
        card_r = int(round(np.mean([color[0] for color in card_values])))
        card_g = int(round(np.mean([color[1] for color in card_values])))
        card_b = int(round(np.mean([color[2] for color in card_values])))
        glare_rgb_values.append((card_r, card_g, card_b))
    return glare_rgb_values

# Reading the colour names on the screenshots from a text file.
def load_original_colors(colors_path):
    '''
    :param colors_path: The full path to the text file containing the 
    color names for all cards in all screenshots in the correct order.
    '''
    if os.path.exists(colors_path):
        # Reading the text in the text file. 
        with open(colors_path, "r") as colors_file:
            return colors_file.read()

# Determines dinstance between colors with delta e fomula. 
def determine_distance(color_1_rgb, color_2_rgb):
    '''
    :param color_1_rgb: The first color.
    :param color_2_rgb: The second color.
    '''
    # Converting the colors from the sRGB to the Lab color space.
    lab_1 = convert_color(color_1_rgb, LabColor)
    lab_2 = convert_color(color_2_rgb, LabColor)

    if cie_1976:
        # Old formula for color differneces. 
        detlta_e = delta_e_cie1976(lab_1, lab_2)
    else:
        # Updated formula that is supposed to picture color differences 
        # more accurately. 
        detlta_e = delta_e_cie2000(lab_1, lab_2, Kl=1, Kc=1, Kh=1)
    return detlta_e

# Returns a matrix for a screenshot.
def create_similarity_matrix(rgb_values, original_colors):
    '''
    :param rgb_values: The average rgb values for all the cards 
    on a screenshot of a glare effect memory game. 
    :param original_colors: A list containing tuples with the card name, 
    the mappimg number for the card, and the index for each card on the screenshot.
    '''
    combinations_for_calculation = []
    matrix_values = []
    # Iterating over all combinations of card colors on the screenshot.
    for i in range(len(rgb_values)):
        # The color name, the mapping and the index of the first card for comparison.
        original_color_1 = original_colors[i]
        # The rgb values for the first card for comparison from the glare effect screenshot.
        glare_rgb_1 = rgb_values[i]
        glare_rgb_1 = sRGBColor(glare_rgb_1[0], glare_rgb_1[1], glare_rgb_1[2], is_upscaled=True)

        for l in range(len(rgb_values)):
            # The color name, the mapping and the index of the second card. for comparison
            original_color_2 = original_colors[l]
            # The rgb values for the second card for comparison 
            # from the glare effect screenshot.
            glare_rgb_2 = rgb_values[l]
            glare_rgb_2 = sRGBColor(glare_rgb_2[0], glare_rgb_2[1], glare_rgb_2[2], is_upscaled=True)

            # Only adding the values, if the cards compared are not not same card 
            # and the same comparison with switched values and positions
            # has not been included yet.
            if original_color_1[2] != original_color_2[2] and \
                    (original_color_2, original_color_1) not in combinations_for_calculation:
                # Adding the combination to the alredy included combinations.
                combinations_for_calculation.append((original_color_1, original_color_2))
                # The mapping of both colors in order to determine the place in the 
                # similarity matrix.
                position = (original_color_1[1], original_color_2[1])
                # Adding the the distance between the two colors and the position in
                # similarity matrix to the list of values for the matrix.
                matrix_values.append((determine_distance(glare_rgb_1, glare_rgb_2), position))
    
    # The entries in the similarity matrix
    values = {'1.1': [], '1.2': [], '1.3': [], '1.4': [], '1.5': [], '1.6': [], \
        '1.7': [], '2.2': [], '2.3': [], '2.4': [], '2.5': [], '2.6': [], '2.7': [], \
        '3.3': [], '3.4': [], '3.5': [], '3.6': [], '3.7': [], '4.4': [], '4.5': [], \
        '4.6': [], '4.7': [], '5.5': [], '5.6': [], '5.7': [], '6.6': [], '6.7': [], \
        '7.7': []}

    # Adding each distance value in the corresponding list of values 
    # for the similarity matrix.
    for value in matrix_values:
        position = value[1]
        if position[0] < position[1]:
            matrix_pos = '%s.%s' %(position[0], position[1])
        else:
            matrix_pos = '%s.%s' %(position[1], position[0])
        values[matrix_pos].append(value[0])

    # Calculating the average for each color combination.
    # Note: For entires comparing the same color there is only one value 
    # (comparing the two cards) and for entries comparing different colors 
    # there are four values (2 * 2).
    values = {key: np.mean(values[key]) for key in values.keys()}
    # The maximum distance in this similarity matrix.
    max_distance_matrix = max(values.values())
    global max_distance
    # Updating the overall maximum color distance. 
    if max_distance_matrix > max_distance: max_distance = max_distance_matrix
    return values

# Determines average of all matrices. 
def create_similarity_matrix_average(matrices_list, downscale):
    '''
    :param matrices_list: A list containing all determined 
    similarity matrices for the screenshots.
    :param downscale: Wether to downscale the data to 
    values between 0 and 1.
    '''
    # Exiting if there are no matrices. 
    if not matrices_list:
        exit()
    
    similarity_matrix = ''
    for key in matrices_list[0].keys():
        # All values for a specific color combination.
        cell_values = []
        for matrix in matrices_list:
            cell_values.append(matrix[key])
        # Calcualting the average and downscaling the values if desired.
        if downscale:    
            similarity_matrix += '%s=%s,' %(key, (np.mean(cell_values)) / max_distance)
        else: 
            similarity_matrix += '%s=%s,' %(key, np.mean(cell_values))
    return similarity_matrix

# For calculating how many of the combinations are includes in the calcuation. 
def determine_coverage(original_colors):
    '''
    :param original_colors: A list containing tuples with the card name, 
    the mappimg number for the card, and the index for each card on the 
    screenshot for all screenshots.
    '''
    # The calculations are based on having 14 cards on the field.
    # The number of position combinations for the same colors.
    number_of_combinations_same_colors = int(7 * (14 * 13 / 2))
    # The number of position combinations for different colors.
    number_of_combinations_different_colors = 21 * 14 * 13
    # Total number of combinations.
    number_of_total_combinations = number_of_combinations_same_colors \
        + number_of_combinations_different_colors

    # All the combinations without duplicates.
    combinations = []
    # Iterating over all combinations of cards of all screenshots.
    for colors in original_colors:
        for i in range(len(colors)):
            # First color for combination.
            original_color_1 = colors[i]
            for l in range(len(colors)):
                # Second color for combination.
                original_color_2 = colors[l]
                # Only adding the values, if the cards compared are not not same card 
                # and the same comparison or a comparison with switched values 
                # and positions has not been included yet.
                if original_color_1[2] != original_color_2[2] and \
                        (original_color_2, original_color_1) not in combinations and \
                        (original_color_1, original_color_2) not in combinations:
                    # Adding the combination to the alredy included combinations.
                    combinations.append((original_color_1, original_color_2))    
    # The percentage of coverage.  
    return len(combinations) / number_of_total_combinations


if __name__ == '__main__':

    # Argument handling.
    parser = argparse.ArgumentParser(description='Used to create a static similarity matrix \
        for the simulating games with the glare effect.')
    parser.add_argument("--i", default=(r"C:\Users\dylin\Documents\BA_Glare_Effect\screenshots_glare_effect"), \
        help='The directory the glare effect screenshots are stored in.')
    parser.add_argument("--c", default=(r"C:\Users\dylin\Documents\BA_Glare_Effect\color_names\colors_of_cards_glare_effect.txt"), \
        help='The file containing the original colors of the crads on the screenshots.')
    parser.add_argument('--color_blindness_mapping', action='store_true', \
        help='Pass if the mapping for color blindness should be used.')
    parser.add_argument('--cie1976', action='store_true', \
        help='Pass if the cie1976 formular for color difference should be used. Otherwise cie2000 will be.')
    parser.add_argument('--unscaled', action='store_true', \
        help='Pass if the similarity matrix should be normalized.')

    # Assigning parameters. 
    args = parser.parse_args()
    image_dir = args.i
    colors_path = args.c
    color_blindness_mapping = args.color_blindness_mapping
    global cie_1976
    cie_1976 = args.cie1976
    downscale = not args.unscaled

    # The maximum distance between all colors over all matrices.
    # Will be updated when calculating the matrices for each screenshot
    # and used for scaling the distances down to be between 0 and 1.
    global max_distance
    max_distance = 0

    # Choosing the mapping depending on the use case. 
    # These mappings are extracted from the memory game app.
    if not color_blindness_mapping:
        # Mapping for glare effect an the originl colors.
        color_mapping = {'orange': 1, 'brown': 2, 'green': 3, 'dark green': 4, 
            'light green': 5, 'dark red': 6, 'red': 7}
    else:
        # Mapping for color blindness. 
        color_mapping = {'dark green': 1, 'brown': 2, 'red': 3, 'light green': 4, 
            'green': 5, 'orange': 6, 'dark red': 7}

    # Loading the screenshots. 
    images = load_images(image_dir)
    # Loading the original colors.
    original_colors = load_original_colors(colors_path)

    # A list of a strings containing color names, one string for each screenshot.
    original_colors = original_colors.split(';\n')
    # Splitting each string into the colors and combining it with 
    # the mapping number for the color.
    original_colors = [[(color, color_mapping[color]) for color in colors.split(',')] for colors in original_colors]
    # Adding the index of the card on the screenshot for
    # each color for each screenshot. 
    for colors_idx in range(len(original_colors)):
        for color_idx in range(len(original_colors[colors_idx])):
            current = original_colors[colors_idx][color_idx]
            original_colors[colors_idx][color_idx] = (current[0], current[1], color_idx)

    if len(images) == len(original_colors):
        print('Number of screenshots use: %s' %len(images))
    # Exiting if the number of images does not equal the number of color lists.
    else:
        print('The numbers of screenshots and original colors differs.')
        exit()

    coverage = determine_coverage(original_colors)
    print('Combination coverage: %s' %coverage)
    
    matrices = []
    # Determining a similarity matrix for each screenshot.
    for i in range(len(images)):
        image = images[i]
        # The color name, the mapping and the index for each card on a screenshot.
        original_colors_image = original_colors[i]
        rgb_values = determine_glare_rgb_values(image)
        matrix = create_similarity_matrix(rgb_values, original_colors_image)
        matrices.append(matrix)
    
    # Calculating the average 
    similarity_matrix = create_similarity_matrix_average(matrices, downscale)

    print('Done creating similarity matrix:')
    print(similarity_matrix)
    
    if downscale:
        print('Maximum distance used for scaling: %s' %max_distance)