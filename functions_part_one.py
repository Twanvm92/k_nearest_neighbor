# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:34:03 2021

@author: 20172458
"""
import time
from collections import defaultdict
from operator import itemgetter
from sklearn.neighbors import BallTree

def euclidean_distance(img_1, img_2):
    '''
    Calculate the euclidean distance
    '''
    return (sum((img_1 - img_2) ** 2)) ** (1/2)

def minkowski_distance(img_1, img_2, p): # Question C
    ''' 
    Calculate the minkowski distance
    '''
    return (sum((abs(img_1 - img_2)) ** p)) ** (1 / p)

def cosine_distance(img_1, img_2):
    '''
    Calculate the cosine distance
    '''
    return 1 - sum(img_1 * img_2) / (((sum(img_1 ** 2)) ** (1/2)) * ((sum(img_2 ** 2)) ** (1/2)))

def find_majority_class(labels):
    '''
    Finding the majority class from a set of labels
    '''
    label_freq_n_diff_pairs = defaultdict(list)
    for diff, label in labels:
        if label not in label_freq_n_diff_pairs:
            label_freq_n_diff_pairs[label] = [0, 0]

        label_freq_n_diff_pairs[label][0] += 1 # Frequency
        label_freq_n_diff_pairs[label][1] += diff # Total difference

    max_freq = max(label_freq_n_diff_pairs.values(), key=itemgetter(0))[0] # maximum frequency
    # get list of all labels that equally occur most frequently.
    max_keys = [label for label, freq_diff_list in label_freq_n_diff_pairs.items() if freq_diff_list[0] == max_freq]
    
    if len(max_keys) == 1:
        return(max_keys[0])
    else:
        # dict key (label), value (list of freq, diff) pairs filtered by max freq labels
        filt_freq_diff_pairs = filter(lambda elem: elem[1][0] == max_freq, label_freq_n_diff_pairs.items())
        # take dict key, value pair with smallest diff amongst max freq labels
        filt_key_val =  min(filt_freq_diff_pairs, key=lambda x: x[1][1]) 
        # take label
        label_sml_diff = filt_key_val[0]
        # label_sml_diff_old = min(sum_diff, key = sum_diff.get)

        return label_sml_diff # Return the label with smallest difference
                               

def predict_digits(k, train_images, train_labels, test_image, p=2, distance_m=None):
    '''
    Predicting the label of a handwritten digit
    '''
    # Calculate distances
    distances = list()
    # start_dist = time.time()
    for (image, label) in zip(train_images, train_labels):
        # determine distance metric
        if distance_m == None:
            distance_result = minkowski_distance(image, test_image, p)
        else:
            # call the _distance() function prefixed with passed argument distance_m
            distance_result = globals()[f'{distance_m}_distance'](image, test_image)
        distances.append((distance_result, label))
    
    # Sorted distances
    distances.sort(key=lambda x: x[0])
    # Extract k labels
    k_labels = distances[:k]
    
    return find_majority_class(k_labels)
        
def predict_digits_ball(k, train_images, train_labels, test_image, p=2, distance_m=None):
    '''
    Predicting the label of a handwritten digit
    '''
    
    # Calculate distances
    distances = list()
    # start_dist = time.time()
    for (image, label) in zip(train_images, train_labels):
        # determine distance metric
        if distance_m == None:
            distance_result = minkowski_distance(image, test_image, p)
        else:
            # call the _distance() function prefixed with passed argument distance_m
            distance_result = globals()[f'{distance_m}_distance'](image, test_image)
        distances.append((distance_result, label))
    
    # Sorted distances
    distances.sort(key=lambda x: x[0])
    # Extract k labels
    k_labels = distances[:k]
    
    return find_majority_class(k_labels)