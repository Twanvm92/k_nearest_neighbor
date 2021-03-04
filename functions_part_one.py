# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:34:03 2021

@author: 20172458
"""
import time
from collections import defaultdict
from operator import itemgetter

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
    # start_find =  time.time()
    # count_freq = defaultdict(int)
    # sum_diff = defaultdict(float)
    label_freq_n_diff_pairs = defaultdict(list)
    for diff, label in labels:
        if label not in label_freq_n_diff_pairs:
            label_freq_n_diff_pairs[label] = [0, 0]

        label_freq_n_diff_pairs[label][0] += 1 # Frequency
        label_freq_n_diff_pairs[label][1] += diff # Total difference

        # sum_diff[label] += diff # Total difference
        # count_freq[label] += 1

    max_freq = max(label_freq_n_diff_pairs.values(), key=itemgetter(0))[0] # maximum frequency
    # max_freq_old = max(count_freq.values())
    # get list of all labels that equally occur most frequently.
    max_keys = [label for label, freq_diff_list in label_freq_n_diff_pairs.items() if freq_diff_list[0] == max_freq]
    # max_keys_old = [label for label, freq in count_freq.items() if freq == max_freq]


    
    # end_find =  time.time()
    # print("Finding majority: {}".format(end_find - start_find))
    if len(max_keys) == 1:
        # print('only one max key:')
        # print(f'old max key (label): {max_keys_old[0]}')
        # print(f'max key (label): {max_keys[0]}')

        return(max_keys[0])
    else:
        # dict key (label), value (list of freq, diff) pairs filtered by max freq labels
        filt_freq_diff_pairs = filter(lambda elem: elem[1][0] == max_freq, label_freq_n_diff_pairs.items())
        # take dict key, value pair with smallest diff amongst max freq labels
        filt_key_val =  min(filt_freq_diff_pairs, key=lambda x: x[1][1]) 
        # take label
        label_sml_diff = filt_key_val[0]
        # label_sml_diff_old = min(sum_diff, key = sum_diff.get)





        # print(f'labels and diff pairs for k nearest neighbors (distance, label): {labels}')
        # print(f'old label freq (key=label, data=count): {count_freq}')
        # print(f'old label summ_diff (key=label, data=sum of diff): {sum_diff}')
        # print(f'label freq and diff pair (key=label, data=(count, sumdiff)): {label_freq_n_diff_pairs}')
        # print(f'old max freq (count): {max_freq_old}')
        # print(f'max freq (count): {max_freq}')
        # print(f'old max keys list (label): {max_keys_old}')
        # print(f'max keys list (label): {max_keys}')
        # print(f'filtered label freq_diff pairs with max_freq: {max_freq}')
        # print(f'in following data structure: {filt_freq_diff_pairs}')
        # print(f'pair with minimum diff of those pairs: {filt_key_val}')
        # print('multiple max keys:')
        # print(f'old label with smallest diff sum (label): {label_sml_diff_old}')
        # print(f'label with smallest diff sum (label): {label_sml_diff}')
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
        # start_dist_one =  time.time()
        distances.append((distance_result, label))
        # end_dist_one =  time.time()
        # print("Calculating one distance: {}".format(end_dist_one - start_dist_one))
    
    # end_dist = time.time()
    # print("Calculating distances: {}".format(end_dist - start_dist))
    # start_sort = time.time()
    # Sorted distances
    distances.sort(key=lambda x: x[0])
    # end_sort =  time.time()
    # print("Sorting distances: {}".format(end_sort - start_sort))
    # Extract k labels
    k_labels = distances[:k]
    
    return find_majority_class(k_labels)
        