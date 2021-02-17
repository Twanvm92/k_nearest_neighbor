# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:34:03 2021

@author: 20172458
"""

from collections import defaultdict

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
    return sum(img_1 * img_2) / (((sum(img_1 ** 2)) ** (1/2)) * ((sum(img_2 ** 2)) ** (1/2)))

def find_majority_class(labels):
    '''
    Finding the majority class from a set of labels
    '''
    count_freq = defaultdict(int); sum_diff = defaultdict(float)
    for diff, label in labels:
        count_freq[label] += 1 # Frequency
        sum_diff[label] += diff # Total difference
    
    max_freq = max(count_freq.values()) # maximum frequency
    max_keys = [label for label, freq in count_freq.items() if freq == max_freq]
    
    if len(max_keys) == 1:
        return(max_keys[0])
    else:
        return(min(sum_diff, key = sum_diff.get)) # Return the label with smallest difference
                               

def predict_digits(k, train_images, train_labels, test_image):
    '''
    Predicting the label of a handwritten digit
    '''
    # Calculate distances
    distances = list()
    for (image, label) in zip(train_images, train_labels):
        distances.append((euclidean_distance(image, test_image), label))

    # Sorted distances
    distances.sort(key=lambda x: x[0])
    
    # Extract k labels
    k_labels = distances[:k]
    
    return find_majority_class(k_labels)
        