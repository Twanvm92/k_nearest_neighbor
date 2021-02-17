# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:34:03 2021

@author: 20172458
"""

import numpy as np
import pandas as pd
from tqdm import tqdm #Loadingbar
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

train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
test_data = np.genfromtxt("MNIST_test_small.csv", delimiter=',')
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data

empirical_test_loss = []; empirical_train_loss = []
for k in range(1,2): # Takes 6min per iteration of k
    
    i_test = 0; loss_test = 0
    i_train = 0; loss_train = 0
    
    for test_image in tqdm(test_images):
        prediction_test = predict_digits(k, train_images, train_labels, test_image)
        
        if prediction_test != test_labels[i_test]:
            loss_test += 1
    
        i_test += 1
        
    for train_image in train_images:
        prediction_train = predict_digits(k, train_images, train_labels, train_image)
        
        if prediction_train != train_labels[i_train]:
            loss_train += 1
        
        i_train += 1
    
    empirical_test_loss.append(loss_test / len(test_images))
    empirical_train_loss.append(loss_train / len(train_images))
    

df_results_qa = pd.DataFrame(data = {'k': list(range(1,2)), 'Empirical Training Loss': empirical_train_loss,
                              'Empirical Test Loss': empirical_test_loss})
df_results_qa.to_csv("results_q1a", index = False)

        