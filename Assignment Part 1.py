# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:34:03 2021

@author: 20172458
"""

import numpy as np    

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
    count_dict = {}
    # Initiate dictionary
    for i in range(10):
        count_dict[i] = 0
    
    for label in labels:
        count_dict[label] += 1
    
    max_value = max(count_dict.values())  # maximum value
    max_keys = [k for k, v in count_dict.items() if v == max_value]
    
    return sorted(max_keys)[0] # Return the class with lowest label number or we
                               # Or we can think of a rule that takes into account similar digits
                               

def predict_digits(k, train_images, train_labels, test_image):
    
    # Calculate distances
    distances = list()
    for (image, label) in zip(train_images, train_labels):
        distances.append((euclidean_distance(image, test_image), label))

    # Sorted distances
    sorted_distances = sorted(distances, key=lambda x: x[0])
    
    # Extract k labels
    k_labels = [x[1] for x in sorted_distances[:k]]
    
    return find_majority_class(k_labels)

train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
test_data = np.genfromtxt("MNIST_test_small.csv", delimiter=',')
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data

for k in range(1,21):
    
    i = 0; loss = 0
    for test_image in test_images:
        
        prediction = predict_digits(k, train_images, train_labels, test_image)
        if prediction != test_labels[i]:
            loss += 1
        
        i += 1
    
    empirical_loss = loss / len(test_images)
    print("k: {} - {}".format(k, empirical_loss))

        
        
        