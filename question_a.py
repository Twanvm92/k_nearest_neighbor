# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:43:05 2021

@author: 20172458
"""

import numpy as np
import pandas as pd
from tqdm import tqdm #Loadingbar
from functions_part_one import *

train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
test_data = np.genfromtxt("MNIST_test_small.csv", delimiter=',')
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data

empirical_test_loss = []; empirical_train_loss = []
max_k = 20
min_k = 1
for k in range(min_k,max_k + 1): # Takes 6min per iteration of k
    
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
    
    emp_test_l = loss_test / len(test_images)
    emp_train_l = loss_train / len(train_images)
    empirical_test_loss.append(emp_test_l)
    empirical_train_loss.append(emp_train_l)
    

df_results_qa = pd.DataFrame(data = {'k': list(range(min_k,max_k+1)), 'Empirical Training Loss': empirical_train_loss,
                              'Empirical Test Loss': empirical_test_loss})
df_results_qa.to_csv("results_q1a_new_tie_breaker_fixed", index = False)



