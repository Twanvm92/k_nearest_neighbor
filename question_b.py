# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:09:20 2021

@author: 20172458
"""

import numpy as np
import pandas as pd
from tqdm import tqdm #Loadingbar
from functions_part_one import *

train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data

cross_validation_score = [] 
for k in range(1,2): # Takes 20min per iteration of k
    
    loss = 0
    for image_index in tqdm(range(len(train_images))):
        
        train_images_loocv = np.delete(train_images, image_index) # Leave one out, images
        train_labels_loocv = np.delete(train_labels, image_index) # Leave one out, indeces
        
        prediction = predict_digits(k, train_images_loocv, train_labels_loocv, 
                                    train_images[image_index], euclidean_distance)
        
        if prediction != train_labels[image_index]:
            loss += 1
            
    cross_validation_score.append(loss / len(train_images))


df_results_qb = pd.DataFrame(data = {'k': list(range(1,21)), 'CVS training data': cross_validation_score})
df_results_qb_.to_csv("results_q1b", index = False)