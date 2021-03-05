import numpy as np
import pandas as pd
from tqdm import tqdm #Loadingbar
from functions_part_one import *
import sys, getopt
from sklearn.model_selection import LeaveOneOut, KFold
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.neighbors import BallTree, DistanceMetric, KDTree
from scipy.ndimage import interpolation
import time
from collections import defaultdict
from operator import itemgetter

############################## Deskewed images #########################################
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

def deskew_images(images):
    deskewed_images = np.zeros(shape=(len(images), 28*28))
    for x in range(len(images)):
        image = images[x].reshape(28,28)
        deskew_image = deskew(image)
        deskew_image = np.reshape(deskew_image, 784)
        for i in range(len(deskew_image)):
            if deskew_image[i] < 0: # If the value of the pixel is higher than the threshold
                deskew_image[i] = 0  # Replace the corresponding pixel in the new image by 1

        deskewed_images[x] = deskew_image      
    return deskewed_images

############################## Blurred images #########################################  
def blur_images(images):
    """ This function blurs the images in a given set of images"""
    pix_on_axis = int(len(images[0])**(0.5)) # Number of pixels on each axis of the images
    blur_images = np.zeros(shape=(len(images), int((0.5*pix_on_axis)**2))) # Empty array for the blurred images
    for x in range(len(images)):
        image = images[x].reshape(pix_on_axis, pix_on_axis) # Transform the array to a matrix
        new_image = []
        for i in range(int(0.5*pix_on_axis)): 
            for j in range(int(0.5*pix_on_axis)):
                # Take the average of four pixels in a square of 2x2 from topleft corner to bottomright corner of the image
                new_image.append((image[2*i][2*j] + image[2*i+1][2*j] +  image[2*i][2*j+1] +  image[2*i+1][2*j+1])/4)    
        blur_images[x] = new_image   
    return blur_images

print(1)
# train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
train_data = np.genfromtxt("MNIST_train.csv", delimiter=',') # Import the large training data

print(2)
# smaller_split = 30000
# train_labels, train_images = train_data[:smaller_split,0], train_data[:smaller_split, 1:] # Splitting training data
# train_images = blur_images(train_images) # Blur the images once
# train_images = blur_images(train_images) # Blur the images again
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
train_images = deskew_images(train_images)
train_images = blur_images(train_images) # Blur the images once
#train_images = blur_images(train_images) # Blur the images once
print(3)
# test_data = np.genfromtxt("MNIST_test_small.csv", delimiter=',')
test_data = np.genfromtxt("MNIST_test.csv", delimiter=',')
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data
test_images = deskew_images(test_images)
test_images = blur_images(test_images) # Blur the images once
#test_images = blur_images(test_images) # Blur the images once
print(4)

def question_f(s_train_images, s_test_images, s_train_labels, s_test_labels, k):
    p = 8
    loss = 0
    print(f"for k: {k} and p: {p}")

    #kf = KFold(n_splits=10)
    #for train_index, test_index in kf.split(test_images):
    #    s_train_images = test_images[train_index]
     #   s_test_images = test_images[test_index]
    #    s_train_labels = test_labels[train_index]
    #    s_test_labels = test_labels[test_index]

        # tree = BallTree(s_train_images, leaf_size=400, metric='minkowski', p=p)
    tree = KDTree(s_train_images, leaf_size=400,  metric='minkowski', p=p) 
    for t_image_index in tqdm(range(len(s_test_images))): # 1/10 of one k
        dist, ind = tree.query(s_test_images[t_image_index:t_image_index+1], k=k)
        dist_labels = list()
        dist_index = 0
        for i in ind[0]:
            dist_labels.append((dist[0][dist_index], s_train_labels[i]))
            dist_index += 1
            
            # print("dist labels list with tuples: ")
            # print(dist_labels)
        prediction = find_majority_class(dist_labels)

        if prediction != s_test_labels[t_image_index]:
            loss += 1

    cvs = loss / len(test_images)
    print(cvs)
    
question_f(train_images, test_images, train_labels, test_labels, 4)