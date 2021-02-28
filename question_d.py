import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #Loadingbar
from scipy.ndimage import interpolation
from functions_part_one import *

def calculate_loss(k, train, test):
    """ Function to calculate the loss for values for 1 to k """
    empirical_test_loss = []
    for k in range(1,k+1): 
        i_test = 0; loss_test = 0
        for test_image in tqdm(test):
            prediction = predict_digits(k, train, train_labels, test_image) # Makes a prediction for the given k and training set
            if prediction != test_labels[i_test]: # If the prediction is not correct, add 1 to the loss
                loss_test += 1
            i_test += 1
        loss = loss_test / len(test)
        print("The loss for k = {} is {}".format(k, loss))

train_data = np.genfromtxt(r"C:\Users\daanv\Documents\DSiE 1\Q3\2DI70 Statistical Learning Theory\Assignment 1\k_nearest_neighbor\MNIST_train_small.csv", delimiter=',') # Import the small training data
test_data = np.genfromtxt(r"C:\Users\daanv\Documents\DSiE 1\Q3\2DI70 Statistical Learning Theory\Assignment 1\k_nearest_neighbor\MNIST_test_small.csv", delimiter=',') # Import the small test data
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data

############################## Binary images #########################################
def binary_images(images, threshold):
    """This function replaces the values of the pixels by 1 or 0 depending on the threshold"""
    binary_images = np.zeros(shape=(len(images), len(images[0]))) # Empty array for the binary images
    for x in range(len(images)):
        image = images[x]
        new_image = np.zeros(shape=len(image)) # Set all values of the pixels of the new image to 0
        for i in range(len(image)):
            if image[i] > threshold: # If the value of the pixel is higher than the threshold
                new_image[i] = 1  # Replace the corresponding pixel in the new image by 1
        binary_images[x] = new_image
    return binary_images

binary_train_images = binary_images(train_images, 50)
binary_test_images = binary_images(test_images, 50)
#calculate_loss(20, binary_train_images, binary_test_images) # Determine the loss for k = [1,20] for the binary images       

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

blur_train_images = blur_images(train_images) # Blur the images once
blur_test_images = blur_images(test_images)

blur_train_images_2 = blur_images(blur_train_images) # Blur the images even more
blur_test_images_2 = blur_images(blur_test_images)

#calculate_loss(20, blur_train_images, blur_test_images) # Determine the loss for k = [1,20] for the blurred images
#calculate_loss(20, blur_train_images_2, blur_test_images_2) #Determine the loss for k = [1,20] for the twice blurred images

# If you want to see the blurred image:
    
#plt.imshow(blur_train_images_2[1].reshape(7,7), cmap='hot', interpolation='nearest')
#plt.show()
#plt.imshow(blur_train_images[1].reshape(14,14), cmap='hot', interpolation='nearest')
#plt.show()

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
        image = train_images[x].reshape(28,28)
        #plt.imshow(image, cmap='hot', interpolation='nearest')
        #plt.show()
        deskew_image = deskew(image)
        #plt.imshow(deskew_image, cmap='hot', interpolation='nearest')
        #plt.show()
        deskew_image = deskew_image.reshape(1, 784)
        deskewed_images[x] = deskew_image
    return deskewed_images

deskew_train_images = deskew_images(train_images) # Deskew the images
deskew_test_images = deskew_images(test_images)

#calculate_loss(20, deskew_train_images, deskew_test_images) #Determine the loss for k = [1,20] for the deskewed images

############################## Combinations #########################################  

# Deskew and blur 
blur_deskew_train_images = blur_images(deskew_train_images)
blur_deskew_test_images = blur_images(deskew_test_images)
#calculate_loss(20, blur_deskew_train_images, blur_deskew_test_images) #Determine the loss for k = [1,20] for the deskewed, blurred images

# Deskew and double blur
double_blur_deskew_train_images = blur_images(blur_deskew_train_images)
double_blur_deskew_test_images = blur_images(blur_deskew_test_images)
#calculate_loss(20, double_blur_deskew_train_images, double_blur_deskew_test_images) #Determine the loss for k = [1,20] for the deskewed, double-blurred images

# Deskew and binary
binary_deskew_train_images = binary_images(deskew_train_images, 100)
binary_deskew_test_images = binary_images(deskew_test_images, 100)
#calculate_loss(20, binary_deskew_train_images, binary_deskew_test_images) #Determine the loss for k = [1,20] for the deskewed, binary images
