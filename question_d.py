import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #Loadingbar
from functions_part_one import *

def calculate_loss(k, train, test):
    """ Function to calculate the loss for values for 1 to k """
    empirical_test_loss = []
    for k in range(1,k): # Takes 6min per iteration of k
        i_test = 0; loss_test = 0
        for test_image in tqdm(test):
            prediction = predict_digits(k, train, train_labels, test_image)
            if prediction != test_labels[i_test]:
                loss_test += 1
            i_test += 1
        loss = loss_test / len(test)
        print("The loss for k = {} is {}".format(k, loss))

train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
test_data = np.genfromtxt("MNIST_test_small.csv", delimiter=',')
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data

############################## Binary images #########################################
def binary_images(images):
    binary_images = np.zeros(shape=(len(images), len(images[0])))
    for x in range(len(images)):
        image = images[x]
        new_image = np.zeros(shape=len(image))
        for i in range(len(image)):
            if image[i] > 50:
                new_image[i] = 1  
        binary_images[x] = new_image
    return binary_images

binary_train_images = binary_images(train_images)
binary_test_images = binary_images(test_images)
calculate_loss(2, binary_train_images, binary_test_images)              

############################## Blurred images #########################################  
def blur_images(images):
    pix_on_axis = int(len(images[0])**(0.5))
    blur_images = np.zeros(shape=(len(images), int((0.5*pix_on_axis)**2)))
    for x in range(len(images)):
        
        # Reduce the number of pixels
        image = images[x].reshape(pix_on_axis, pix_on_axis)
        new_image = []
        for i in range(int(0.5*pix_on_axis)):
            for j in range(int(0.5*pix_on_axis)):
                new_image.append((image[2*i][2*j] + image[2*i+1][2*j] +  image[2*i][2*j+1] +  image[2*i+1][2*j+1])/4)       
        blur_images[x] = new_image
        
    return blur_images

blur_train_images = blur_images(train_images)
blur_train_images_2 = blur_images(blur_train_images)
blur_test_images = blur_images(test_images)
blur_test_images_2 = blur_images(blur_test_images)
#plt.imshow(blur_train_images_2[1].reshape(7,7), cmap='hot', interpolation='nearest')
#plt.show()
#plt.imshow(blur_train_images[1].reshape(14,14), cmap='hot', interpolation='nearest')
#plt.show()
calculate_loss(2, blur_train_images, blur_test_images)
calculate_loss(2, blur_train_images_2, blur_test_images_2)

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

deskew_train_images = np.zeros(shape=(len(train_images), 28*28))
for x in range(len(train_images)):
    image = train_images[x].reshape(28,28)
    #plt.imshow(image, cmap='hot', interpolation='nearest')
    #plt.show()
    deskew_image = deskew(image)
    #plt.imshow(deskew_image, cmap='hot', interpolation='nearest')
    #plt.show()
    deskew_image = deskew_image.reshape(1, 784)
    deskew_train_images[x] = deskew_image
    
deskew_test_images = np.zeros(shape=(len(test_images), 28*28))
for x in range(len(test_images)):
    image = test_images[x].reshape(28,28)
    #plt.imshow(image, cmap='hot', interpolation='nearest')
    #plt.show()
    deskew_image = deskew(image)
    #plt.imshow(deskew_image, cmap='hot', interpolation='nearest')
    #plt.show()
    deskew_image = deskew_image.reshape(1, 784)
    deskew_test_images[x] = deskew_image

calculate_loss(2, deskew_train_images, deskew_test_images)
