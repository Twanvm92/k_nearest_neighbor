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


# train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
train_data = np.genfromtxt("MNIST_train.csv", delimiter=',') # Import the large training data

# smaller_split = 30000
# train_labels, train_images = train_data[:smaller_split,0], train_data[:smaller_split, 1:] # Splitting training data
# train_images = blur_images(train_images) # Blur the images once
# train_images = blur_images(train_images) # Blur the images again
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
train_images = deskew_images(train_images)
train_images = blur_images(train_images) # Blur the images once
train_images = blur_images(train_images) # Blur the images once

# test_data = np.genfromtxt("MNIST_test_small.csv", delimiter=',')
test_data = np.genfromtxt("MNIST_test.csv", delimiter=',')
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data
test_images = deskew_images(test_images)
test_images = blur_images(test_images) # Blur the images once
test_images = blur_images(test_images) # Blur the images once

empirical_test_loss = []; empirical_train_loss = []
cross_validation_score = [] 


def question_a(k):
    

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


def question_b(k):

    print(f"for k: {k}")
    loss = 0
    for image_index in tqdm(range(len(train_images))):
        train_images_loocv = np.delete(train_images, image_index, axis=0) # Leave one out, images
        train_labels_loocv = np.delete(train_labels, image_index, axis=0) # Leave one out, indeces


        prediction = predict_digits(k, train_images_loocv, train_labels_loocv, 
                                    train_images[image_index])
        
        if prediction != train_labels[image_index]:
            loss += 1

    cross_validation_score.append(loss / len(train_images))

def question_c(k, max_p=15):
    # for each k loop through possible values for p in minkowski distance
    for p in range(1, max_p+1):
        print(f"for k: {k} and p: {p}")

        loss = 0
        for image_index in tqdm(range(len(train_images))):
            
            train_images_loocv = np.delete(train_images, image_index, axis=0) # Leave one out, images
            train_labels_loocv = np.delete(train_labels, image_index, axis=0) # Leave one out, indeces
            
            prediction = predict_digits(k, train_images_loocv, train_labels_loocv, 
                                        train_images[image_index], p=p)
            
            if prediction != train_labels[image_index]:
                loss += 1

        # just append all scores to one long list. Scores will get grouped later by k and p with MultiIndex on df.
        cross_validation_score.append(loss / len(train_images)) 

def question_f(k):
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
    tree = KDTree(train_images, leaf_size=400,  metric='minkowski', p=p) 
    for t_image_index in tqdm(range(len(test_images))): # 1/10 of one k
        dist, ind = tree.query(test_images[t_image_index:t_image_index+1], k=k)
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
    cross_validation_score.append(cvs)

def question_e(k):
    p = 8
    loss = 0
    print(f"for k: {k} and p: {p}")

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(train_images):
        s_train_images = train_images[train_index]
        s_test_images = train_images[test_index]
        s_train_labels = train_labels[train_index]
        s_test_labels = train_labels[test_index]

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

    emp_train_l = loss / len(train_images)
    empirical_train_loss.append(emp_train_l)

    # p = 8
    # loss = 0
    # print(f"for k: {k} and p: {p}")
        
    # for image_index in tqdm(range(len(train_images))):
    #     train_images_loocv = np.delete(train_images, image_index, axis=0) # Leave one out, images
    #     train_labels_loocv = np.delete(train_labels, image_index, axis=0) # Leave one out, indeces
        
    #     # tree = BallTree(train_images_loocv, leaf_size=3000, metric='minkowski', p=p)
    #     tree = KDTree(train_images_loocv, leaf_size=10000,  metric='minkowski', p=p)
    #     dist, ind = tree.query(train_images[image_index:image_index+1], k=k)
    #     dist_labels = list()
    #     dist_index = 0
    #     # print(f"ind array: ")
    #     # print(ind[0])
    #     # print(f'ind size: {ind[0].size} and shape: {np.shape(ind[0])}')
    #     # print(f"dist array: ")
    #     # print(dist[0])
    #     # print(f'dist size: {dist[0].size} and shape: {np.shape(dist[0])}')
    #     # print(f' train lables at index 0: {train_labels[0]}')
    #     for i in ind[0]:
    #         dist_labels.append((dist[0][dist_index],train_labels_loocv[i]))
    #         dist_index += 1
        
    #     # print("dist labels list with tuples: ")
    #     # print(dist_labels)
    #     prediction = find_majority_class(dist_labels)

        
    #     # prediction = predict_digits(k, train_images_loocv, train_labels_loocv, 
    #     #                             train_images[image_index], p=p)

    #     if prediction != train_labels[image_index]:
    #         loss += 1


    # emp_train_l = loss / len(train_images)
    # empirical_train_loss.append(emp_train_l)

def usage():
    # commands surrounded with [] are optional
    print('usage: main.py -q question_letter [-p p_for_minkowski] [--mink min_k_iter] [--maxk max_k_iter] ' +
        '[--kfold cross_v_k] [--title csv_title]. For help: main.py -h')


def main(argv):
    csv_title = None
    max_k = 20
    min_k = 1
    max_p = 15 
    q = None
    try:
        opts, args = getopt.getopt(argv,"hq:p",["help","mink=","maxk=","kfold=", "title="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o == "-q":
            q = a
        elif o == "-p":
            try:
                p = int(a)
            except:
                print(f'argument passed with flag -p: {a} should be a positive whole number')
                sys.exit(2)
        elif o == "--mink":
            # default is 1
            try:
                min_k = int(a)
            except:
                print(a)
                print(f'argument passed with flag --mink: {a} should be a positive whole number')
                sys.exit(2)
        elif o == "--maxk":
            # default is 20
            try:
                max_k = int(a)
            except:
                print(f'argument passed with flag --maxk: {a} should be a positive whole number')
                sys.exit(2)
        elif o == "--kfold":
            try:
                kfold = int(a)
            except:
                print(f'argument passed with flag --kfold: {a} should be a positive whole number')
                sys.exit(2)
        elif o == "--title":
            # default is 'results_1{question letter}'
            csv_title = a
        else:
            assert False, "unhandled option"

    # passing the flag -q is mandatory
    if q == None:
        usage()
        sys.exit(2)

    print(f'min k: {min_k}, max k: {max_k}')

    for k in range(min_k,max_k + 1): # Takes 6min per iteration of k

        # question c needs max p to have range to loop through
        if q == 'c':
            print(f"run question c with k: {k} and max_p: {max_p}")
            globals()[f'question_{q}'](k, max_p)
        else:
            globals()[f'question_{q}'](k)

    df_index = False
    csv_index = False
    df_results = None
    # change shape of data and indexes based on question executed
    if q == 'a':
        data = {'k': list(range(min_k,max_k+1)), 'Empirical Training Loss': empirical_train_loss,
                                'Empirical Test Loss': empirical_test_loss}
        df_results = pd.DataFrame(data=data)
    elif q == 'b':
        data = {'k': list(range(min_k,max_k+1)), 'CVS training data': cross_validation_score}
        df_results = pd.DataFrame(data=data)
    elif q == 'c':
        # create multi index 
        k_range = np.arange(1, max_k+1)
        p_range = np.arange(1, max_p+1)
        print(f"size of index: {k_range.size * p_range.size}")
        df_index = pd.MultiIndex.from_product((k_range, p_range), names=('k', 'p'))
        data = cross_validation_score
        print(f'size of data: {len(data)}')
        csv_index = True
        df_results = pd.DataFrame(data=data, index=df_index)
    elif q == 'e':
        data = {'k': list(range(min_k,max_k+1)), 'Empirical Training Loss': empirical_train_loss}
        df_results = pd.DataFrame(data=data)
    elif q == 'f':
        data = {'Cross Validation Score': cross_validation_score}
        df_results = pd.DataFrame(data=data)

    if csv_title == None:
        csv_title = f'results_q1{q}'

    # put k's accross the rows and p's accross the columns for question c
    if q == 'c':
        df_results = df_results.unstack()
    df_results.to_csv(csv_title, index=csv_index)


if __name__ == '__main__':
    main(sys.argv[1:])

    