import numpy as np
import pandas as pd
from tqdm import tqdm #Loadingbar
from functions_part_one import *
import sys, getopt
from sklearn.model_selection import LeaveOneOut
from timeit import default_timer as timer
from datetime import timedelta

train_data = np.genfromtxt("MNIST_train_small.csv", delimiter=',')
test_data = np.genfromtxt("MNIST_test_small.csv", delimiter=',')
train_labels, train_images = train_data[:,0], train_data[:, 1:] # Splitting training data
test_labels, test_images = test_data[:,0], test_data[:, 1:] # Splitting test data

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

    if csv_title == None:
        csv_title = f'results_q1{q}'

    # put k's accross the rows and p's accross the columns for question c
    if q == 'c':
        df_results = df_results.unstack()
    df_results.to_csv(csv_title, index=csv_index)


if __name__ == '__main__':
    main(sys.argv[1:])

    