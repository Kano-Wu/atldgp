#
# Demo application for the ATL-DGP model.
#
# Paper: M. Kandemir, Asymmetric Transfer Learning with Deep Gaussian Processes, ICML, 2015
#
# Contact: melihkandemir@gmail.com
#
# All rights reserved.
#
# from DSGPSymmetricTransferClassifier import DSGPSymmetricTransferClassifier
from DSGPAsymmetricTransferClassifier import DSGPAsymmetricTransferClassifier
# from DSGPAsymmetricTransferClassifier_mod import DSGPAsymmetricTransferClassifier
from RBFKernel import RBFKernel
from data_util import load_train_data, load_test_data, calculate_metrics
import numpy as np
import scipy.io
import cPickle

def ATL_DGP():

    num_inducing_points=10    # Number of inducing points
    num_hidden_units_source=2 # Number of hidden units per class for the source task
    num_hidden_units_target=2 # Number of hidden units per class for the target task    
    max_iteration_count=1     # Maximum number of iterations allowed
    learning_rate_start=0.001 # Starting learning rate
    inducing_kernel = RBFKernel(np.sqrt(num_hidden_units_source))

    
    Xtrain_source, ytrain_source, Xtrain_target, ytrain_target = load_train_data()    

    Xtrain_source = Xtrain_source[:1000, :]
    ytrain_source = ytrain_source[:1000]
    Xtrain_target = Xtrain_target[:1000, :]
    ytrain_target = ytrain_target[:1000]


    print 'Xtrain_source.shape:', Xtrain_source.shape
    print 'ytrain_source.shape:', ytrain_source.shape

    Nsrc = Xtrain_source.shape[0]              
 
    print 'Xtrain_target.shape:', Xtrain_target.shape
    print 'ytrain_target.shape:', ytrain_target.shape
    
    # Concatenate source and target data sets
    Data = np.concatenate((Xtrain_source, Xtrain_target))
    labels = np.concatenate((ytrain_source, ytrain_target))

    print 'Data.shape:' , Data.shape
    print 'labels.shape:', labels.shape 

    # Construct the source-target info map. 
    # 0: data point is on the source task
    # 1: data point is on the target task
    source_target_info = np.ones([Data.shape[0],1])
    source_target_info[0:Nsrc] = 0

    # Construct kernel lists
    kernels_source = list()  
    for rr in range(num_hidden_units_source):
       length_scale = Data.shape[1]
       kernel = RBFKernel(length_scale)  
       kernels_source.append(kernel)
         
    kernels_target = list()  
    for rr in range(num_hidden_units_target):
       length_scale = Data.shape[1]
       kernel = RBFKernel(length_scale)  
       kernels_target.append(kernel)     

#  Comment in the lines below to try out the symmetric classifier
#    common_dimensions=2
#    model=DSGPSymmetricTransferClassifier(inducing_kernel,kernels_source, kernels_target, 2, \
#                                             num_inducing=num_inducing_points, \
#                                             max_iter=iter_cnt, \
#                                             learning_rate_start=learning_rate_start)                         
                         

    # Create the class object for the asymmetric classifier                         
    model = DSGPAsymmetricTransferClassifier(inducing_kernel, kernels_source, kernels_target, num_inducing_points, \
            max_iteration_count, learning_rate_start=learning_rate_start)                         
                        
             
    print 'model training ...'

    # Train the model    
    model.train(Data,labels,source_target_info)

    print 'model train done.'

    Xtest_target, ytest_target = load_test_data()

    print 'Xtest_target.shape:', Xtest_target.shape
    print 'ytest_target.shape:', ytest_target.shape

    print 'predicting ...'

    # Predict on test data and report accuracy
    predictions = model.predict(Xtest_target)
    print "Accuracy: %.2f %% " % ( np.mean(predictions.predictions==ytest_target)*100)

    calculate_metrics(ytest_target[0], predictions.predictions)


if __name__ == "__main__":
    ATL_DGP()