import numpy as np
import cPickle

def load_train_data():

    data_dir = '/data/'

    Xtrain_source_file = data_dir + 'yelp13_IMDB_train_feature.pk'
    ytrain_source_file = data_dir + 'IMDB_c5_train_labels.pk'
    Xtrain_target_file = data_dir + 'yelp13_train_feature.pk'
    ytrain_target_file = data_dir + 'yelp13_c5_train_labels.pk'

    # Load data
    Xtrain_source = cPickle.load(open(Xtrain_source_file, 'rb'))
    ytrain_source = cPickle.load(open(ytrain_source_file, 'rb'))
    Xtrain_target = cPickle.load(open(Xtrain_target_file, 'rb'))
    ytrain_target = cPickle.load(open(ytrain_target_file, 'rb'))

    Xtrain_source = np.asarray(Xtrain_source)
    ytrain_source = np.asarray(ytrain_source)
    ytrain_source = np.reshape(ytrain_source, [ytrain_source.shape[0], 1])

    Xtrain_target = np.asarray(Xtrain_target)
    ytrain_target = np.asarray(ytrain_target)
    ytrain_target = np.reshape(ytrain_target, [ytrain_target.shape[0], 1])

    print 'train data loaded.' 

    return [Xtrain_source, ytrain_source, Xtrain_target, ytrain_target]


def load_test_data():

    data_dir = '/data/'

    Xtest_target_file = data_dir + 'yelp13_test_feature.pk'
    ytest_target_file = data_dir + 'yelp13_c5_test_labels.pk'

    Xtest_target = cPickle.load(open(Xtest_target_file, 'rb'))
    ytest_target = cPickle.load(open(ytest_target_file, 'rb'))

    Xtest_target = np.asarray(Xtest_target)
    ytest_target = [ytest_target]
    ytest_target = np.asarray(ytest_target)

    print 'test data loaded.'    

    return [Xtest_target, ytest_target]

def calculate_metrics(y_true, y_pred):
    from sklearn import metrics
    import math

    # Accuracy
    test_acc = metrics.accuracy_score(y_true, y_pred)
    print 'Acc: ', test_acc

    # Macro F1-Score
    test_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    print 'F1:  ', test_f1

    # MSE : mean squared error
    test_mse = metrics.mean_squared_error(y_true, y_pred)
    # print 'MSE: ', test_mse
    
    # RMSE : root mean square error
    test_rmse = math.sqrt(test_mse)
    print 'RMSE:', test_rmse
