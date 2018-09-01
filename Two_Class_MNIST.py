import keras
from keras.datasets import mnist
from keras import backend as K

import numpy as np
import pandas as pd

class TwoClassMnist:

    def returnDataSet(classOne, classTwo):
        '''
        Reducing size of class form 10 to 2
        :param classOne: Label of first class
        :param classTwo: Label of second class        
        :return:
        '''

        # Max size of class
        numClasses = 2
        # set seed for reproducibility / used inside Keras function we can't see
        seed = 16
        np.random.seed(seed)

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        train_picks = np.logical_or(y_train == classOne, y_train == classTwo)
        test_picks = np.logical_or(y_test == classOne, y_test == classTwo)

        x_train = x_train[train_picks]
        x_test = x_test[test_picks]
        y_train = np.array(y_train[train_picks] == classTwo, dtype=int)
        y_test = np.array(y_test[test_picks] == classTwo, dtype=int)
        
        #We also do a one-hot encoding of the labels
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
            x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
            input_shape = (1, 28, 28)
        else:
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
            input_shape = (28, 28, 1)
        
        #normalizing 
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, numClasses)
        y_test = keras.utils.to_categorical(y_test, numClasses)
        
    
        return x_train, x_test, y_train, y_test, input_shape

#x_train, x_test, y_train, y_test, input_shape = TwoClassMnist.returnDataSet(1, 2, 28)