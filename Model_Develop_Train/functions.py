import xarray as xr
import pandas as pd
import numpy as np
import itertools
import sys

import tensorflow as tf
from tensorflow.keras import utils, layers
from tensorflow.keras.models import Model
from keras.regularizers import l1, l2

def testing_parameters():
    """
    Determine the different combinations of parameters included in testing
    """
    Num_con_layers = [2, 3]
    Num_con_filters = [64]

    Num_dense_layers = [2,3]
    Num_dense_filters = [64]

    Learning_rate = [1e-2, 1e-5]
    Batch_size = [60, 30]
    

    options = [ Num_con_layers, Num_con_filters, Num_dense_layers, Num_dense_filters, Learning_rate,  Batch_size]

    return(list(itertools.product(*options)))

def testing_parameters2():
    """
    Determine the different combinations of parameters included in testing
    """
    Num_con_layers = [2]
    Num_con_filters = [64]

    Num_dense_layers = [2]
    Num_dense_filters = [64]

    Learning_rate = [1e-5, 1e-7]
    Batch_size = [60, 80]
    

    options = [ Num_con_layers, Num_con_filters, Num_dense_layers, Num_dense_filters, Learning_rate,  Batch_size]

    return(list(itertools.product(*options)))

def testing_parameters3():
    """
    Determine the different combinations of parameters included in testing
    """
    list = [(2, 64, 2, 64, 1e-5, 100),
    (2, 64, 2, 64, 1e-4, 80),
    (2, 100, 2, 64, 1e-5, 80),
    (2, 64, 2, 100, 1e-5, 80),
    ]
    
    return(list)


def get_variable_names():
    return([
    'H500_lead0', 
    'U800_lead0',
    'V800_lead0',
    'SLP_lead0',
    'IWV_lead0'])
def get_variables():
    return([
    'H',
    'U',
    'V',
    'SLP',
    'IWV'])

def get_variable_level():
    return([
    '500',
    '800',    
    '800',    
    None,  
    None])

def make_IWV_climo_stats(data_input):
    """
    input data and create a symmetric distribution based on the right hand side of the distribution. 
    Use this for IWV before calculating the std and mean so that the std is not skewed by the limit
    of the distribution at 0. 
    """
    mean = data_input.mean(dim = 'time')
    data1 = data_input.where(data_input>= mean)
    data2 = mean - np.abs(data1 - mean)
    data2['time'] = data2.time +pd.Timedelta('1H')

    data_updated = xr.concat((data1, data2), dim = 'time')

    mean_out = data_updated.groupby("time.month").mean('time',skipna=True)
    std_out = data_updated.groupby("time.month").std('time',skipna=True)
    return(mean_out, std_out)


def cnn_model(parameter_select, input_shape, name, output_bias = None):
    """
    Determine the different combinations of parameters included in testing
    """
    NUM_CON_LAYERS, NUM_CON_FILTERS, NUM_DENSE_LAYERS, NUM_DENSE_FILTERS, LEARNING_RATE, BATCH_SIZE = parameter_select

    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(NUM_CON_FILTERS, (3, 3), padding='valid', input_shape=input_shape, activity_regularizer=l1(1e-6),
                  activation = 'relu', name = name+'_convolution_0')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="valid", name = name+'_maxpool_0')(x)

    for a in range(NUM_CON_LAYERS -1):
        x = layers.Conv2D(NUM_CON_FILTERS, (3, 3), padding='valid', activity_regularizer=l1(1e-6),
                  activation = 'relu', name = name+'_convolution_'+str(a+1))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="valid", name = name+'_maxpool_'+str(a+1))(x)

    x = layers.Flatten()(x)

    for b in range(NUM_DENSE_LAYERS):
        x = layers.Dense(NUM_DENSE_FILTERS, activation='relu', name = name+'_dense_layer_'+str(b))(x)
        x = layers.Dropout(.2)(x)
    if output_bias is not None:
        x = layers.Dense(5, 'sigmoid', name = name, bias_initializer=tf.keras.initializers.constant(output_bias))(x)
    else:
        x = layers.Dense(5, 'sigmoid', name = name)(x)
    model = Model(inputs = inputs, outputs = x)
    return model

def ann_model(NUM_DENSE_LAYERS, NUM_DENSE_FILTERS, BATCH_SIZE, input_shape, name):
    """
    Determine the different combinations of parameters included in testing
    """
  
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Flatten()(inputs)
    x = layers.Dense(NUM_DENSE_FILTERS, name = name+'_dense_layer_0', activation = 'relu')(x)

    for b in range(NUM_DENSE_LAYERS-1):
        x = layers.Dense(NUM_DENSE_FILTERS, activation='relu', name = name+'_dense_layer_'+str(b+1))(x)
            
    x = layers.Dropout(.4)(x)
    x = layers.Dense(5, 'sigmoid', name = name)(x)
    model = Model(inputs = inputs, outputs = x)
    return model


def cnn_model_activationchange(parameter_select, input_shape, name, activation = 'relu', output_bias = None):
    """
    Determine the different combinations of parameters included in testing
    """
    NUM_CON_LAYERS, NUM_CON_FILTERS, NUM_DENSE_LAYERS, NUM_DENSE_FILTERS, LEARNING_RATE, BATCH_SIZE = parameter_select

    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(NUM_CON_FILTERS, (3, 3), padding='valid', input_shape=input_shape, activity_regularizer=l1(1e-6),
                  activation = activation, name = name+'_convolution_0')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="valid", name = name+'_maxpool_0')(x)

    for a in range(NUM_CON_LAYERS -1):
        x = layers.Conv2D(NUM_CON_FILTERS, (3, 3), padding='valid', activity_regularizer=l1(1e-6),
                  activation = activation, name = name+'_convolution_'+str(a+1))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="valid", name = name+'_maxpool_'+str(a+1))(x)

    x = layers.Flatten()(x)

    for b in range(NUM_DENSE_LAYERS):
            x = layers.Dense(NUM_DENSE_FILTERS, activation=activation, name = name+'_dense_layer_'+str(b))(x)
            
    x = layers.Dropout(.2)(x)
    if output_bias is not None:
        x = layers.Dense(5, 'sigmoid', name = name, bias_initializer=tf.keras.initializers.constant(output_bias))(x)
    else:
        x = layers.Dense(5, 'sigmoid', name = name)(x)
    model = Model(inputs = inputs, outputs = x)
    return model

def cnn_model_regularizationchange(parameter_select, input_shape, name, regular = None, output_bias = None):
    """
    Determine the different combinations of parameters included in testing
    """
    NUM_CON_LAYERS, NUM_CON_FILTERS, NUM_DENSE_LAYERS, NUM_DENSE_FILTERS, LEARNING_RATE, BATCH_SIZE = parameter_select

    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(NUM_CON_FILTERS, (3, 3), padding='valid', input_shape=input_shape, activity_regularizer=l1(1e-6),
                  activation = 'relu', name = name+'_convolution_0')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="valid", name = name+'_maxpool_0')(x)

    for a in range(NUM_CON_LAYERS -1):
        x = layers.Conv2D(NUM_CON_FILTERS, (3, 3), padding='valid', activity_regularizer=regular,
                  activation = 'relu', name = name+'_convolution_'+str(a+1))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides = 2, padding="valid", name = name+'_maxpool_'+str(a+1))(x)

    x = layers.Flatten()(x)

    for b in range(NUM_DENSE_LAYERS):
            x = layers.Dense(NUM_DENSE_FILTERS, activation='relu', name = name+'_dense_layer_'+str(b))(x)
            
    x = layers.Dropout(.2)(x)
    if output_bias is not None:
        x = layers.Dense(5, 'sigmoid', name = name, bias_initializer=tf.keras.initializers.constant(output_bias))(x)
    else:
        x = layers.Dense(5, 'sigmoid', name = name)(x)
    model = Model(inputs = inputs, outputs = x)
    return model
