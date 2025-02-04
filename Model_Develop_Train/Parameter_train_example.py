import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils, layers
from tensorflow.keras.models import Model
from keras.regularizers import l1
import pandas as pd
import os
import itertools
from glob import glob
import sys
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from functions import testing_parameters, make_IWV_climo_stats

#select which combo of parameters you would like to use, here we choose 4
n = 4
testing_parameters_select = testing_parameters[n]
print(testing_parameters_select)
name = str(n)

#select which input variables to inclue
variable_list = ['V','IWV']

##################
# Train Data
##################

# Load in data
full_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/full_data.nc', chunks = 'auto')

years = np.unique(np.array(full_data.time.dt.year))
test_years = np.array([1980, 1982, 1985, 2004, 2007, 2017]).tolist() # pre determined 
train_years = np.sort(np.random.choice(years[~np.isin(years, test_years)], int(len(years)*.70), replace=False)).tolist()
val_years = years[~np.isin(years, train_years) & (~np.isin(years, test_years))].tolist()
print('Train years: '+str(train_years))
print('Val years: '+str(val_years))

ds_train = full_data.isel(time = np.where(full_data.time.dt.year.isin(train_years))[0])
ds_train_labels = ds_train.labels # save labels to use later
ds_train.drop_vars('labels') #drop them to make anomalies

ds_val = full_data.isel(time = np.where(full_data.time.dt.year.isin(val_years))[0])
ds_val_labels = ds_val.labels # save labels to use later
ds_val.drop_vars('labels') #drop them to make anomalies

# Create ds_train climotology
climo_mean = ds_train.features.groupby('time.month').mean(dim = 'time',skipna=True)
climo_std = ds_train.features.groupby('time.month').std(dim = 'time',skipna=True)

IWV_climo_mean, IWV_climo_std = make_IWV_climo_stats(ds_train.features.sel(n_channel = 'IWV'))

#update climo mean and std with IWV values
climo_mean = climo_mean.where(climo_mean.n_channel.isin(['H','U','V',"SLP"]), IWV_climo_mean)
climo_std = climo_std.where(climo_std.n_channel.isin(['H','U','V',"SLP"]), IWV_climo_std)

ds_train = xr.apply_ufunc(
        lambda x, c, s: (x - c) / s,
        ds_train.groupby("time.month"),
        climo_mean,
        climo_std,
        dask = 'allowed'
    )

ds_val = xr.apply_ufunc(
        lambda x, c, s: (x - c) / s,
        ds_val.groupby("time.month"),
        climo_mean,
        climo_std,
        dask = 'allowed'
    )

ds_train = ds_train.fillna(0)
ds_val = ds_val.fillna(0)

ds_train['labels'] = ds_train_labels
ds_val['labels'] = ds_val_labels

del full_data
del climo_mean
del climo_std
del IWV_climo_mean
del IWV_climo_std

#Shuffle data
train_random_shuffle = np.arange(len(ds_train.features))
np.random.shuffle(train_random_shuffle)
X_train = ds_train.where(ds_train.n_channel.isin(variable_list), drop = True).features.values[train_random_shuffle]
Y_train = ds_train.labels.values[train_random_shuffle]
del ds_train

#calculate initial bias
pos = np.array(Y_train).sum()
neg = np.where(np.array(Y_train) == 1, 0, 1).sum()
initial_bias = np.log([pos/neg])

train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_data = tf.data.Dataset.from_tensor_slices((ds_val.where(ds_val.n_channel.isin(variable_list), drop = True).features, ds_val.labels))

#batch both 
batch_size = testing_parameters_select[-1]
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

for batch in train_data:
    break 
for batch in val_data:
    break 
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
    if output_bias is not None:
        x = layers.Dense(5, 'sigmoid', name = name, bias_initializer=tf.keras.initializers.constant(output_bias))(x)
    else:
        x = layers.Dense(5, 'sigmoid', name = name)(x)
    model = Model(inputs = inputs, outputs = x)
    return model

    
model  = cnn_model(parameter_select=testing_parameters_select, input_shape = train_data.element_spec[0].shape[1:], name = 'test'+name, output_bias = initial_bias)
print(model.summary())

epoch_num = 1000
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = .01,patience=20,  verbose = 1, restore_best_weights = True)
model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing = True, alpha = .85,
                                                           gamma=2, from_logits=False),
              optimizer=tf.keras.optimizers.Adam(learning_rate=testing_parameters_select[4]), 
              metrics=['accuracy', tf.keras.metrics.AUC(curve = 'PR')])

history = model.fit(train_data, validation_data=val_data, epochs=epoch_num, callbacks = [callback])
history_pd = pd.DataFrame(history.history)

y_preds = model.predict(val_data)
results_pd = pd.DataFrame(y_preds)
results_pd = results_pd.set_index(np.array(ds_val.time))

val_pd = pd.DataFrame(ds_val.labels.values)
val_pd = val_pd.set_index(np.array(ds_val.time))


out_folder = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Test_Results/'
results_pd.to_csv(out_folder+'/test'+name+'_preds.csv')
val_pd.to_csv(out_folder+'/test'+name+'_val.csv')
history_pd.to_csv(out_folder+'/test'+name+'_history.csv',index= False)
