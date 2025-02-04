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

testing_parameters_select = (2, 100, 3, 100, 1e-6, 100)
print(testing_parameters_select)
name = 'Final'

ds_train = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_train_ds.nc')
ds_val = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_val_ds.nc')
ds_test = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_test_ds.nc')


##################
# Train Data
##################
variable_list = ['V','IWV']


#Shuffle data
train_random_shuffle = np.arange(len(ds_train.features))
np.random.shuffle(train_random_shuffle)
X_train = ds_train.where(ds_train.n_channel.isin(variable_list), drop = True).features.values[train_random_shuffle]
Y_train = ds_train.labels.values[train_random_shuffle]
del ds_train


train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_data = tf.data.Dataset.from_tensor_slices((ds_val.where(ds_val.n_channel.isin(variable_list), drop = True).features, ds_val.labels))
test_data = tf.data.Dataset.from_tensor_slices((ds_test.where(ds_test.n_channel.isin(variable_list), drop = True).features, ds_test.labels))

#batch both 
batch_size = testing_parameters_select[-1]
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)
test_data = test_data.batch(batch_size)

for batch in train_data:
    break 
for batch in val_data:
    break 
for batch in test_data:
    break 

def cnn_model(parameter_select, input_shape, name):
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
    x = layers.Dense(5, 'sigmoid', name = name)(x)
    model = Model(inputs = inputs, outputs = x)
    return model

    
model  = cnn_model(parameter_select=testing_parameters_select, input_shape = train_data.element_spec[0].shape[1:], name = name)
print(model.summary())

epoch_num = 1000
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = .01,patience=40,  verbose = 1, restore_best_weights = True)
model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=False),
              optimizer=tf.keras.optimizers.Adam(learning_rate=testing_parameters_select[4]), 
              metrics=['accuracy', tf.keras.metrics.AUC(curve = 'PR')])

history = model.fit(train_data, validation_data=val_data, epochs=epoch_num, callbacks = [callback])
history_pd = pd.DataFrame(history.history)

y_preds = model.predict(test_data)
results_pd = pd.DataFrame(y_preds)
results_pd = results_pd.set_index(np.array(ds_test.time))

test_pd = pd.DataFrame(ds_test.labels.values)
test_pd = test_pd.set_index(np.array(ds_test.time))


out_folder = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/'
results_pd.to_csv(out_folder+name+'_preds.csv')
test_pd.to_csv(out_folder+name+'_test.csv')
history_pd.to_csv(out_folder+name+'_history.csv',index= False)
model.save(out_folder+name+'_model.keras')
model.save_weights(out_folder+name+'_model_weights.h5')