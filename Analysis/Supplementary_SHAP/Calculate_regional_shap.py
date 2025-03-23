import xarray as xr 
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils, layers
from tensorflow.keras.models import Model
from keras.regularizers import l1
import pandas as pd
import glob
import sys
colors = ['#F6A5AE', '#228833', '#4577AA','#67CCED', '#AA3377']
import shap

name = 'Final'
file_path = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/'
inputs = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_test_ds.nc')
inputs = inputs.where(inputs.n_channel.isin(['V','IWV']), drop = True)
results = pd.read_csv(file_path+name+'_preds.csv', index_col = 0)
predict = results.where(results>=.45, 0)
predict = predict.where(predict==0, 1)
test = pd.read_csv(file_path+name+'_test.csv', index_col = 0)

#make background sample V and IWV climatology (anomalies=0)
background_sample = np.zeros((50, 81, 576, 2))

#select region
r = 0
tp_id = [np.where((predict[str(r)]==1) & (test[str(r)]==1))[0] for r in range(5)]
sample_tp = inputs.isel(time = tp_id[r])

testing_parameters_select = (2, 100, 3, 100, 1e-6, 100)
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

model  = cnn_model(parameter_select=testing_parameters_select, input_shape = inputs.features.shape[1:], name = name)

model.load_weights(file_path+name+'_model_weights.h5')

e = shap.DeepExplainer(model, background_sample)
print('deep explainer calculated')

#this is chunked to run 5 timesteps at a time to preserve memory. 63 chunks of 5 include any region's number of correctly identified top snow days in test data
for i in range(63):
    
    if len(glob.glob(file_path+'shap/TP_shap_'+str(r)+'_subsample'+str(i))) ==0:
        print('starting chunk '+str(i))
        subsample_sel = [i*5,i*5+5]
        if subsample_sel[0]< len(sample_tp.time):
            shap_values = e.shap_values(sample_tp.isel(time = slice(subsample_sel[0], subsample_sel[1])).features.values)

            shap_ds = xr.DataArray(
                data = shap_values,
                dims = ['regions', 'time', 'lat', 'lon', 'n_channel'],
                coords= {'n_channel':sample_tp.n_channel, 'time':sample_tp.time[subsample_sel[0]:subsample_sel[1]], 
                         'lat':sample_tp.lat, 'lon':sample_tp.lon, 'regions': sample_tp.regions}
            )
            shap_ds.to_netcdf(file_path+'shap/TP_shap_'+str(r)+'_subsample'+str(i))
            del shap_values
            del shap_ds
