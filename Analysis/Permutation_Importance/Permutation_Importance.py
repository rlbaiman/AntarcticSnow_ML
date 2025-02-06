import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils, layers
from tensorflow.keras.models import Model
from keras.regularizers import l1, l2
import pandas as pd
import os
import itertools
from glob import glob
from sklearn import metrics
import sys
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


region_labels = ['1','2','3', '4','5']
regions = len(region_labels)
testing_parameters_select = (2, 100, 3, 100, 1e-6, 100)
name = 'Final'
cutoff = .45
fp_out = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Permutation_Importance/'   

fp = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/'
variable_list = ['V', 'IWV']
region_test_X = xr.open_mfdataset(fp+'final_test_ds.nc')
region_test_X = region_test_X.where(region_test_X.n_channel.isin(variable_list), drop = True).features
region_test_Y = xr.open_mfdataset(fp+'final_test_ds.nc').labels
results = pd.read_csv(fp+'CNN_retry_2V/'+name+'_preds.csv', index_col = 0)
titles = np.array(region_test_X.n_channel.values)



test_years = [1980, 1982, 1985, 2004, 2007, 2017]

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

model  = cnn_model(parameter_select=testing_parameters_select, input_shape = region_test_X.shape[1:], name = name)
model.load_weights(fp+'CNN_retry_2V/'+name+'_model_weights.h5')

auc_pr_variables_r0 = []
accuracy_variables_r0 = []
auc_pr_variables_r1 = []
accuracy_variables_r1 = []
auc_pr_variables_r2 = []
accuracy_variables_r2 = []
auc_pr_variables_r3 = []
accuracy_variables_r3 = []
auc_pr_variables_r4 = []
accuracy_variables_r4 = []
for v in range(len(titles)):
    auc_pr_list_r0 = []
    accuracy_list_r0 = []
    auc_pr_list_r1 = []
    accuracy_list_r1 = []
    auc_pr_list_r2 = []
    accuracy_list_r2 = []
    auc_pr_list_r3 = []
    accuracy_list_r3 = []
    auc_pr_list_r4 = []
    accuracy_list_r4 = []
    for p in range(100):
        time_shuffle = np.random.choice(a=np.shape(region_test_X.isel(n_channel = v))[0], size=np.shape(region_test_X.isel(n_channel = v))[0], replace=False)
        region_test_X_shuffle = []
        for var in range(len(titles)):
            if var == v:
                region_test_X_shuffle.append(region_test_X.isel(n_channel = var)[time_shuffle])
            else:
                region_test_X_shuffle.append(region_test_X.isel(n_channel = var))

        region_test_X_shuffle = np.stack(region_test_X_shuffle, axis = -1)
        print(np.shape(region_test_X_shuffle))
        preds = model.predict(region_test_X_shuffle)
        preds_binary = np.where(preds >=.45,1,0)
        
        r = 0
        precision, recall, thresholds = metrics.precision_recall_curve(region_test_Y.values[:,r],preds[:,r], pos_label = 1)
        auc_pr_list_r0.append(np.round(metrics.auc(recall, precision),3))
        accuracy_list_r0.append(np.round(np.where(preds_binary[:,r] == region_test_Y.values[:,r],1,0).sum()/len(preds),2))
        print(np.round(metrics.auc(recall, precision),3))
        
        r = 1
        precision, recall, thresholds = metrics.precision_recall_curve(region_test_Y.values[:,r],preds[:,r], pos_label = 1)
        auc_pr_list_r1.append(np.round(metrics.auc(recall, precision),3))
        accuracy_list_r1.append(np.round(np.where(preds_binary[:,r] == region_test_Y.values[:,r],1,0).sum()/len(preds),2))
        print(np.round(metrics.auc(recall, precision),3))
        
        r = 2
        precision, recall, thresholds = metrics.precision_recall_curve(region_test_Y.values[:,r],preds[:,r], pos_label = 1)
        auc_pr_list_r2.append(np.round(metrics.auc(recall, precision),3))
        accuracy_list_r2.append(np.round(np.where(preds_binary[:,r] == region_test_Y.values[:,r],1,0).sum()/len(preds),2))
        print(np.round(metrics.auc(recall, precision),3))
        
        r = 3
        precision, recall, thresholds = metrics.precision_recall_curve(region_test_Y.values[:,r],preds[:,r], pos_label = 1)
        auc_pr_list_r3.append(np.round(metrics.auc(recall, precision),3))
        accuracy_list_r3.append(np.round(np.where(preds_binary[:,r] == region_test_Y.values[:,r],1,0).sum()/len(preds),2))
        print(np.round(metrics.auc(recall, precision),3))
        
        r = 4
        precision, recall, thresholds = metrics.precision_recall_curve(region_test_Y.values[:,r],preds[:,r], pos_label = 1)
        auc_pr_list_r4.append(np.round(metrics.auc(recall, precision),3))
        accuracy_list_r4.append(np.round(np.where(preds_binary[:,r] == region_test_Y.values[:,r],1,0).sum()/len(preds),2))
        print(np.round(metrics.auc(recall, precision),3))
        
        del region_test_X_shuffle
    auc_pr_variables_r0.append(np.array(auc_pr_list_r0))
    accuracy_variables_r0.append(np.array(accuracy_list_r0))
    auc_pr_variables_r1.append(np.array(auc_pr_list_r1))
    accuracy_variables_r1.append(np.array(accuracy_list_r1))
    auc_pr_variables_r2.append(np.array(auc_pr_list_r2))
    accuracy_variables_r2.append(np.array(accuracy_list_r2))
    auc_pr_variables_r3.append(np.array(auc_pr_list_r3))
    accuracy_variables_r3.append(np.array(accuracy_list_r3))
    auc_pr_variables_r4.append(np.array(auc_pr_list_r4))
    accuracy_variables_r4.append(np.array(accuracy_list_r4))
    print('done with '+titles[v])

data_output_r0 = pd.DataFrame({
    'accuracy_'+titles[0]: accuracy_variables_r0[0],
    'auc_pr_'+titles[0]:auc_pr_variables_r0[0],
    'accuracy_'+titles[1]: accuracy_variables_r0[1],
    'auc_pr_'+titles[1]:auc_pr_variables_r0[1],

})  

data_output_r0.to_csv(fp_out+'PermutationImportance_region0')

data_output_r1 = pd.DataFrame({
    'accuracy_'+titles[0]: accuracy_variables_r1[0],
    'auc_pr_'+titles[0]:auc_pr_variables_r1[0],
    'accuracy_'+titles[1]: accuracy_variables_r1[1],
    'auc_pr_'+titles[1]:auc_pr_variables_r1[1],

})  

data_output_r1.to_csv(fp_out+'PermutationImportance_region1')

data_output_r2 = pd.DataFrame({
    'accuracy_'+titles[0]: accuracy_variables_r2[0],
    'auc_pr_'+titles[0]:auc_pr_variables_r2[0],
    'accuracy_'+titles[1]: accuracy_variables_r2[1],
    'auc_pr_'+titles[1]:auc_pr_variables_r2[1],

})  

data_output_r2.to_csv(fp_out+'PermutationImportance_region2')

data_output_r3 = pd.DataFrame({
    'accuracy_'+titles[0]: accuracy_variables_r3[0],
    'auc_pr_'+titles[0]:auc_pr_variables_r3[0],
    'accuracy_'+titles[1]: accuracy_variables_r3[1],
    'auc_pr_'+titles[1]:auc_pr_variables_r3[1],

})  

data_output_r3.to_csv(fp_out+'PermutationImportance_region3')

data_output_r4 = pd.DataFrame({
    'accuracy_'+titles[0]: accuracy_variables_r4[0],
    'auc_pr_'+titles[0]:auc_pr_variables_r4[0],
    'accuracy_'+titles[1]: accuracy_variables_r4[1],
    'auc_pr_'+titles[1]:auc_pr_variables_r4[1],

})  

data_output_r4.to_csv(fp_out+'PermutationImportance_region4')

