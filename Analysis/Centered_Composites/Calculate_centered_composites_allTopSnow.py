import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import itertools
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from sklearn import metrics
from random import randint
import scipy
from scipy import stats
import mmap
import matplotlib.path as mpath
import sys

r = int(sys.argv[1])
print(r)


name = 'Final'
cutoff =.45
file_path = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/'
test = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_test_ds.nc', chunks = 'auto')
train = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_train_ds.nc', chunks = 'auto')
val = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_val_ds.nc', chunks = 'auto')

all_data = xr.concat([test, train, val], dim = 'time')
all_data_labels = all_data.labels.sum(dim = 'regions').load()
all_top_snow = all_data.where(all_data_labels>0, drop = True)

top_snow_region_labels = all_top_snow.labels.isel(regions = r).load()
region_top_snow = all_top_snow.isel(time = np.where(top_snow_region_labels == 1)[0])
region_top_snow = region_top_snow.where(region_top_snow.n_channel.isin(['V','IWV', 'SLP', 'H']), drop = True)

titles = np.array(region_top_snow.n_channel.values)

basins_slice_start = [-72,  0,  72,  144, -144]
basins_slice_end = [ 71.5, 143.5, 216, 288,  -.5]

if r ==2 or r ==3:
    region_top_snow= region_top_snow.assign_coords({'lon' :('lon',np.concatenate([np.arange(180,360,.625),np.arange(0,180,.625)]))})
    region_top_snow= region_top_snow.sortby('lon')
region_top_snow = region_top_snow.sel(lon = slice(basins_slice_start[r],basins_slice_end[r])).load()

precip_weights = np.array(pd.read_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Precip/allTopSnow_precip_weights_region'+str(r)+'.csv', index_col = 0))

slice_start = np.array([-36, 36, 108, 180 , -108])
slice_end = np.array([36, 108, 180, 252, -36])
region_lons = region_top_snow.sel(lon = slice(slice_start[r], slice_end[r])).lon.values


basins = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AIS_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-80,-40)).load()
if r==2 or r == 3:
    basins = basins.assign_coords({'lon' :('lon',np.concatenate([np.arange(180,360,.625),np.arange(0,180,.625)]))})
    basins = basins.sortby('lon')
basins = basins.Zwallybasins == 0
basins = basins.sel(lon  = slice(basins_slice_start[r], basins_slice_end[r]))

len_shifted_weights = 230

shifted_V= np.empty((len(region_top_snow.time), len(region_top_snow.lat), len_shifted_weights))
shifted_V[:,:,:] = np.nan
shifted_IWV= np.empty((len(region_top_snow.time), len(region_top_snow.lat), len_shifted_weights))
shifted_IWV[:,:,:] = np.nan
shifted_SLP= np.empty((len(region_top_snow.time), len(region_top_snow.lat), len_shifted_weights))
shifted_SLP[:,:,:] = np.nan
shifted_H= np.empty((len(region_top_snow.time), len(region_top_snow.lat), len_shifted_weights))
shifted_H[:,:,:] = np.nan

for t in range(len(region_top_snow.time)):
    timestamp = str(pd.to_datetime(region_top_snow.isel(time = t).time.values).year) + "%02d"%pd.to_datetime(region_top_snow.isel(time = t).time.values).month+"%02d"%pd.to_datetime(region_top_snow.isel(time = t).time.values).day
    max_id = int(np.where(precip_weights[t] == precip_weights[t].max())[0])
    trim_data = region_top_snow.sel(lon = slice(region_lons[max_id] - 36, region_lons[max_id] + 36)).isel(time = t)
    trim_basins = basins.sel(lon = slice(region_lons[max_id] - 36, region_lons[max_id] + 36))
    max_id_data = int(np.where(trim_data.lon.values == region_lons[max_id])[0])
    
    trim_V = np.where(trim_basins, trim_data.sel(n_channel = 'V').features, 0)
    shifted_V[t][:,(115-max_id_data):(230-max_id_data)] = trim_V
    trim_IWV = np.where(trim_basins, trim_data.sel(n_channel = 'IWV').features, 0)
    shifted_IWV[t][:,(115-max_id_data):(230-max_id_data)] = trim_IWV    
    trim_SLP = np.where(trim_basins, trim_data.sel(n_channel = 'SLP').features, 0)
    shifted_SLP[t][:,(115-max_id_data):(230-max_id_data)] = trim_SLP
    trim_H = np.where(trim_basins, trim_data.sel(n_channel = 'H').features, 0)
    shifted_H[t][:,(115-max_id_data):(230-max_id_data)] = trim_H

out_folder = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Centered_Results/'

pandas_data_V = pd.DataFrame(np.nanmean(shifted_V, 0))
pandas_data_V.to_csv(out_folder+'AllTopSnow_V_Region'+str(r), index = False) 

pandas_data_IWV = pd.DataFrame(np.nanmean(shifted_IWV, 0))
pandas_data_IWV.to_csv(out_folder+'AllTopSnow_IWV_Region'+str(r), index = False) 

pandas_data_SLP= pd.DataFrame(np.nanmean(shifted_SLP, 0))
pandas_data_SLP.to_csv(out_folder+'AllTopSnow_SLP_Region'+str(r), index = False) 

pandas_data_H = pd.DataFrame(np.nanmean(shifted_H, 0))
pandas_data_H.to_csv(out_folder+'AllTopSnow_H_Region'+str(r), index = False) 
