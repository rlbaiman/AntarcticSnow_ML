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
import glob


r = int(sys.argv[1])
print(r)

slice_start = np.array([-36, 36, 108, -180 , -108])
slice_end = slice_start + 72

name = 'Final'
cutoff =.45
file_path = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/'
inputs = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_test_ds.nc')
inputs = inputs.where(inputs.n_channel.isin(['V','IWV', 'SLP', 'H']), drop = True)
results = pd.read_csv(file_path+name+'_preds.csv', index_col = 0)
predict = results.where(results>=.45, 0)
predict = predict.where(predict==0, 1)
test = pd.read_csv(file_path+name+'_test.csv', index_col = 0)
titles = np.array(inputs.n_channel.values)
tp_id = [np.where((predict[str(r)]==1) & (test[str(r)]==1))[0] for r in range(5)]

ds_test = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_test_ds.nc')
test_years = np.unique(pd.to_datetime(ds_test.time.values).year)

tp_time_region = ds_test.isel(time = tp_id[r]).time

if len(glob.glob('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/H500_leadtimes/H500_full_lead0_Region'+str(r)+'*')) == 0:

    out_data = []
    for year in test_years:

        tp_time_region_year = tp_time_region[pd.to_datetime(tp_time_region.values).year == year]
        data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'+str(year)+'*', chunks = 100)
        climo = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research2/climo_3hrly_merra2_hemisphere/H_climo.nc')
        data = data.groupby('time.month') - climo
        data = data.resample(time = '24H').mean()
        data = data.sel(time = tp_time_region_year).load()
        out_data.append(data)

    out_nc = xr.concat(out_data, dim = 'time')
    out_nc.to_netcdf('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/H500_leadtimes/H500_full_lead0_Region'+str(r)+'.nc')


