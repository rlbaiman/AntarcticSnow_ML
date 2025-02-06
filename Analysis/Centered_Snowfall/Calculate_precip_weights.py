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


#Get times of top snow
test = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_test_ds.nc', chunks = 'auto')
train = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_train_ds.nc', chunks = 'auto')
val = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_val_ds.nc', chunks = 'auto')
all_data = xr.concat([test, train, val], dim = 'time')
all_data_labels = all_data.labels.sum(dim = 'regions').load()
all_top_snow = all_data.where(all_data_labels>0, drop = True)
top_snow_region_labels = all_top_snow.labels.isel(regions = r).load()
region_top_snow = all_top_snow.isel(time = np.where(top_snow_region_labels == 1)[0])
region_top_snow = region_top_snow.where(region_top_snow.n_channel.isin(['V']), drop = True)


basins = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AIS_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-85,-60)).load()
basins = basins.Zwallybasins == 0
# Region slices
slice_start = np.array([-36, 36, 108, -180 , -108])
slice_end = slice_start + 72
lon_slice = slice(slice_start[r], slice_end[r])
#Grounded +shelf mask
Region_basin_mask = xr.open_mfdataset('/projects/reba1583/Research/Data/AIS_Full_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-90,-40), lon = lon_slice).Zwallybasins
Region_basin_mask=(Region_basin_mask> 0).values
#grid cell area
grid_cell = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research1/SOM_data/MERRA2_gridarea.nc').sel(lat = slice(-90,-40), lon = lon_slice).cell_area

def precipitation_weights(day_xr):
    str_yr, int_month, int_day = str(day_xr.time.values)[:4], int(str(day_xr.time.values)[5:7]), int(str(day_xr.time.values)[8:10])
    precip = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/Maclennan/MERRA2/PRECSN_hourly_'+str_yr+'.nc').sel(lat = slice(-90,-40), lon = lon_slice)
    precip = precip.isel(time = ((precip.time.dt.month == int_month) & (precip.time.dt.day == int_day))).load()
    day_precip = (precip*Region_basin_mask*grid_cell*60*60/(10**12)).sum(dim = 'time')
    weights = day_precip.sum(dim = 'lat').PRECSN.values
    return(day_precip, weights)


precip_weights = []
for t in range(len(region_top_snow.time)):
    spatial, weights = precipitation_weights(region_top_snow.isel(time = t))
    precip_weights.append(weights)
    print(t)


precip_df = pd.DataFrame(precip_weights)
precip_df.to_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Precip/allTopSnow_precip_weights_region'+str(r)+'.csv')
