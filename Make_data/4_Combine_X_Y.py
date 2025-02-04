# ##################################################################
# ## Part 4: Combine X and Y data 
# ##################################################################
   
# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os   
from Make_data_functions import get_variables, get_variable_names, get_variable_level
    
variables = get_variables()
variable_names = get_variable_names()
variable_levels = get_variable_level()

# Get X data
fp = '/scratch/alpine/reba1583/variable_yr_files3/'
data = [xr.open_mfdataset(fp+variable_names[i]+'*')[variables[i]].transpose('time', 'lat','lon').values for i in range(len(variables))]
data = np.stack(data, 3) # order the dimensions time, lat, lon, variable

# Get Y data
region_st = ['1','2','3','4','5']
Y = [np.array(pd.read_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/precip_Y_region'+region_st[r]+'.csv')['precip_q85']) for r in range(len(region_st))]
Y = np.stack(Y, 0)
Y = np.swapaxes(Y, 0, 1) # order dimensions time, regions

# Time values
variable_times = pd.to_datetime(xr.open_mfdataset(fp+variable_names[0]+'*').time.values)
variable_lons = xr.open_mfdataset(fp+variable_names[0]+'*').lon.values
variable_lats = xr.open_mfdataset(fp+variable_names[0]+'*').lat.values

var_data = dict(
    features = ([ 'time', 'lat', 'lon', 'n_channel'], data),
    labels = (['time', 'regions'], Y)
)

coords = dict(
    time = (['time'], variable_times), 
    lon = (['lon'], variable_lons),
    lat = (['lat'], variable_lats),
    regions = (['regions'], np.arange(1,6,1)),
    n_channel = (['n_channel'], np.array(variables)),     
)

ds = xr.Dataset(
    data_vars = var_data, 
    coords = coords
)

ds.to_netcdf('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/full_data.nc')
