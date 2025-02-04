##################################################################
## Part 2: resample to limited latitudes and daily resolution
##################################################################

# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import xesmf as xe
import glob
from Make_data_functions import get_variables, get_variable_names, get_variable_level


year = 1980+int(sys.argv[1])


def resample(file_name, directory, index): 
    print("resampling "+file_name)
    """
    Function to open the variable file at a single level and resample according training needs.
    This includes smoothing in time, lat, and lon. Also includes making LWTNET binary and 
    shifting variables to selected variable_leadtimes
    """
    mask = xr.open_mfdataset('/projects/reba1583/Research/Data/AIS_Full_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-80, -40)).load()

    if directory == 1:
        year_data = xr.open_mfdataset(glob.glob(fp_out_1+variables[index]+'*'+str(year)+'*')[0])

    elif directory ==2:
        year_data = xr.open_mfdataset(glob.glob(fp_out_2+variables[index]+'*'+str(year)+'*')[0])
        year_data = year_data.isel(lev = 0)
        
    data = year_data.resample(time = '24H').mean()
    data = data.sel(lat = slice(-80,-40))
    
    variable_mask = xr.where(mask.Zwallybasins>0, np.nan,1)
    data = (data[variables[i]]*variable_mask.values).load()
    
    data.to_netcdf(fp_out_3+file_name)
    del data



fp = '/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'
fp_out_1 = '/scratch/alpine/reba1583/variable_yr_files1/'
fp_out_2 = '/scratch/alpine/reba1583/variable_yr_files2/' 
fp_out_3 = '/scratch/alpine/reba1583/variable_yr_files3/'


variables = get_variables()
variable_names = get_variable_names()
variable_levels = get_variable_level()

variable_files = [fp+str(year)+'*',  
    fp+str(year)+'*',    
    fp+str(year)+'*',    
    fp+str(year)+'*',
    fp+'IWV/'+str(year)+'*']


for i in range(len(variables)):

    variable = variables[i]
    file_name = variable_names[i]+'_'+str(year)
    
    if os.path.exists(fp_out_3+file_name+'.nc'): #skip if already processed 
        print(file_name+' already processed')
    else:
        if np.isin(i, [0,1,2]):
            resample(file_name, directory = 2, index = i)
            print(file_name+' processed')
            
        elif np.isin(i, [3,4]):
            resample(file_name, directory = 1, index = i)
            print(file_name+' processed')
        
    
