###############################################
# Part 1: select single variable and level
###############################################

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
fp = '/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'
fp_out_1 = '/scratch/alpine/reba1583/variable_yr_files1/'
fp_out_2 = '/scratch/alpine/reba1583/variable_yr_files2/' 


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

    if len(glob.glob(fp_out_1+variables[i]+'*'+str(year)+'*'))==0: #if that variable hasn't already been processed to fp_out_1
        # select the variable
        command = 'cdo -select,name='+variable+' '+variable_files[i]+' '+fp_out_1+file_name
        os.system(command)

    if np.isin(i, [0,1,2]): # if it is a variable that you need to select the level
        if len(glob.glob(fp_out_2+variables[i]+'*'+str(year)+'*'))==0: # if that variable hasn't already been processed to fp_out_2
            command_3 = 'cdo -sellevel,'+variable_levels[i]+' '+glob.glob(fp_out_1+variables[i]+'*'+str(year)+'*')[0]+' '+fp_out_2+file_name
            os.system(command_3)
            os.system('rm '+fp_out_2+file_name)


