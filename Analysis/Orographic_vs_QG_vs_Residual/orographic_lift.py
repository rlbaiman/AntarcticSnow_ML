import xarray as xr 
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cartopy.feature as cfeature
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import colorsys
#plot parameters that I personally like, feel free to make these your own.
import matplotlib
import matplotlib.patheffects as path_effects


from metpy.interpolate import cross_section
import metpy.calc as mpcalc
from scipy import stats
from metpy.calc import lat_lon_grid_deltas, first_derivative
import sys

r = int(sys.argv[1])
print(r)

extend_slice_start = [-72,  0,  72,  144, -144]
extend_slice_end = [ 71.5, 144, 216, 288,  -.5]

name = 'Final'
cutoff =.45
file_path = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/'
inputs = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/final_test_ds.nc')
inputs = inputs.where(inputs.n_channel.isin(['V','IWV']), drop = True)
results = pd.read_csv(file_path+name+'_preds.csv', index_col = 0)
predict = results.where(results>=cutoff, 0)
predict = predict.where(predict==0, 1)
test = pd.read_csv(file_path+name+'_test.csv', index_col = 0)
titles = np.array(inputs.n_channel.values)
tp_id = [np.where((predict[str(r)]==1) & (test[str(r)]==1))[0] for r in range(5)]

topo = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/Maclennan/MERRA2/AIS_Elevation_MERRA2.nc').isel(time = 0)
if r == 2 or r ==3:
    topo = topo.assign_coords({'lon' :('lon',np.concatenate([np.arange(180,360,.625),np.arange(0,180,.625)]))})
    topo = topo.sortby('lon')

## Elevation Data
topo_data = topo.sel(lon = slice(extend_slice_start[r], extend_slice_end[r]))
dx, dy = lat_lon_grid_deltas(topo_data.lon, topo_data.lat)
topo_dx = first_derivative(f = topo_data.PHIS.values, axis = 1, delta = np.array(dx))
topo_dy = first_derivative(f = topo_data.PHIS.values, axis = 0, delta = np.array(dy))

inputs_regionalTP = inputs.isel(time = tp_id[r])



regional_spatial_lift_estimates = []
for t in range(len(tp_id[r])):
    timestep = inputs_regionalTP.isel(time = t).time
    variable_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'+str(pd.to_datetime(timestep.values).year)+str(pd.to_datetime(timestep.values))[5:7]+str(pd.to_datetime(timestep.values))[8:10]+'.nc')
    if r == 2 or r ==3:
        variable_data = variable_data.assign_coords({'lon' :('lon',np.concatenate([np.arange(180,360,.625),np.arange(0,180,.625)]))})
        variable_data = variable_data.sortby('lon')
        
#     variable_data = variable_data.sel(lat = slice(-90, -40), lon = slice(extend_slice_start[r],extend_slice_end[r])).mean(dim = 'time')[['V','U','SLP','T']].sel(lev = 700).load()

#     lift = ((variable_data.V*np.array(topo_dy)) + (variable_data.U*np.array(topo_dx)))
#     row = 70000/(287*variable_data.T)
    
#     total_lift_spatial = lift*row*9.80665
#     regional_spatial_lift_estimates.append(total_lift_spatial)

# spatial_lift = xr.concat(regional_spatial_lift_estimates, dim = 'time') 
# spatial_lift = spatial_lift.to_dataset(name='Orographic_Lift')

# spatial_lift.to_netcdf('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Omega_new/new_extendlon_700_orographic_omega_spatial_Region'+str(r)+'.nc')
# print(str(r)+' done')

    variable_data = variable_data.sel(lat = slice(-90, -40), lon = slice(extend_slice_start[r],extend_slice_end[r])).mean(dim = 'time')[['V','U','SLP','T']].sel(lev = 800).load()

    lift = ((variable_data.V*np.array(topo_dy)) + (variable_data.U*np.array(topo_dx)))
    row = 80000/(287*variable_data.T)
    
    total_lift_spatial = lift*row*9.80665
    regional_spatial_lift_estimates.append(total_lift_spatial)

spatial_lift = xr.concat(regional_spatial_lift_estimates, dim = 'time') 
spatial_lift = spatial_lift.to_dataset(name='Orographic_Lift')

spatial_lift.to_netcdf('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Omega_new/new_extendlon_800_orographic_omega_spatial_Region'+str(r)+'.nc')
print(str(r)+' done')