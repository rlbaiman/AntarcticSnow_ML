# ##################################################################
# ## Part 3: Make Y data
# ##################################################################

import xarray as xr 
import numpy as np
import pandas as pd
import sys


r = int(sys.argv[1])
region_selects = [[9,0], [1,2], [3,4], [5,6], [7,8]]
lon_slices = [slice(-36, 36), slice(36, 108), slice(108, 180), slice(-180, -108), slice(-108, -36)]
names = ['region1', 'region2', 'region3', 'region4', 'region5']

region_select = region_selects[r]
lon_slice = lon_slices[r]
name = names[r]

AR_catalogue = xr.open_mfdataset('/pl/active/icesheetsclimate/ARTMIP/Wille_AR_catalogues/MERRA2.ar_tag.Wille_v2.3_vIVT.3hourly*').sel(lon = lon_slice, lat = slice(-90,-60))
AR_catalogue = AR_catalogue.sum(dim = ('lat','lon')).load()
AR_catalogue = AR_catalogue.resample(time = '1D').sum()

Y_land = pd.read_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AR_binary_daily.csv', index_col = False)
Y_land= np.array(Y_land.loc[Y_land.index.isin(region_select)].sum())
Y_land = np.where(Y_land>0,1,0)  

Y_data = pd.DataFrame({'time': pd.to_datetime(AR_catalogue.time)[:14610],
                        'Num_AR_cells_south60': AR_catalogue.ar_binary_tag[:14610],
                       'AR_south60': np.where(AR_catalogue.ar_binary_tag>0, 1, 0)[:14610],
                       'AR_land': Y_land 
                      })

#Grounded +shelf mask
Region_basin_mask = xr.open_mfdataset('/projects/reba1583/Research/Data/AIS_Full_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-90,-40), lon = lon_slice).Zwallybasins
Region_basin_mask=(Region_basin_mask> 0).values

#grid cell area
grid_cell = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research1/SOM_data/MERRA2_gridarea.nc').sel(lat = slice(-90,-40), lon = lon_slice).cell_area

precsn = []
for i in range(len(Y_data)):
    precip = xr.open_mfdataset('/pl/active/icesheetsclimate/MERRA2/PRECSN_hourly_'+str(Y_data['time'][i].year)+'.nc').sel(lat = slice(-90,-40), lon = lon_slice)
    precip = xr.open_mfdataset('/pl/active/icesheetsclimate/MERRA2/PRECSN_hourly_'+str(Y_data['time'][i].year)+'.nc').sel(lat = slice(-90,-40), lon = lon_slice)
    day_precip = (precip.sel(time = slice(Y_data['time'][i],Y_data['time'][i]+pd.Timedelta(1,'d')))*Region_basin_mask*grid_cell*60*60/(10**12)).sum(dim = 'time').sum(dim = ('lat','lon'))
    precsn.append(float(day_precip.PRECSN.values))
    if i in np.arange(0, len(Y_data), 100):
        print(i)
        
Y_data['precsn'] = precsn

q95 = np.quantile(Y_data['precsn'],.95)
q90 = np.quantile(Y_data['precsn'],.90)
q85 = np.quantile(Y_data['precsn'],.85)
q80 = np.quantile(Y_data['precsn'],.80)

Y_data['precip_q80'] = np.where(Y_data['precsn']>=q80,1, 0)
Y_data['precip_q85'] = np.where(Y_data['precsn']>=q85,1, 0)
Y_data['precip_q90'] = np.where(Y_data['precsn']>=q90,1, 0)
Y_data['precip_q95'] = np.where(Y_data['precsn']>=q95,1, 0)

Y_data.to_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/precip_Y_'+name+'.csv')