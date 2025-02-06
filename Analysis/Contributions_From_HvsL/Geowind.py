# Import all modules
import numpy as np
from datetime import datetime
import xarray as xr
import metpy
import metpy.calc as mpcalc
from metpy.units import units
import mmap
import pandas as pd
import sys

r = int(sys.argv[1])

## low anomalies

H_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/H500_leadtimes/H500_full_lead0_Region'+str(r)+'.nc', chunks = 'auto').sel(lat = slice(-90,-40))['H']

# mask out all negative H anomalies
H_TP_region = xr.where(H_data>0, H_data, 0).load()            
H_TP_region   = H_TP_region * units.meter
press = H_TP_region.lev.values * 100 * units.pascal
lats  = H_TP_region.lat.values * units.degrees
lons  = H_TP_region.lon.values * units.degrees


### Calculate and save geostrophic wind
# Create 3D arrays of pressure, lat, and lon
lats_3d, press_3d, lons_3d = np.meshgrid(lats, press, lons)
# Create arrays of the x and y distance between adjacent values in the latitude and longitude arrays
dx_3d, dy_3d = mpcalc.lat_lon_grid_deltas(lons_3d, lats_3d)


geo_list = []
for t in range(len(H_TP_region.time)):

    ugeo, vgeo = mpcalc.geostrophic_wind(H_TP_region.isel(time = t), f = np.mean(mpcalc.coriolis_parameter(lats)), dx = dx_3d, dy = dy_3d)
    da = xr.Dataset(
        data_vars =dict(
            ugeo = (["time", "lev", "lat", "lon"], np.reshape(ugeo.magnitude, (1, np.shape(ugeo.magnitude)[0], np.shape(ugeo.magnitude)[1], np.shape(ugeo.magnitude)[2]))),
            vgeo = (["time", "lev", "lat", "lon"], np.reshape(vgeo.magnitude, (1, np.shape(vgeo.magnitude)[0], np.shape(vgeo.magnitude)[1], np.shape(vgeo.magnitude)[2])))
        ),
        coords=dict(
            time =[pd.to_datetime(H_TP_region.isel(time = 0).time.values)],
            lev=(["lev"], H_TP_region.lev.values),
            lat=(["lat"], H_TP_region.lat.values),
            lon=(["lon"], H_TP_region.lon.values),

        ),
        attrs=dict(
            units="m/s",
        ),
    )
    geo_list.append(da)
geo_xr = xr.concat(geo_list, dim = 'time')
geo_xr.to_netcdf('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Contribution_H_L/highH_geostrophic_wind_region'+str(r))

del geo_list
del geo_xr



## low anomalies

H_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/H500_leadtimes/H500_full_lead0_Region'+str(r)+'.nc', chunks = 'auto').sel(lat = slice(-90,-40))['H']
# mask out all positive H anomalies
H_TP_region = xr.where(H_data<0, H_data, 0).load()            
H_TP_region   = H_TP_region * units.meter
press = H_TP_region.lev.values * 100 * units.pascal
lats  = H_TP_region.lat.values * units.degrees
lons  = H_TP_region.lon.values * units.degrees


### Calculate and save geostrophic wind
# Create 3D arrays of pressure, lat, and lon
lats_3d, press_3d, lons_3d = np.meshgrid(lats, press, lons)
# Create arrays of the x and y distance between adjacent values in the latitude and longitude arrays
dx_3d, dy_3d = mpcalc.lat_lon_grid_deltas(lons_3d, lats_3d)


geo_list = []
for t in range(len(H_TP_region.time)):

    ugeo, vgeo = mpcalc.geostrophic_wind(H_TP_region.isel(time = t), f = np.mean(mpcalc.coriolis_parameter(lats)), dx = dx_3d, dy = dy_3d)
    da = xr.Dataset(
        data_vars =dict(
            ugeo = (["time", "lev", "lat", "lon"], np.reshape(ugeo.magnitude, (1, np.shape(ugeo.magnitude)[0], np.shape(ugeo.magnitude)[1], np.shape(ugeo.magnitude)[2]))),
            vgeo = (["time", "lev", "lat", "lon"], np.reshape(vgeo.magnitude, (1, np.shape(vgeo.magnitude)[0], np.shape(vgeo.magnitude)[1], np.shape(vgeo.magnitude)[2])))
        ),
        coords=dict(
            time =[pd.to_datetime(H_TP_region.isel(time = 0).time.values)],
            lev=(["lev"], H_TP_region.lev.values),
            lat=(["lat"], H_TP_region.lat.values),
            lon=(["lon"], H_TP_region.lon.values),

        ),
        attrs=dict(
            units="m/s",
        ),
    )
    geo_list.append(da)
geo_xr = xr.concat(geo_list, dim = 'time')
geo_xr.to_netcdf('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Contribution_H_L/lowH_geostrophic_wind_region'+str(r))

del geo_list
del geo_xr

