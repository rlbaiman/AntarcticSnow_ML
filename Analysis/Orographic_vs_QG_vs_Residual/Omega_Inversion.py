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

Test_years = [1980, 1982, 1985, 2004, 2007, 2017]
folder = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/'
preds = pd.read_csv(folder+'Final_preds.csv', index_col = 0)
predict = np.where(preds>=.45,1,0)
test = pd.read_csv(folder+'Final_test.csv', index_col = 0)
TP_id = np.where((predict[:,r]==1) & (np.array(test)[:,r]==1))[0] 


H_data = []
T_data = []

for y in range(len(Test_years)):
    H_data.append(xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'+str(Test_years[y])+'*', chunks = 'auto').sel(lat = slice(-90,-60)).resample(time = '1d').mean()['H'])
    T_data.append(xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'+str(Test_years[y])+'*', chunks = 'auto').sel(lat = slice(-90,-60)).resample(time = '1d').mean()['T'])
H_data = xr.concat(H_data, 'time')
T_data = xr.concat(T_data, 'time')

H_TP_region = H_data.isel(time = TP_id)
T_TP_region = T_data.isel(time = TP_id)


if r==2 or r == 3:
    T_TP_region = T_TP_region.rename({'lon':'lon_original'})
    T_TP_region = T_TP_region.assign_coords({'lon' :('lon_original',np.concatenate([np.arange(180,360,.625),np.arange(0,180,.625)]))})
    T_TP_region = T_TP_region.swap_dims({'lon_original':'lon'})
    T_TP_region = T_TP_region.sortby('lon')

    H_TP_region = H_TP_region.rename({'lon':'lon_original'})
    H_TP_region = H_TP_region.assign_coords({'lon' :('lon_original',np.concatenate([np.arange(180,360,.625),np.arange(0,180,.625)]))})
    H_TP_region = H_TP_region.swap_dims({'lon_original':'lon'})
    H_TP_region = H_TP_region.sortby('lon')

print('starting region '+str(r))
Q_list = []
Qn_list = []
Qs_list = []

for t in range(len(TP_id)):
    # Choose time, domain, and list of pressure levels
    time = str(H_TP_region.isel(time = t).time.values) # select t from xr object
    # Obtain datetime object from string containing time
    time_dt = datetime.strptime(time,'%Y-%m-%dT%H:%M:%S.000000000')
    latN = -60               # northernmost latitude in domain, must be between 90º and -90º
    latS = -85               # southernmost latitude in domain, must be between 90º and -90º
    lonW_list = [-72,  0,  72,  144, -144]
    lonE_list = [ 72, 144, 216, 288,  0]
    lonW = lonW_list[r]      # westernmost longitude in domain
    lonE = lonE_list[r]      # easternmost longitude in domain

    level_bottom  = 1000     # pressure level at bottom of domain, in hPa
    level_top     = 200      # pressure level at top of domain, in hPa

    # Choose which method to use for opening the data file (set to True or False)
    cdsapi_download = False

    # Load in variables we need and add units
    temp_3d = T_TP_region.sel(time=time_dt, lat=slice(latS,latN), lon=slice(lonW,lonE)).load()
    gph_3d   = H_TP_region.sel(time=time_dt, lat=slice(latS,latN), lon=slice(lonW,lonE)).load()

    for m in range(len(temp_3d.lat)):
        for n in range(len(temp_3d.lon)):
            data_t = temp_3d.isel(lat = m, lon = n)
            if len(np.argwhere(np.array(np.isnan(data_t))))>0:
                lev_define_slope = data_t.lev[np.where(np.isnan(data_t))[0][-1]+1: np.where(np.isnan(data_t))[0][-1]+3]
                slope = np.diff(data_t.sel(lev = lev_define_slope).values)/np.diff(lev_define_slope.values)

                replace = data_t.sel(lev = lev_define_slope[0]).values + slope*(data_t.lev[data_t.lev>lev_define_slope[0]].values - lev_define_slope[0].values)
                new = np.concatenate([replace, data_t[np.isnan(data_t)==False].values])
                temp_3d[:,m,n] = new
            data_h = gph_3d.isel(lat = m, lon = n)
            if len(np.argwhere(np.array(np.isnan(data_h))))>0:
                lev_nan = np.argwhere(np.array(np.isnan(data_h))).flatten()
                for i in np.flip(lev_nan):
                    p_solve = np.array(data_h.lev[i:i+2])*units.hectopascal
                    t_solve = np.array(temp_3d[i:i+2,m,n])*units.kelvin
                    gph_3d[i,m,n] = (gph_3d[i+1,m,n].values*units.meters - mpcalc.thickness_hydrostatic(p_solve, t_solve)).magnitude

    # Obtain datetime object from string containing time
    time_dt = datetime.strptime(time,'%Y-%m-%dT%H:%M:%S.000000000')

    # save for dimension information
    xarray_information = T_TP_region.sel(time=time_dt, lat=slice(latS,latN), lon=slice(lonW,lonE)).load()


    # Apply Gaussian smoothing in the latitude and longitude dimensions
    # "degree" is an integer that adjusts the smoothing (a larger degree means more dramatic smoothing)
    degree = 30
    temp_3d = mpcalc.smooth_gaussian(temp_3d, n=degree)
    gph_3d  = mpcalc.smooth_gaussian(gph_3d, n=degree)

    temp_3d = temp_3d * units.kelvin
    gph_3d   = gph_3d * units.meter

    # Obtain values of latitude, longitude, pressure
    # Convert pressure from hectopascals to pascals
    press = xarray_information.lev.values * 100 * units.pascal
    lats  = xarray_information.lat.values * units.degrees
    lons  = xarray_information.lon.values * units.degrees


    # Set constants we'll need later
    Rd = 287.06 * (units.joule) / (units.kilogram * units.kelvin) # gas constant for dry air
    cp = 1005 * (units.joule) / (units.kilogram * units.kelvin)   # specific heat at constant pressure
    rE = 6.3781E6 * units.meter                                   # radius of the Earth in meters

    # Create 3D arrays of pressure, lat, and lon
    lats_3d, press_3d, lons_3d = np.meshgrid(lats, press, lons)

    # Get the difference in pressure between adjacent pressure levels
    # This pressure difference should NOT change across the entire 3D grid
    dp = np.diff(press)[0] * units.pascal

    # Get the difference in latitude and longitude between adjacent values in radians
    # These differences also should NOT change across the entire 3D grid
    dlon = np.radians(np.abs(np.diff(lons)[0]))
    dlat = np.radians(np.abs(np.diff(lats)[0]))

    # Create arrays of the x and y distance between adjacent values in the latitude and longitude arrays
    dx_3d, dy_3d = mpcalc.lat_lon_grid_deltas(lons_3d, lats_3d)

    # Calculate the Coriolis parameter using an f-plane approximation and the average latitude in
    # the domain
    f_0 = mpcalc.coriolis_parameter(np.mean(lats)).magnitude

    # Use the 3D temperature and 3D pressure to calculate 3D potential temperature
    press_ref = 100000 * units.pascal
    tpot_3d = temp_3d*((press_ref/press_3d)**(Rd/cp))

    # Calculate the 3D geostrophic wind
    ugeo_3d, vgeo_3d = mpcalc.geostrophic_wind(gph_3d, f = mpcalc.coriolis_parameter(np.mean(lats)), dx=dx_3d, dy=dy_3d)

    # Initialize DataArray objects with values of zero: one for the x component of the Q vector (Q_x),
    # and one for the y component of the Q vector (Q_y)
    Q_x = xr.DataArray(data=np.zeros(np.shape(ugeo_3d)), coords=xarray_information.coords, dims=xarray_information.dims)
    Q_y = xr.DataArray(data=np.zeros(np.shape(ugeo_3d)), coords=xarray_information.coords, dims=xarray_information.dims)
    Q_x = Q_x * (units.meter**2) / units.kilogram / units.second
    Q_y = Q_y * (units.meter**2) / units.kilogram / units.second

    # Calculate and save the i and j components of the Q vectors one pressure level at a time 
    # (the Metpy q_vector function does not work for 3D inputs)
    for i in range(0,len(press)):
        Q_x_atpress, Q_y_atpress = mpcalc.q_vector(ugeo_3d[i,:,:], 
                                                   vgeo_3d[i,:,:], 
                                                   temp_3d[i,:,:], 
                                                   pressure=press[i],
                                                   dx=dx_3d[i,:,:], dy=dy_3d[i,:,:])
        Q_x[i,:,:] = Q_x_atpress
        Q_y[i,:,:] = Q_y_atpress

    # Find magnitude of the potential temperature gradient vector: |∇θ|
    theta_grad_y, theta_grad_x = mpcalc.gradient(tpot_3d, axes=(1,2), deltas = (dy_3d, dx_3d))
    theta_grad_squared = (theta_grad_x**2) + (theta_grad_y**2)

    # Calculate components of Q_n in the x and y directions
    Q_nx = ((Q_x*(theta_grad_x**2)) + (Q_y*theta_grad_x*theta_grad_y))/theta_grad_squared
    Q_ny = ((Q_y*(theta_grad_y**2)) + (Q_x*theta_grad_x*theta_grad_y))/theta_grad_squared

    # Calculate components of Q_s in the x and y directions
    Q_sx = ((Q_x*(theta_grad_y**2)) - (Q_y*theta_grad_x*theta_grad_y))/theta_grad_squared
    Q_sy = ((Q_y*(theta_grad_x**2)) - (Q_x*theta_grad_x*theta_grad_y))/theta_grad_squared

    # Calculate -2 * divergence of the full Q vector, Qn, and Qs, which gives the forcing
    # on the right hand side of the QG-omega equation
    Q_full_div = -2 * mpcalc.divergence(Q_x,  Q_y,  dx=dx_3d, dy=dy_3d)
    Q_n_div    = -2 * mpcalc.divergence(Q_nx, Q_ny, dx=dx_3d, dy=dy_3d)
    Q_s_div    = -2 * mpcalc.divergence(Q_sx, Q_sy, dx=dx_3d, dy=dy_3d)

    # Calculate the static stability parameter

    # Calculate the average of the temperature and potential temperature across the domain
    # weighted by the cosine of the latitude
    weights = np.cos(np.radians(xarray_information.lat))
    weights_3d = np.tile(np.tile(weights, (len(xarray_information.lon),1)).T, (len(xarray_information.lev),1,1))
    temp_3d_weighted = temp_3d * weights_3d
    tpot_3d_weighted = tpot_3d * weights_3d
    temp_0 = np.mean(temp_3d_weighted, (1,2))#.values * units.kelvin
    tpot_0 = np.mean(tpot_3d_weighted, (1,2)) #.values * units.kelvin

    # Take the derivative of the averaged potential temperature with respect to pressure
    stat_stability = mpcalc.first_derivative(tpot_0, delta=dp)

    # Calculate the static stability parameter (note: this is a 1D array following the pressure dimension)
    sigma = -1 * Rd * temp_0 / press / tpot_0 * stat_stability

    # Convert this 1D array to a full 3D array
    ___, sigma_3d, __ = np.meshgrid(lats, sigma, lons)
    sigma_3d = sigma_3d * sigma.units

    # Calculate arrays of latitude at indices j - 1/2 and j + 1/2
    lats_3d_jminushalf = lats_3d - (dlat/2.0)
    lats_3d_jplushalf  = lats_3d + (dlat/2.0)

    # Convert all latitude and longitude arrays to radians before proceeding
    lats_3d_jminushalf_rad = np.radians(lats_3d_jminushalf)
    lats_3d_jplushalf_rad  = np.radians(lats_3d_jplushalf)
    lats_3d_rad = np.radians(lats_3d)
    lons_3d_rad = np.radians(lons_3d)

    # Calculate each of the coefficients, A1 through A5
    # To prevent issues later in the code, we'll remove all Metpy units from values included in the calculations here
    A_1_term1 = -2 * sigma_3d.m / ((np.cos(lats_3d_rad))**2)
    A_1_term2 = -1 * sigma_3d.m * (dlon**2) / (dlat**2) / np.cos(lats_3d_rad) * (np.cos(lats_3d_jplushalf_rad) + np.cos(lats_3d_jminushalf_rad))
    A_1_term3 = -2 * ((f_0 * rE.m * dlon/dp.m)**2)
    A_1 = A_1_term1 + A_1_term2 + A_1_term3

    A_2 = -1 * sigma_3d.m / ((np.cos(lats_3d_rad))**2)
    A_3 = -1 * sigma_3d.m * (dlon**2) * np.cos(lats_3d_jplushalf_rad) / (dlat**2) / np.cos(lats_3d_rad)
    A_4 = -1 * sigma_3d.m * (dlon**2) * np.cos(lats_3d_jminushalf_rad) / (dlat**2) / np.cos(lats_3d_rad)
    A_5 = -1 * ((f_0 * rE.m * dlon/dp.m)**2)

    # Calculate the term containing the forcing from Q vectors that gets multiplied by some other stuff 
    # for each type of Q-vector 
    # Remove all Metpy units like done above
    forcing_terms = {}
    Qvec_names = ['Q_full', 'Q_n', 'Q_s']
    Qvec_div   = [Q_full_div, Q_n_div, Q_s_div]
    for i in range(0,len(Qvec_names)):
        forcing_terms[Qvec_names[i]] = Qvec_div[i].magnitude * ((rE.m * dlon)**2)

    # This dictionary will hold each of the three omega distributions (due to the three different forcings)
    omega_solutions = {}

    for Qvec_name in Qvec_names:

        forcing_term = forcing_terms[Qvec_name]

        print('\nNow solving for omega due to',Qvec_name)

        # Parameters used in the SOR routine that we can adjust if needed
        alpha          = 1.0    # Over-relaxation parameter (aka learning rate)
        tolerance      = 1.0    # Tolerance to determine when to stop SOR routine (for example, 1e-1 -> stop when solutions change by less than total of 1 dPa)
        max_iterations = 10000  # Max number of times SOR routine will run

        # Get count of longitude, latitude, and pressure grid points (count of i, j, and k indices)
        numpts_i = len(lons)
        numpts_j = len(lats)
        numpts_k = len(press)

        # Initialize array of zeroes to represent omega values. The lateral bouundary conditions for omega will remain 0,
        # while the zeros in place of the other omega values represent a first "solution" that the SOR routine starts with
        omega = np.zeros(np.shape(forcing_term))

        # Initialize arrays of omega that are offset in either the i, j, or k indices to represent the various
        # omega terms at different indices (all omega terms except for omega_i,j,k)
        omega_iadd1 = omega[1:-1,1:-1,2:]
        omega_isub1 = omega[1:-1,1:-1,0:-2]
        omega_jadd1 = omega[1:-1,2:,1:-1]
        omega_jsub1 = omega[1:-1,0:-2,1:-1]
        omega_kadd1 = omega[2:,1:-1,1:-1]
        omega_ksub1 = omega[0:-2,1:-1,1:-1]

        # Iterate to solve the entire 3D grid of omega. Each time an iteration occurs, omega will be solved
        # at every grid point. The difference between the previous result for omega and the current result for omega
        # is calculated and summed across all grid points. This sum of differences will become very small once the SOR 
        # routine "converges" on a final solution for omega. The routine stops iterating/solving once the sum of differences
        # falls below the specified tolerance threshold or the maximum number of iterations is reached.
        total_diff = 0
        num_iterations = 0
        while num_iterations < max_iterations:

            # Calculate the new solution for omega
            omega_temp = (1/A_1[1:-1,1:-1,1:-1]) * (forcing_term[1:-1,1:-1,1:-1] + (A_2[1:-1,1:-1,1:-1]*(omega_iadd1 + omega_isub1)) 
                                         + (A_3[1:-1,1:-1,1:-1]*omega_jadd1) + (A_4[1:-1,1:-1,1:-1]*omega_jsub1) + (A_5*(omega_kadd1 + omega_ksub1)))

            # Calculate the difference between this solution and last updated values for omega
            diff = omega_temp - omega[1:-1,1:-1,1:-1]

            # Use this difference to nudge the current values for omega closer to a final solution
            omega[1:-1,1:-1,1:-1] = omega[1:-1,1:-1,1:-1] + (alpha * diff)

            # Calculate the sum of the differences across all dimensions
            total_diff = np.sum(np.abs(diff))

            num_iterations = num_iterations + 1

            # If the sum of the differences between the previous solution for omega and the new solution for omega at
            # all the grid points is less than our tolerance value, then we consider the solutions to have converged
            if total_diff < tolerance:
                print('Converged on solution')
                print('Total number of iterations: '+format(num_iterations))
                print('Final sum of differences: '+format(total_diff))
                break #End loop
            else:
                if num_iterations%500 == 0:
                    print('Iteration '+format(num_iterations,'04d')+': sum of differences = '+format(total_diff))

            # Reset the sum of the differences before iterating again
            total_diff = 0

        omega = np.reshape(omega, (1,np.shape(omega)[0],np.shape(omega)[1],np.shape(omega)[2])) # add time dimension 
        # Save final omega field into DataArray, and store in our dictionary of omega solutions due to different forcings
        
        omega_da = xr.DataArray(omega, dims = ('time','lev','lat','lon'), 
                                name = 'omega_'+Qvec_name, 
                                coords = {'time':[time],'lev':np.array(press)/100,'lat':np.array(lats), 'lon':np.array(lons)})
        omega_da.attrs['units'] = str('pascal / second')
        if r == 2 or r ==3:
            omega_da = omega_da.assign_coords(lon_original =('lon', np.array(xarray_information.lon_original)))
        
        if Qvec_name == 'Q_full':
            Q_list.append(omega_da)
        if Qvec_name == 'Q_n':
            Qn_list.append(omega_da)
        if Qvec_name == 'Q_s':
            Qs_list.append(omega_da)
        print(str(t))

            
Q_xr = xr.concat(Q_list,  dim = 'time')
# Q_xr['time'] = pd.to_datetime(Q_xr['time'])
Qn_xr = xr.concat(Qn_list,  dim = 'time')
# Qn_xr['time'] = pd.to_datetime(Qn_xr['time'])
Qs_xr = xr.concat(Qs_list,  dim = 'time')
# Qs_xr['time'] = pd.to_datetime(Qs_xr['time'])

out_folder = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/CNN_retry/CNN_retry_2V/Omega_new/'
Q_xr.to_netcdf(out_folder+'Q_region'+str(r))
Qn_xr.to_netcdf(out_folder+'Qn_region'+str(r))
Qs_xr.to_netcdf(out_folder+'Qs_region'+str(r))
