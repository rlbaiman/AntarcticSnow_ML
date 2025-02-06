# Making Data
* From MERRA2 data, create netcdf files for each day 1980-2019 with input variables and total snowfall for that day. Much of this is done in steps because of limited memory.
### 1. Select input variables and Levels
* Select the variables you would like to use as input data at the vertical levels you choose and save as yearly data. 
### 2. Select latitudinal slice and take daily mean of input variables
* Get the daily mean, select 80S to 40S latitudes, and mask out ice sheet and ice shelves. Save as yearly data.
### 3. Make Y
* For each region, make a pdf that lists whether each day falls in the top 85th percentile for snowfall over the ice sheet and ice shelves
### 4. Combine X Y
* Create netcdf with daily input variables and binary y indicating if snowfall falls in the top 85th percentile by region