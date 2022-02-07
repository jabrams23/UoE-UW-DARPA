---
#cover: cover.jpg
title: Data Sets
permalink: /data/
---
{% include base.html %}
## ERA5
From the [European Centre for Medium-Range Forecasts (ECMWF)](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5):
> ERA5 provides hourly estimates of a large number of atmospheric, land and oceanic climate variables.
> The data cover the Earth on a 30km grid and resolve the atmosphere using 137 levels from the surface 
> up to a height of 80km. ERA5 includes information about uncertainties for all variables at reduced 
> spatial and temporal resolutions.

We provide the following <code>python</code> script to download data from ERA5. This script may be downloaded <a href="{{base}}/../scripts/download_pressure_grid.py" download>here</a>. Follow the commented instructions at the beginning of the script before running it.

```python
# Created by Troy Arcomano (https://github.com/Arcomano1234)

### Prerequisites ###
#Make account to download ERA 5 data
#
#Follow instructions at https://cds.climate.copernicus.eu/api-how-to
#
#Make sure CDS API url and key are installed to $HOME/.cdsapirc

#See (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form) for all variable, pressure levels, time dimensions,
#and grid 

###EXAMPLES###
# python download_pressure_grid.py 1990 1991 -path=/example/path/ --np=2  --vars temperature fraction_of_cloud_cover --plev 1000 500 --grid 0.5 0.5 
# This will download hourly temperature and fraction_of_cloud_cover data at 1000 and 500 hPa from 1990 through 1991 to /example/path/ directory 
# using 2 processes. The grid will be 0.5 by 0.5 degrees

import cdsapi
from calendar import monthrange
from multiprocessing import Pool, cpu_count
from functools import partial
from os import path
import logging
import argparse
import numpy as np

#Full list of avaliable ERA5 vars 
avail_vars = ['divergence', 'fraction_of_cloud_cover', 'geopotential',
                'ozone_mass_mixing_ratio', 'potential_vorticity', 'relative_humidity',
                'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 'specific_humidity',
                'specific_rain_water_content', 'specific_snow_water_content', 'temperature',
                'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                'vorticity',
             ]

#ERA 5 api initializer 
c = cdsapi.Client()

#Download a month of data at a time and 
#save to a netCDF file
def month_downloader(client,month,year,variables,pressure_levels,grid,directory):
    """
    Function that downloads hourly pressure level ERA5 reanalysis data to a netCDF file 

    Input: client          : cdsapi client
           month           : integer, month to download
           year            : integer, year to download
           variables       : list of strings, variables wished to download
           pressure_levels : list of ints, pressure levels to download (see ERA site for more info)
           grid            : list of reals, resolution of the data [x,y] in degrees
           directory       : string, directory to write the data to

    Output: None but does write file to directory
    """
     
    days = monthrange(year,month)[1]
    daysarray = np.arange(1,days+1)

    nparraytostr = lambda daysarray:'%02d'%daysarray
    dayslist = list(map(nparraytostr,daysarray))

    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type':'reanalysis',
            'variable':variables,
            'pressure_level':pressure_levels,
            'year':f'{year}',
            'month':f'{month:01}',
            'day':dayslist,
            'time':[
                 '00:00','01:00','02:00',
                 '03:00','04:00','05:00',
                 '06:00','07:00','08:00',
                 '09:00','10:00','11:00',
                 '12:00','13:00','14:00',
                 '15:00','16:00','17:00',
                 '18:00','19:00','20:00',
                 '21:00','22:00','23:00'
             ],
             'format':'netcdf',
             'grid'  : grid
        },
        f'{directory}era_5_m{month:01}_y{year}_data.nc')

def year_loop(capi,args,year):
    months = np.arange(1,13)
    for i in months:
        month_downloader(capi,i,year,args.vars,args.plevs,args.grid,args.path)


###Main part of the code 
parser = argparse.ArgumentParser(description="Download Hourly ERA5 Pressure Level Variables")
parser.add_argument("startyear",     type=int, help="Starting year to download")
parser.add_argument("endyear",       type=int, help="Ending year to download")
parser.add_argument("-path",         type=str, default="",metavar='path',help="Path to write the files to")
parser.add_argument("--np",          type=int, default=1, metavar='num_procs',help='Number of processors to use to download. Np should be <= number of years to download (default=1)')
parser.add_argument("--vars",        type=str, metavar='var', nargs='+' , default='temperature',help='Variable or variables to download (default is temperature') 
parser.add_argument("--plevs",       type=str, metavar='plev',nargs='+', default='1000',help='Pressure levels to download in hPa') 
parser.add_argument("--grid",        type=str, nargs=2 , default=[1.0,1.0], metavar=('gridx', 'gridy'), help='Resolution of ERA5 data in the x direction in degrees (default=1.0)')

args = parser.parse_args() 

years = np.arange(args.startyear,args.endyear+1)

func = partial(year_loop,c,args)
pool = Pool(args.np)
pool.map(func,years)
pool.close()
```
