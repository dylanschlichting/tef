import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from scipy import signal
import glob
import functions #the .py file that contains all the relevant functions
from datetime import datetime
# from dask.distributed import Client

# client = Client(processes=False)

import warnings
warnings.filterwarnings("ignore")
#Open model output for an entire year, e.g. 2010
paths = glob.glob('../../../dylan.schlichting/TXLA_Outputs/parent/2010/ocean_his_00*.nc')
print('loading data')

ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(260,381) 
etaslice=slice(47,149)

ds = xroms.open_mfnetcdf(paths, 
                       chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

dt = 3600
tendencies = functions.tendencies(ds, xislice, etaslice, dt)
tendencies.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/budgets/storage_hourly_2010.nc', engine = 'h5netcdf')