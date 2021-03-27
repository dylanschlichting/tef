'''
Script to bin the volume and total salt content of a control volume in salinity coordinates. This is used to calculate entrainment,
as described in Wang et al. (2017) JPO. 
'''

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

import warnings
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.dates import DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 12})

import warnings
warnings.filterwarnings("ignore")

paths = glob.glob('../../../dylan.schlichting/TXLA_Outputs/parent/2010/ocean_his_00*.nc')

ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(260,381) 
etaslice=slice(47,149)

dV = ds.dV.isel(eta_rho = etaslice, xi_rho = xislice)
dV.attrs = ''
dV.name = 'dV'

saltbins = np.linspace(0,40,501) #number of salinity bins

salt = ds.salt.isel(xi_rho = xislice, eta_rho = etaslice)
salt.attrs = ''
salt.name = 'salt'

#subset the data by loading it daily and saving to .nc file. This will avoid memory problems. 
dates = np.arange('2010-01-01', '2011-01-01', dtype = 'datetime64[D]')

for d in range(len(dates)):
    saltl = salt.sel(ocean_time = str(dates[d])).load()
    dVl = dV.sel(ocean_time = str(dates[d])).load()
    
    Vh = histogram(saltl, 
               bins = [saltbins],
               weights = dVl,
               dim = ['s_rho', 'eta_rho', 'xi_rho'])
    Vh.name = 'dV_saltcoord'
    
    path = '/scratch/user/dylan.schlichting/tef/largecv/entrainment/V_histogram_2010_%s.nc' %d
    Vh.to_netcdf(path)
    
    dVs = dVl*saltl
    dVs.name = 'salt_content'

    Vsh = histogram(saltl, 
                    bins = [saltbins],
                    weights = dVs,
                    dim = ['s_rho', 'eta_rho', 'xi_rho'])
    Vsh.name = 'dVs_saltcoord'
    
    path = '/scratch/user/dylan.schlichting/tef/largecv/entrainment/Vs_histogram_2010_%s.nc' %d
    Vsh.to_netcdf(path)
