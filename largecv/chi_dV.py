'''
Mixing per unit volume in TS space. 
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

saltbins = np.linspace(0,40,501)
tempbins = np.linspace(0,40,501)

dsdz = grid.derivative(ds.salt, 'Z', boundary = 'extend') 

#salinity variance dissipation - denoted by chi.
chi = 2*(ds.AKs*(dsdz**2)).isel(eta_rho = etaslice, 
                                xi_rho = xislice) 

dV = (ds.dx*ds.dy*ds.dz).isel(eta_rho = etaslice, 
                              xi_rho = xislice)

#Interpolate to get chi on the rho points (from the w points)
chiint = grid.interp(chi, 'Z')
chivint = chiint*dV
chi = chivint.rename('chi')

chi.attrs = ''

salt = ds.salt.isel(xi_rho = xislice, eta_rho = etaslice)
salt.attrs = ''

temp = ds.temp.isel(xi_rho = xislice, eta_rho = etaslice)
temp.attrs = ''

chiint.name = 'chi'

dV = ds.dV.isel(xi_rho = xislice, eta_rho = etaslice)
chi_dV = chi/dV #mixing per unit volume

dates = np.arange('2010-06-01', '2011-08-31', dtype = 'datetime64[D]')

for d in range(len(dates)):
    chih = histogram(salt.sel(ocean_time = str(dates[d])).load(), 
                     temp.sel(ocean_time = str(dates[d])).load(),
                     bins = [saltbins, tempbins], 
                     weights = chi_dV.sel(ocean_time = str(dates[d])).load(),
                     dim = ['s_rho', 'eta_rho', 'xi_rho'])
    chih.name = 'chi'
    path = '/scratch/user/dylan.schlichting/tef/largecv/budgets/ts/chidv/chi_unitvolume_hourly_2010_highres_%s.nc' %d
    chih.to_netcdf(path, mode = 'w')