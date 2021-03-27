'''
Computes the dissipation of vertical salinity variance (chi) for roms model output. This script is designed to chi*dV at each grid cell and time step in salinity coordiantes. To get the volume integrated chi, just sum over the salinity bins. We save the output this way so you can access chi in both salinity and time coordinates. 

This script is modified to calculate a 2D histogram by depth as well. 
'''

import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime

#10 minute output of nested child grid 
paths = ['/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00001.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00002.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00003.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00004.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00005.nc',
        ]
#1 hour output of nested child grid 
# paths = ['/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00001.nc',
#          '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00002.nc',
#          '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00003.nc',
#         ]
print('loading data')         
ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

#Remove the dataset attributes or saving chi to a netcdf file will yield an error because the xgcm metrics have incompatible symbols. This is an annoying but persistant bug. 
ds.attrs = ''
xislice=slice(50,251)
etaslice=slice(150,351)

dsdz = grid.derivative(ds.salt, 'Z', boundary = 'extend')

#salinity variance dissipation - denoted by chi.
chi = 2*(ds.AKs*(dsdz**2)).isel(eta_rho = etaslice, 
                                xi_rho = xislice) 

dV = (ds.dx*ds.dy*ds.dz).isel(eta_rho = etaslice, 
                              xi_rho = xislice)

#Interpolate to get chi on the rho points (from the w points)
chiint = grid.interp(chi, 'Z')
chivint = chiint #note we don't multiply by grid cell volume because we want units of (g/kg)^2 s^{-1}, which is the salinity variance transport/dV.
#So if we volume integrated this and divided by grid cell volume it would be redundant 
chi = chivint.rename('chi')

salt = ds.salt.isel(xi_rho = xislice, eta_rho = etaslice)
salt.attrs = ''
saltbins = np.linspace(0,40,201)
depthbins = np.linspace(-65,0, 51) #max depth or this control volume. Remember to change!!!

chiint.name = 'chi'

#Load in the depths 
depths = ds.z_rho.isel(eta_rho = etaslice, 
                   xi_rho = xislice) 
depths.name = 'depth'
depths.attrs = ''

chih = histogram(salt,
                 depths,
                 bins = [saltbins, depthbins], 
                 weights = chi,
                 dim = ['s_rho', 'eta_rho', 'xi_rho']
                )
chih.name = 'chi'
#Set some attributes to label the variation of chi here
# chih.attrs = 'chi/dV'
chih.to_netcdf('../analysis/normalized/chi/dissipation_histogram_10min_xi50250_eta150350_notintegrated.nc')
