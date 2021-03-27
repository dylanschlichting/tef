import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime

paths = ['/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00001.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00002.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00003.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00004.nc',
         '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00005.nc',
        ]
print('loading data')         
ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(50,251)
etaslice=slice(150,351)

dsdz = grid.derivative(ds.salt, 'Z', boundary = 'extend')

#salinity variance dissipation - denoted by chi.
chi = 2*(ds.AKs*(dsdz**2)).isel(eta_rho = etaslice, 
                                xi_rho = xislice) 

dV = (ds.dx*ds.dy*ds.dz).isel(eta_rho = etaslice, 
                              xi_rho = xislice)
V = dV.sum(dim = ['s_rho', 'eta_rho', 'xi_rho'])

#Interpolate to get chi on the rho points 
chiint = grid.interp(chi, 'Z')
chivint = (chiint*dV).sum(dim = ['s_rho', 'eta_rho', 'xi_rho'])
chivint = chivint.rename('chi')

print('saving chi')
chivint.to_netcdf('../outputs/mixing/dissipation_nested_10min_xi50250_eta150350.nc')

chivint.attrs['Description'] = 'Salinity variance dissipation'
chivint.attrs['Author'] = 'Dylan Schlichting'
chivint.attrs['Created'] = datetime.now().isoformat()
chivint.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

print('nc saved')