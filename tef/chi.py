import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime
import sys


ds = xroms.open_netcdf(sys.argv[1], 
                      chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(370,375)
etaslice=slice(140,145)
tslice = slice(0,len(ds.ocean_time))

dsdz = grid.derivative(ds.salt, 'Z', boundary = 'extend')

#salinity variance dissipation - denoted by chi.
chi = 2*(ds.AKs*(dsdz**2)).isel(ocean_time = tslice, 
                                eta_rho = etaslice, 
                                xi_rho = xislice) 

dV = (ds.dx*ds.dy*ds.dz).isel(ocean_time = tslice, 
                              eta_rho = etaslice, 
                              xi_rho = xislice)
V = dV.sum(dim = ['s_rho', 'eta_rho', 'xi_rho'])

#Interpolate to get chi on the rho points 
chiint = grid.interp(chi, 'Z')
chivint = (chiint*dV).sum(dim = ['s_rho', 'eta_rho', 'xi_rho'])
chivint = chivint.rename('chi')

chivint.attrs['Description'] = 'Salinity variance dissipation'
chivint.attrs['Author'] = 'Dylan Schlichting'
chivint.attrs['Created'] = datetime.now().isoformat()
chivint.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

print('saving chi')
chivint.to_netcdf('../outputs/mixing/dissipation_basecase.nc')