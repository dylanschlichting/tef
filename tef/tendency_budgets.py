import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime

ds = xroms.open_netcdf('/d1/shared/shelf_ho_0_dh_0_vwind_0_his.nc', 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(370,375)
etaslice=slice(140,145)
tslice = slice(0,len(ds.ocean_time))

#dVdt
V = ((ds.dx*ds.dy*ds.dz)).isel(ocean_time = tslice,
                             eta_rho = etaslice, 
                             xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
V = V.chunk(3) #
dVdt = V.differentiate('ocean_time', datetime_unit = 's')
dVdt.name = 'dVdt'

#d(salt)/dt
dsV = ((ds.dx*ds.dy*ds.dz*ds.salt)).isel(ocean_time = tslice,
                                         eta_rho = etaslice, 
                                         xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
dsV = dsV.chunk({'ocean_time':1})
dsVdt = dsV.differentiate('ocean_time', datetime_unit = 's')
dsVdt.name = 'dsVdt'

#d(salt**2)/dt
dssV = ((ds.dx*ds.dy*ds.dz*ds.salt**2)).isel(ocean_time = tslice,
                                         eta_rho = etaslice, 
                                         xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
dssV = dssV.chunk(3)
dssVdt = dssV.differentiate('ocean_time', datetime_unit = 's')
dssVdt.name = 'dssVdt'

#d(svar)/dt
dV = (ds.dx*ds.dy*ds.dz).isel(ocean_time = tslice,
                              eta_rho = etaslice, 
                              xi_rho = xislice)
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
salt = ds.salt.isel(ocean_time = tslice,
                    eta_rho = etaslice, 
                    xi_rho = xislice)

sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])

sbar = sbar.chunk({'ocean_time':1})
salt = salt.chunk({'ocean_time':1})
dV = dV.chunk({'ocean_time':1})

svar = (((salt-sbar)**2)*(dV)).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
dsvardt = svar.differentiate('ocean_time', datetime_unit = 's')
dsvardt.name = 'dsvardt'

ds = xr.merge([dVdt, dsVdt, dssVdt, dsvardt], compat = 'override')

ds.attrs['Description'] = 'Tendency Budget Terms'
ds.attrs['Author'] = 'Dylan Schlichting'
ds.attrs['Created'] = datetime.now().isoformat()
ds.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

ds.to_netcdf('../outputs/tendencies/tendencies_basecase.nc')