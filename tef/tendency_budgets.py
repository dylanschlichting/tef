import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime

paths = ['/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00001.nc',
          '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00002.nc',
          '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00003.nc',
          # '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00004.nc',
          # '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00005.nc',
         ]
#paths = ['/scratch/user/dylan.schlichting/shelfstrait/project/shelf_base_case_his.nc']
print('loading data')  
ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

# xislice=slice(300,311)
# etaslice=slice(300,311)
xislice=slice(50,251) 
etaslice=slice(150,351)

#dVdt
V = ((ds.dx*ds.dy*ds.dz)).isel(
                             eta_rho = etaslice, 
                             xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
#V = V.chunk(3) #
dVdt = V.diff('ocean_time')/3600 #divide by 600 bc output frequency is 10 min,
#60 seconds per min X 10 min = 600
dVdt.name = 'dVdt'

#d(salt)/dt
dsV = ((ds.dx*ds.dy*ds.dz*ds.salt)).isel(
                                         eta_rho = etaslice, 
                                         xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
#dsV = dsV.chunk({'ocean_time':3})
dsVdt = dsV.diff('ocean_time')/3600
dsVdt.name = 'dsVdt'

#d(salt**2)/dt
dssV = ((ds.dx*ds.dy*ds.dz*ds.salt**2)).isel(
                                         eta_rho = etaslice, 
                                         xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
#dssV = dssV.chunk(3)
dssVdt = dssV.diff('ocean_time')/3600
dssVdt.name = 'dssVdt'

#d(svar)/dt
dV = (ds.dx*ds.dy*ds.dz).isel(
                              eta_rho = etaslice, 
                              xi_rho = xislice)
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
salt = ds.salt.isel(
                    eta_rho = etaslice, 
                    xi_rho = xislice)

sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])

#sbar = sbar.chunk({'ocean_time':3})
#salt = salt.chunk({'ocean_time':3})
#dV = dV.chunk({'ocean_time':3})

svar = (((salt-sbar)**2)*(dV)).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
dsvardt = svar.diff('ocean_time')/3600
dsvardt.name = 'dsvardt'

dVdt.attrs = ''
dsVdt.attrs = ''
dssVdt.attrs = ''
dsvardt.attrs = ''

ds1 = xr.merge([dVdt, dsVdt, dssVdt, dsvardt], compat = 'override')

ds.attrs['Description'] = 'Tendency Budget Terms'
ds.attrs['Author'] = 'Dylan Schlichting'
ds.attrs['Created'] = datetime.now().isoformat()
ds.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

print('saving data')
ds1.to_netcdf('../outputs/tendencies/tendencies_nested_hourly_xi_50_250_eta_150_350.nc', format = 'NETCDF4_CLASSIC')
