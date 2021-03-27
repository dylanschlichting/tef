'''
This script calcualtes total exchange flow parameters for volume flux and salinity variance flux through a 3D control volume.
This only differs from 'exchange_flow.py' because it omits salinity and salinity squared fluxes. This is for testing out different normalization schemes.
'''

import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime


ds = xr.open_mfdataset('../analysis/normalized/*.nc')

print('Calculating exchange flow')
#Volume Flux - need to apply divergence here. Sign convention is EN+
#so the net tracer flux is W - E + S - N; Drawing a picture helps too. 
Qnet = (ds.QWh-ds.QEh+ds.QSh-ds.QNh)
Qnet.name = 'Qnet'
Qout = (Qnet.where(Qnet<0).sum('salt_bin'))
Qout.name = 'Qout'
Qin = (Qnet.where(Qnet>0).sum('salt_bin'))
Qin.name = 'Qin'


#Calculate the different tracer advections.
voladv = (Qin+Qout)
voladv.name = 'voladv'

#Salinity variance
Qsvarnet = (ds.QsvarWh-ds.QsvarEh+ds.QsvarSh-ds.QsvarNh)
Qsvarnet.name = 'Qsvarnet'

Qsvarout = (Qsvarnet.where(Qsvarnet<0).sum('salt_bin'))
Qsvarout.name = 'Qsvarout'
Qsvarin = (Qsvarnet.where(Qsvarnet>0).sum('salt_bin'))
Qsvarin.name = 'Qsvarin'

svarin = (Qsvarin/Qin)
svarin.name = 'svarin'
svarout = (Qsvarout/Qout)
svarout.name = 'svarout'

svaradv = ((Qin*svarin).fillna(0)+(Qout*svarout).fillna(0))
svaradv.name = 'svaradv'

ds = xr.merge([Qnet, Qin, Qout, Qsvarnet, Qsvarin, Qsvarout, svarin,
               svarout, voladv, svaradv], compat = 'override')

ds.attrs['Description'] = 'Exchange flow Dataset'
ds.attrs['Author'] = 'Dylan Schlichting'
ds.attrs['Created'] = datetime.now().isoformat()

ds.to_netcdf('../analysis/normalized/tef_nested_10min_xi_50_250_eta_150_350_znormalized.nc')
print('netcdf saved')
