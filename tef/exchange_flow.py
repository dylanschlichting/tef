import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime

#Comment the next four lines out if not using histograms_divided.py 
# volds = xr.open_mfdataset('../outputs/histograms/Divided_histograms/vol/1001/*.nc')
# saltds = xr.open_mfdataset('../outputs/histograms/Divided_histograms/salt/1001/*.nc')
# ssaltds = xr.open_mfdataset('../outputs/histograms/Divided_histograms/ssquare/1001/*.nc')
# svards = xr.open_mfdataset('../outputs/histograms/Divided_histograms/svar/1001/*.nc')

# ds = xr.merge([volds, saltds, ssaltds, svards])
ds = xr.open_mfdataset('../analysis/2d/z/*.nc')

print('Calculating exchange flow')
#Volume Flux - need to apply divergence here. Sign convention is EN+
#so the net tracer flux is W - E + S - N; Drawing a picture helps too. 
Qnet = (ds.QWh-ds.QEh+ds.QSh-ds.QNh)
Qnet.name = 'Qnet'
Qout = (Qnet.where(Qnet<0).sum('salt_bin'))
Qout.name = 'Qout'
Qin = (Qnet.where(Qnet>0).sum('salt_bin'))
Qin.name = 'Qin'

#Salt Flux
Qsnet = (ds.QsWh-ds.QsEh+ds.QsSh-ds.QsNh)
Qsnet.name = 'Qsnet'
Qsout = (Qsnet.where(Qsnet<0).sum('salt_bin'))
Qsout.name = 'Qsout'
Qsin = (Qsnet.where(Qsnet>0).sum('salt_bin'))
Qsin.name = 'Qsin'

#Inflowing and outflowing salinities: salt flux divided by volume flux = salt 
sin = (Qsin/Qin)
sin.name = 'sin'
sout = (Qsout/Qout)
sout.name = 'sout'

#Calculate the different tracer advections.
voladv = (Qin+Qout)
voladv.name = 'voladv'
#Need to fill NaNs because the flow can be unidirectional causing division by zero.
saltadv = (Qin*sin).fillna(0)+(Qout*sout).fillna(0) 
saltadv.name = 'saltadv'

#Salt square
Qssnet = (ds.QssWh-ds.QssEh+ds.QssSh-ds.QssNh)
Qssnet.name = 'Qssnet'

Qssout = (Qssnet.where(Qssnet<0).sum('salt_bin'))
Qssout.name = 'Qssout'
Qssin = (Qssnet.where(Qssnet>0).sum('salt_bin'))
Qssin.name = 'Qssin'

#Inflowing and outflowing salinities squared
ssin = (Qssin/Qin)
ssin.name = 'ssin'
ssout = (Qssout/Qout)
ssout.name = 'ssout'

ssaltadv = (Qin*ssin).fillna(0)+(Qout*ssout).fillna(0)
ssaltadv.name = 'ssaltadv'

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

ds = xr.merge([Qnet, Qin, Qout, Qsnet, Qsin, Qsout, Qssnet, Qssin, Qssout, 
               Qsvarnet, Qsvarin, Qsvarout, sin, sout, ssin, ssout, svarin,
               svarout, voladv, saltadv, ssaltadv, svaradv], compat = 'override')

ds.attrs['Description'] = 'Exchange flow Dataset'
ds.attrs['Author'] = 'Dylan Schlichting'
ds.attrs['Created'] = datetime.now().isoformat()

ds.to_netcdf('../analysis/2d/tef_nested_10min_xi_50_250_eta_150_350_z.nc')
print('netcdf saved')
