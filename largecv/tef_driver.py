''' 
Driver script to compute transport weighted histograms for an arbitrary control volume
of TXLA model output. Currently outfitted for the parent grid. 
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
from dask.distributed import Client

client = Client(processes=False)

import warnings
warnings.filterwarnings("ignore")

#Open model output for an entire year, e.g. 2010
paths = glob.glob('../../../dylan.schlichting/TXLA_Outputs/parent/2010/ocean_his_00*.nc')
print('loading data')

ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(260,380) 
etaslice=slice(47,148)

print('Isolating control volume and computing tracer fluxes')

#Salinity for the four control surfaces 
saltda = functions.salt_cv(ds, grid, xislice, etaslice)

#Volume fluxes for the four control surfaces 
Qda = functions.volume_flux(ds, xislice, etaslice) 

#Salinity and salinity squared transport
Qsda, Qssda = functions.salt_flux(saltda, Qda) 

#Salinity variance and variance transport 
svarda,Qsvarda = functions.Qcsvar_faces(ds, grid, saltda, Qda, xislice, etaslice) 

saltbins = np.linspace(0,40,501) #number of salinity bins

#Calculate histograms of tracer transport for the control volume during summer.
dates = np.arange('2010-01-01', '2011-01-01', dtype = 'datetime64[D]')
# dates = np.arange('2010-12-22', '2011-01-01', dtype = 'datetime64[D]')

for d in range(len(dates)):
    saltdal = saltda.sel(ocean_time = str(dates[d])).load()
    Qdal = Qda.sel(ocean_time = str(dates[d])).load()
    
    Qh_da = functions.volflux_hist(saltbins, saltdal, Qdal)
    
    path = '/scratch/user/dylan.schlichting/tef/largecv/histograms/vol/Q_histogram_2010_%s.nc' %d
    Qh_da.to_netcdf(path)
    
    Qsdal = Qsda.sel(ocean_time = str(dates[d])).load()
    
    Qsh_da = functions.saltflux_hist(saltbins, saltdal, Qsdal)
    path = '/scratch/user/dylan.schlichting/tef/largecv/histograms/salt/Qs_histogram_2010_%s.nc' %d
    Qsh_da.to_netcdf(path)
    
    Qsvardal = Qsvarda.sel(ocean_time = str(dates[d])).load()
    
    Qsvarh_da = functions.svarflux_hist(saltbins, saltdal, Qsvardal)
    path = '/scratch/user/dylan.schlichting/tef/largecv/histograms/svar/Qsvar_histogram_2010_%s.nc' %d
    Qsvarh_da.to_netcdf(path)
    
# Qsvarh_da = functions.svarflux_hist(saltbins, saltda.sel(ocean_time = '2010'), Qsvarda.sel(ocean_time = '2010'))
# Qsvarh_da.attrs['Description'] = 'Svar transport histograms'
# Qsvarh_da.attrs['Author'] = 'Dylan Schlichting'
# Qsvarh_da.attrs['Created'] = datetime.now().isoformat()
# Qsvarh_da.attrs['Grid Slice'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

# print('Saving salinity variance histograms')

# dates = np.arange('2010-01', '2011-01-01', dtype = 'datetime64[D]')

# for d in range(len(dates)):
#     Qsvarh_slice = Qsvarh_da.sel(ocean_time = str(dates[d]))
    
#     path = '/scratch/user/dylan.schlichting/tef/largecv/histograms/svar/Qsvar_histogram_2010_%s.nc' %d
#     Qsvarh_slice.to_netcdf(path)
# Qsvarh_da.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/histograms/Qsvar_histogram_2010-Jan.nc')

# #Save salinity histograms
# Qsh_da = functions.saltflux_hist(saltbins, saltda, Qsda)
# Qsh_da.attrs['Description'] = 'Salt transport histograms'
# Qsh_da.attrs['Author'] = 'Dylan Schlichting'
# Qsh_da.attrs['Created'] = datetime.now().isoformat()
# Qsh_da.attrs['Grid Slice'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

# print('Saving salinity histograms')
# Qsh_da.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/histograms/Qs_histogram_2010.nc', engine = 'h5netcdf')

# # #Save volume histograms
# Qh_da = functions.volflux_hist(saltbins, saltda, Qda)
# Qh_da.attrs['Description'] = 'Volume transport histograms'
# Qh_da.attrs['Author'] = 'Dylan Schlichting'
# Qh_da.attrs['Created'] = datetime.now().isoformat()
# Qh_da.attrs['Grid Slice'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

# print('Saving salinity histograms')
# Qh_da.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/histograms/Qh_histogram_2010.nc', engine = 'h5netcdf')
# Qh_rechunked = Qh_da.chunk({'ocean_time':30})

# months, datasets = zip(*Qh_rechunked.groupby("ocean_time.month"))
# paths = ["/home/dylan/tef/variability/histograms/summer/Qh_histograms_%s.nc" % (m-1) for m in months]

# print('Saving volume histograms')
# xr.save_mfdataset(datasets, paths, engine = 'h5netcdf', mode = 'w')

# ds = xr.open_mfdataset('/scratch/user/dylan.schlichting/tef/largecv/histograms/*.nc') #Change the path if necessary
# ds = functions.exchange_flow(ds)

# ds.attrs['Description'] = 'Exchange flow Dataset'
# ds.attrs['Author'] = 'Dylan Schlichting'
# ds.attrs['Created'] = datetime.now().isoformat()
# ds.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)

# ds.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/budgets/tef_hourly_2010.nc', engine = 'h5netcdf')
# print('Exchange flow saved')
