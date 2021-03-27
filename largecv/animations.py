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

import warnings
warnings.filterwarnings("ignore")
from celluloid import Camera
from IPython.display import HTML

paths = glob.glob('../../../dylan.schlichting/TXLA_Outputs/parent/2010/ocean_his_00*.nc')

ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(260,381) 
etaslice=slice(47,149)


land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                edgecolor='face',
                                facecolor=cfeature.COLORS['land'])
states_provinces = cfeature.NaturalEarthFeature(
                   category='cultural',
                   name='admin_1_states_provinces_lines',
                   scale='10m',
                   facecolor='none')

plt.rcParams.update({'font.size': 16})
from matplotlib import animation

fig = plt.figure(figsize=(8,4), dpi = 100)
camera = Camera(fig)

ax = fig.add_axes([0.06, 0.01, 0.93, 0.95], projection=ccrs.PlateCarree(central_longitude=-92.0))

lon_rho = ds.salt['lon_rho'].isel(eta_rho = etaslice, xi_rho = xislice)
lat_rho = ds.salt['lat_rho'].isel(eta_rho = etaslice, xi_rho = xislice)

sal = ds.salt.isel(s_rho = -1, eta_rho = etaslice, xi_rho = xislice).sel(ocean_time = slice('2010-06', '2010-08'))

def animate(i):
    mappable = ax.pcolormesh(lon_rho, lat_rho, sal[i], 
                             cmap = cmo.haline, 
                             transform = cartopy.crs.PlateCarree(),
                             vmin = 20, vmax = 38)
    ax.set_extent([-96, -91, 27.5, 30.5], ccrs.PlateCarree())
    ax.add_feature(land_10m, facecolor='0.8')
    ax.coastlines(resolution='10m')  
    ax.add_feature(states_provinces, )
    ax.add_feature(cfeature.BORDERS, linestyle='-', )
    ax.add_feature(cartopy.feature.RIVERS, linewidth = 2)
    ax.set_aspect('auto')

    cax = fig.add_axes([0.09, 0.91, 0.32, 0.02]) 
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cb.set_label(r'Surface salinity [g$\cdot$kg$^{-1}$]', color='0.2')
    cb.ax.tick_params(length=2, color='0.2', labelcolor='0.2')
    
    #Plot isobaths and label the contours

    ax.tick_params(axis='y', labelsize=13)    
    return mappable
    
anim = animation.FuncAnimation(fig, animate, frames = 200)
plt.close()
anim.save('animation_sal.mp4', writer = 'ffmpeg', fps = 60)