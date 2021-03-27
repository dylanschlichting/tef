'''
Contains functions used to compute transport weighted histograms of a tracer, c for a control volume specified by slices
in the xi and eta direction. The first set of functions are designed for flows with no normalization scheme so that the 
user can see how closed the tracer budgets are. 
'''

import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from scipy import signal
import glob
from datetime import datetime

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib.pyplot as plt

#Start of TEF related functions. Make note to name all Xarray DataArray to avoid annoying
#automatically generated names, which are usually really long and undesirable. 
#---------------------------
def volume_flux(ds, xislice, etaslice):
    '''
Computes the boundary volume transport of a control volume for ROMS model output. 
-----
Input: 
ds - xarray roms dataset
grid - xgcm grid of roms output
xislice - slice object of desired xi points
etaslice -slice object of desired eta points
-----
Output:
Qds: Xarray Dataset of volume transport at the four horizontal control surfaces. 
    '''
    Qu = ds.dz_u*ds.dy_u*ds.u
    Qv = ds.dz_v*ds.dx_v*ds.v

    Qu = Qu.sel(eta_rho = etaslice, xi_u = slice(xislice.start-1, xislice.stop))
    Qv = Qv.sel(eta_v = slice(etaslice.start-1, etaslice.stop), xi_rho = xislice)

    QW = Qu.isel(xi_u = 0) #West face of control volume
    QE = Qu.isel(xi_u = -1) #East face of control volume
    QN = Qv.isel(eta_v = -1) #North face of control volume
    QS = Qv.isel(eta_v = 0) #South face of control volume
    
    QW.name = 'QW'
    QE.name = 'QE'
    QN.name = 'QN'
    QS.name = 'QS'
    
    Qda = xr.merge([QW, QE, QN, QS], compat='override') #volume flux data array >> Qda
    return Qda

def salt_cv(ds, grid, xislice, etaslice):
    '''
Computes the boundary salinity of a control volume for ROMS model output. 
-----
Input: 
ds - xarray roms dataset
grid - xgcm grid of roms output
xislice - slice object of desired xi points
etaslice -slice object of desired eta points
-----
Output:
saltda: Xarray DataArray of salinity at the four horizontal control surfaces. 
    '''
    su = grid.interp(ds.salt, 'X')
    sv = grid.interp(ds.salt, 'Y')

    #Align the tracer at the control volume boundaries to account for the u/v points:
    #subtract 1 point from the start of the u/v so there are no leaky corners
    su = su.sel(eta_rho = etaslice, xi_u = slice(xislice.start-1, xislice.stop)) 
    sv = sv.sel(eta_v = slice(etaslice.start-1, etaslice.stop), xi_rho = xislice)

    sW = su.isel(xi_u = 0) #West face of control volume
    sE = su.isel(xi_u = -1) #East face of control volume
    sN = sv.isel(eta_v = -1) #North face of control volume
    sS = sv.isel(eta_v = 0) #South face of control volume
   
    #DataArray Metadata
    sW.name = 'sW'
    sE.name = 'sE'
    sN.name = 'sN'
    sS.name = 'sS'
    
    saltda = xr.merge([sW, sE, sN, sS], compat='override') #salt data array aka saltda
    return saltda

def salt_flux(saltda, Qda):
    '''
Computes the boundary salinity transport of a control volume for ROMS model output. 
-----
Input: 
saltda - Xarray DataArray of the salinity at the boundaries
Qda - Xarray DataArray of the voume flux at the boundaries
-----
Output:
Qsda: Xarray DataArray of salinity transport at the four horizontal control surfaces. 
Qssda: Xarray DataArray of salinity squared transport at the four horizontal control surfaces. 
    '''
    QsW = saltda.sW*Qda.QW
    QsE = saltda.sE*Qda.QE
    QsN = saltda.sN*Qda.QN
    QsS = saltda.sS*Qda.QS
    
    QsW.name = 'QsW'
    QsE.name = 'QsE'
    QsN.name = 'QsN'
    QsS.name = 'QsS'
    
    QssW = (saltda.sW)**2*Qda.QW
    QssE = (saltda.sE)**2*Qda.QE
    QssN = (saltda.sN)**2*Qda.QN
    QssS = (saltda.sS)**2*Qda.QS
    
    QssW.name = 'QssW'
    QssE.name = 'QssE'
    QssN.name = 'QssN'
    QssS.name = 'QssS'
    
    Qsda = xr.merge([QsW, QsE, QsN, QsS], compat='override')
    Qssda = xr.merge([QssW, QssE, QssN, QssS], compat='override')
    
    return Qsda, Qssda

def Qcsvar_faces(ds, grid, saltda, Qda, xislice, etaslice):
    '''
Computes the boundary fluxes of salinity variance for a control volume of ROMS output. 
-----
Input: 
ds - xarray dataset
grid - xgcm grid
xislice - slice object of desired xi grid points
etaslice - slice object of desired eta grid points
saltda - salinity at each face of the control volume
Qda - volume flux at each face of the control volume
-----
Output:
Qsvarda: salinity variance transport at each face of the control volume
svarda: salinity variance at each face of the control volume
    '''
    
    #Note we need to adjust the slices in the xi and eta slices for the salinity variance budget.
    #Since we are computing the volume averaged variance, add 1 point to the stopping slices so 
    #dsvardt matches up with the fluxes. Drawing a picture of the grid is really helpful for this. 
#     xislice = slice(xislice.start, xislice.stop+1)
#     etaslice = slice(etaslice.start, etaslice.stop+1)
    xislice=slice(260,381) 
    etaslice=slice(47,149)
    
    dV = (ds.dx*ds.dy*ds.dz).isel(eta_rho = etaslice, 
                                  xi_rho = xislice)
 
    V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
   
    salt = ds.salt.isel(eta_rho = etaslice, 
                        xi_rho = xislice)

    sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])

    svarW = ((saltda.sW-sbar)**2)
    svarE = ((saltda.sE-sbar)**2)
    svarN = ((saltda.sN-sbar)**2)
    svarS = ((saltda.sS-sbar)**2)

    QsvarW = Qda.QW*svarW
    QsvarE = Qda.QE*svarE
    QsvarN = Qda.QN*svarN
    QsvarS = Qda.QS*svarS
    
    QsvarW.name = 'QsvarW'
    QsvarE.name = 'QsvarE'
    QsvarN.name = 'QsvarN'
    QsvarS.name = 'QsvarS'

    svarW.name = 'svarW'
    svarE.name = 'svarE'
    svarN.name = 'svarN'
    svarS.name = 'svarS'
    
    svarda = xr.merge([svarW, svarE, svarN, svarS], compat='override')
    Qsvarda = xr.merge([QsvarW, QsvarE, QsvarN, QsvarS], compat='override')
    
    return svarda,Qsvarda

#Compute transport weighted histograms of the tracers in salinity coordinates.
#Xhistogram has this very annoying feature where it automatically names the bins, which
#we don't want since the salinity bins are constant. To save time when saving the .nc files,
#I'll drop the extra coordiantes and add the salinity bins at the end of the resulting dataset.
#See link for info -  https://xhistogram.readthedocs.io/en/latest/_modules/xhistogram/xarray.html

def svarflux_hist(saltbins, saltda, Qsvarda):
    '''
Computes salinity variance transport weighted histograms for 3D control volume. Here, we consider the transport from four horizontal directions. Dependent on the number of salinity bins, and the salinity/salinity variance transport at the four control surfaces.
-------
Input:
saltbins - numpy array of number of salinity bins, e.g. np.linspace(0,40,101)
saltda - Xarray DataArray of the salinity at the control surfaces
Qsvarda - Xarray DataArray of the salinity variance transport at the control surfaces
-------
Output: 
Qsvarh_da - Xarray DataArray of salinity variance transport weighted histograms
    '''
    QsvarWh = histogram(saltda.sW, 
                        bins = [saltbins], 
                        weights = Qsvarda.QsvarW,
                        dim = ['s_rho', 'eta_rho'])
    
    QsvarWh = QsvarWh.rename({'sW_bin':'salt_bin'})
    QsvarWh.name = 'QsvarWh'
    
    QsvarEh = histogram(saltda.sE, 
                        bins = [saltbins], 
                        weights = Qsvarda.QsvarE,
                        dim = ['s_rho', 'eta_rho'])
    
    QsvarEh = QsvarEh.rename({'sE_bin':'salt_bin'})
    QsvarEh.name = 'QsvarEh'
    
    QsvarNh = histogram(saltda.sN, 
                        bins = [saltbins], 
                        weights = Qsvarda.QsvarN,
                        dim = ['s_rho', 'xi_rho'])
    
    QsvarNh = QsvarNh.rename({'sN_bin':'salt_bin'})
    QsvarNh.name = 'QsvarNh'
    
    QsvarSh = histogram(saltda.sS, 
                        bins = [saltbins],
                        weights = Qsvarda.QsvarS,
                        dim = ['s_rho', 'xi_rho'])
    
    QsvarSh = QsvarSh.rename({'sS_bin':'salt_bin'})
    QsvarSh.name = 'QsvarSh'
    
    Qsvarh_da = xr.merge([QsvarWh, QsvarEh, QsvarNh, QsvarSh]) #Data array of salinity variance transport histograms
    return Qsvarh_da

def ssquaredflux_hist(saltbins, saltda, Qssda):
    '''
Computes salinity squared transport weighted histograms for 3D control volume. Here, we consider the transport from four horizontal directions. Dependent on the number of salinity bins, and the salinity squared transport at the four control surfaces.
-------
Input:
saltbins - numpy array of number of salinity bins, e.g. np.linspace(0,40,101)
saltda - Xarray DataArray of the salinity at the control surfaces
Qssda - Xarray DataArray of the salinity squared transport at the control surfaces
-------
Output: 
Qssh_da - Xarray DataArray of salinity squared transport weighted histograms
    '''
    QssWh = histogram(saltda.sW, 
                      bins = [saltbins], weights = Qssda.QssW,
                      dim = ['s_rho', 'eta_rho'])
    
    QssWh = QssWh.rename({'sW_bin':'salt_bin'})
    QssWh.name = 'QssWh'
    
    QssEh = histogram(saltda.sE, 
                      bins = [saltbins], 
                      weights = Qssda.QssE,
                      dim = ['s_rho', 'eta_rho'])
    
    QssEh = QssEh.rename({'sE_bin':'salt_bin'})
    QssEh.name = 'QssEh'
    
    QssNh = histogram(saltda.sN, 
                      bins = [saltbins], 
                      weights = Qssda.QssN,
                      dim = ['s_rho', 'xi_rho'])
    
    QssNh = QssNh.rename({'sN_bin':'salt_bin'})
    QssNh.name = 'QssNh'
    
    QssSh = histogram(saltda.sS, 
                      bins = [saltbins], 
                      weights = Qssda.QssS,
                      dim = ['s_rho', 'xi_rho'])
    
    QssSh = QssSh.rename({'sS_bin':'salt_bin'})
    QssSh.name = 'QssSh'
    
    Qssh_da = xr.merge([QssWh, QssEh, QssNh, QssSh]) #Data array of salinity squared transport histograms
    return Qssh_da

def saltflux_hist(saltbins, saltda, Qsda):
    '''
Computes salinity transport weighted histograms for 3D control volume. Here, we consider the transport from four horizontal directions. Dependent on the number of salinity bins, and the salinity transport at the four control surfaces.
-------
Input:
saltbins - numpy array of number of salinity bins, e.g. np.linspace(0,40,101)
saltda - Xarray DataArray of the salinity at the control surfaces
Qsda - Xarray DataArray of the salinity transport at the control surfaces
-------
Output: 
Qsh_da - Xarray DataArray of salinity transport weighted histograms
    '''
    QsWh = histogram(saltda.sW, 
                     bins = [saltbins], 
                     weights = Qsda.QsW,
                     dim = ['s_rho', 'eta_rho'])
    
    QsWh = QsWh.rename({'sW_bin':'salt_bin'})
    QsWh.name = 'QsWh'
    
    QsEh = histogram(saltda.sE, 
                     bins = [saltbins], 
                     weights = Qsda.QsE,
                     dim = ['s_rho', 'eta_rho'])
    
    QsEh = QsEh.rename({'sE_bin':'salt_bin'})
    QsEh.name = 'QsEh'
    
    QsNh = histogram(saltda.sN, 
                     bins = [saltbins], 
                     weights = Qsda.QsN,
                     dim = ['s_rho', 'xi_rho'])
    
    QsNh = QsNh.rename({'sN_bin':'salt_bin'})
    QsNh.name = 'QsNh'
    
    QsSh = histogram(saltda.sS, 
                     bins = [saltbins], 
                     weights = Qsda.QsS,
                     dim = ['s_rho', 'xi_rho'])
    QsSh = QsSh.rename({'sS_bin':'salt_bin'})
    QsSh.name = 'QsSh'
    
    Qsh_da = xr.merge([QsWh, QsEh, QsNh, QsSh], compat = 'override') #Data array of salinity transport histograms
    return Qsh_da

def volflux_hist(saltbins, saltda, Qda):
    '''
Computes volume transport weighted histograms for 3D control volume. Here, we consider the transport from four horizontal directions. Dependent on the number of salinity bins, and the salinity at the four control surfaces.
-------
Input:
saltbins - numpy array of number of salinity bins, e.g. np.linspace(0,40,101)
saltda - Xarray DataArray of the salinity at the control surfaces
Qda - Xarray DataArray of the volume transport at the control surfaces
-------
Output: 
Qh_da - Xarray DataArray of volume transport weighted histograms
    '''
    QWh = histogram(saltda.sW, 
                    bins = [saltbins], 
                    weights = Qda.QW,
                    dim = ['s_rho', 'eta_rho'])
    
    QWh = QWh.rename({'sW_bin':'salt_bin'})
    QWh.name = 'QWh'
    
    QEh = histogram(saltda.sE, 
                    bins = [saltbins], 
                    weights = Qda.QE,
                    dim = ['s_rho', 'eta_rho'])
    
    QEh = QEh.rename({'sE_bin':'salt_bin'})
    QEh.name = 'QEh'
    
    QNh = histogram(saltda.sN, 
                    bins = [saltbins], 
                    weights = Qda.QN,
                    dim = ['s_rho', 'xi_rho'])
    
    QNh = QNh.rename({'sN_bin':'salt_bin'})
    QNh.name = 'QNh'
    
    QSh = histogram(saltda.sS, 
                    bins = [saltbins], 
                    weights = Qda.QS,
                    dim = ['s_rho', 'xi_rho'])
    
    QSh = QSh.rename({'sS_bin':'salt_bin'})
    QSh.name = 'QSh'
    
    Qh_da = xr.merge([QWh, QEh, QNh, QSh], compat = 'override') #Data array of volume transport histograms
    return Qh_da

def exchange_flow(ds):
    '''
Computes the total exchange flow parameters define by MacCready (2011) JPO and MacCready et al. (2018).
Requires a dataset containing tracer transport weighted histograms of volume, salt, and salt variance.
Currently based on the sign method. 
    '''
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

    ds = xr.merge([Qnet, Qin, Qout, Qsnet, Qsin, Qsout, 
                   Qsvarnet, Qsvarin, Qsvarout, sin, sout, svarin,
                   svarout, voladv, saltadv, svaradv], compat = 'override')

    return ds

#End of direct TEF functions. Next, move into the tracer budgets
#----------------------------------------------------------------------------------------------
def tendencies(ds, xislice, etaslice, dt):
    V = ((ds.dx*ds.dy*ds.dz)).isel(eta_rho = etaslice, 
                                   xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
    #d(V)/dt
    dVdt = V.diff('ocean_time')/dt 
    dVdt.name = 'dVdt'

    #d(salt)/dt
    dsV = ((ds.dx*ds.dy*ds.dz*ds.salt)).isel(eta_rho = etaslice, 
                                             xi_rho = xislice).sum(dim = ['xi_rho', 's_rho', 'eta_rho'])
    dsVdt = dsV.diff('ocean_time')/dt
    dsVdt.name = 'dsVdt'

    #d(svar)/dt
    salt = ds.salt.isel(eta_rho = etaslice, 
                        xi_rho = xislice)
    dV = (ds.dx*ds.dy*ds.dz).isel(eta_rho = etaslice, 
                                  xi_rho = xislice)
    sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
    svar = (((salt-sbar)**2)*(dV)).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
    dsvardt = svar.diff('ocean_time')/dt
    dsvardt.name = 'dsvardt'

    dVdt.attrs = ''
    dsVdt.attrs = ''
    dsvardt.attrs = ''

    tendencies = xr.merge([dVdt, dsVdt, dsvardt], compat = 'override')
    return tendencies

def chi(ds, grid, xislice, etaslice, saltbins):
    '''
Computes the destruction of salinity variance, denoted by chi. For the TXLA model, 
the vertical mixing term dominates. However, I may do some experiments in the future
to test the variability of horizontal mixing. See Burchard and Rennau (2008) for more details. 
-----
Input: 
ds - xarray dataset
grid - xgcm grid
xislice - slice object of desired xi grid points
etaslice - slice object of desired eta grid points
saltbins - array of salinity bins. e.g. np.linspace(0,40,101)
-----
Output:
chih: histogram of salinity variance dissipation in salinity coordinates 
    '''
    
    #Compute the vertical salinity gradient
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

    chiint.name = 'chi'
    chih = histogram(salt, 
                     bins = [saltbins], 
                     weights = chi,
                     dim = ['s_rho', 'eta_rho', 'xi_rho']
                    )
    chih.name = 'chi'
    return chih

#---
#Start of plotting functions
def shelfplot(ds, figsize, var, cmap, extent, vmin, vmax):
    '''
Produces a surface contour map of some TXLA model variable. 
-----
Inputs:
ds - Xarray Dataset of model output
figsize - Figure size. E.g. (12,8)
var - Xarray DataArray of variable to be plotted. e.g. ds.salt.sel(ocean_time='2010-06-03-01').isel(s_rho=-1).values
cmap - Cmocean colormap. e.g. cmo.haline
extent - Extent of the grid. E.g. [-98, -88, 25, 30.5]
vmin - minimum colormap value,
vmax - maximum colormap value
------
Outputs:
Returned figure
    '''
    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                edgecolor='face',
                                facecolor=cfeature.COLORS['land'])
    states_provinces = cfeature.NaturalEarthFeature(
                       category='cultural',
                       name='admin_1_states_provinces_lines',
                       scale='10m',
                       facecolor='none')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.06, 0.01, 0.93, 0.95], projection=ccrs.PlateCarree())

    lon_rho = var['lon_rho'][:].data
    lat_rho = var['lat_rho'][:].data
    hlevs = [10, 20, 30, 50, 100]  # isobath contour depths

    mappable = ax.pcolormesh(lon_rho, lat_rho, var, 
                             cmap = cmap, 
                             transform = cartopy.crs.PlateCarree(),
                             vmin = vmin, vmax = vmax)
    
    gl = ax.gridlines(linewidth=0.4, color='black', alpha=0.5, linestyle='-', draw_labels=True)
    ax.set_extent(extent, ccrs.PlateCarree())
    # ax.set_extent([-94.5, -93.5, 27, 30.5], ccrs.PlateCarree())
    ax.add_feature(land_10m, facecolor='0.8')
    ax.coastlines(resolution='10m')  
    ax.add_feature(states_provinces, )
    ax.add_feature(cfeature.BORDERS, linestyle='-', )
    ax.add_feature(cartopy.feature.RIVERS, linewidth = 2)
    ax.set_aspect('auto')

    cax = fig.add_axes([0.09, 0.91, 0.32, 0.02]) 
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cb.set_label(r'Surface salinity [g$\cdot$kg$^{-1}$]', fontsize=18, color='0.2')
    cb.ax.tick_params(labelsize=18, length=2, color='0.2', labelcolor='0.2')
    
    #Plot isobaths and label the contours
    CS = ax.contour(lon_rho, lat_rho, ds.h, hlevs, 
                    colors='1', transform=ccrs.PlateCarree(), 
                    inline = 1, linewidths=1)
    ax.clabel(CS, fmt = '%1.0f', fontsize = 10)
    ax.tick_params(axis='y', labelsize=18)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.right_labels = False
    gl.top_labels = False