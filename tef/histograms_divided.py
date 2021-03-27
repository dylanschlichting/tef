'''
Computes transport-weighted histograms using more than 100 salinity bins. We need to output a netcdf
file for a subset of model output otherwise Ada will kill the script, so we implement a basic for loop. Currently configured for saving the histograms at approximately every 50 time steps Note you can just do all model outputs for less salinity bins. You might need to lower the amount of time steps for each .nc file if you use more salinity bins, but the threshold is unclear. 
'''

import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from datetime import datetime

def volume_flux(ds, xislice, etaslice):
    '''
-----
Input: 
ds - xarray roms dataset
grid - xgcm grid of roms output
xislice - slice object of desired xi points
etaslice -slice object of desired eta points
-----
Output:
Qda: volume flux at the west, east, north, and south control volume faces
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
Selects the salinity for the boundaries of a control volume for ROMS model output.
-----
Input: 
ds - xarray roms dataset
grid - xgcm grid of roms output
xislice - slice object of desired xi points
etaslice -slice object of desired eta points
-----
Output:
s'variable': salinity at the west, east, north, and south control volume faces
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
    
    eta_rho = slice(50,251)
    xi_rho = slice(150,351)
    dV = (ds.dx*ds.dy*ds.dz).isel(eta_rho = etaslice, #Add +1 points so dsvar/dt matches up with variance
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

def svarflux_hist(saltbins, svarda, Qsvarda):
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
    
    Qsh_da = xr.merge([QsWh, QsEh, QsNh, QsSh], compat = 'override') #Data array of salinity squared transport histograms
    return Qsh_da

def volflux_hist(saltbins, saltda, Qda):
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

paths = ['/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00001.nc',
          '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00002.nc',
          '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_hourly/ocean_his_child_00003.nc',
         ]

#paths = ['/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00001.nc',
 #        '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00002.nc',
  #       '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00003.nc',
   #      '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00004.nc',
    #     '/scratch/user/dylan.schlichting/TXLA_Outputs/nested_10min/ocean_his_child_00005.nc',
     #   ]

xislice=slice(50,250) #note tendencies have x+1 stopping points
etaslice=slice(150,350)

ds = xroms.open_mfnetcdf(paths, 
                     chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

#Break up the history files in to 20 evenly space intervals. 
#We need to do this so that the memory usage isn't past the cap
#and so concatenating the data is possible. 
nfiles = np.floor(np.linspace(0,len(ds.ocean_time), 60))

for item in (range(len(nfiles)-1)):
    ds1 = ds.isel(ocean_time = slice(int(nfiles[item]), int(nfiles[item+1])))
    print('Isolating control volume and computing tracer fluxes')
    
    saltda = salt_cv(ds1, grid, xislice, etaslice)
    Qda = volume_flux(ds1, xislice, etaslice)
    Qsda, Qssda = salt_flux(saltda, Qda)
    svarda,Qsvarda = Qcsvar_faces(ds1, grid, saltda, Qda, xislice, etaslice)

    saltbins = np.linspace(0,40,1001)

    print('Computing histograms')
    #Compute the histograms and add attributes for saving as netcdf
    Qsvarh_da = svarflux_hist(saltbins, saltda, Qsvarda)
    Qssh_da = ssquaredflux_hist(saltbins, saltda, Qssda)
    Qsh_da = saltflux_hist(saltbins, saltda, Qsda)
    Qh_da = volflux_hist(saltbins, saltda, Qda)

    Qsvarh_da.attrs['Description'] = 'Salinity variance transport weighted histograms'
    Qsvarh_da.attrs['Author'] = 'Dylan Schlichting'
    Qsvarh_da.attrs['Created'] = datetime.now().isoformat()
    Qsvarh_da.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)
    Qsvarh_da.attrs['Salinity Bins'] = str(len(saltbins)-1)
    Qsvarh_da.attrs['Qsvarh units'] = '(g/kg)^2 m^3 s^-1'
    
    print('Saving salinity variance histograms')
    
    path = str('../outputs/histograms/Divided_histograms/svar/1001/' +
           'Qsvarh_nested_hourly_xi_50250_eta150350_s1000_' +
           'timesteps_' + str(int(nfiles[item])) +  '_' +  
            str(int(nfiles[item+1])) +'.nc'
           )
    
    Qsvarh_da.to_netcdf(path)
    
    Qssh_da.attrs['Description'] = 'Salinity squared transport weighted histograms'
    Qssh_da.attrs['Author'] = 'Dylan Schlichting'
    Qssh_da.attrs['Created'] = datetime.now().isoformat()
    Qssh_da.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)
    Qssh_da.attrs['Salinity Bins'] = str(len(saltbins)-1)
    Qssh_da.attrs['Qssh units'] = '(g/kg)^2 m^3 s^-1'

    print('Saving salinity squared histograms')
    
    path = str('../outputs/histograms/Divided_histograms/ssquare/1001/' +
           'Qssh_nested_hourly_xi_50250_eta150350_s1000_' +
           'timesteps_' + str(int(nfiles[item])) +  '_' +  
            str(int(nfiles[item+1])) +'.nc'
           )
    
    Qssh_da.to_netcdf(path)

    Qsh_da.attrs['Description'] = 'Salinity transport weighted histograms'
    Qsh_da.attrs['Author'] = 'Dylan Schlichting'
    Qsh_da.attrs['Created'] = datetime.now().isoformat()
    Qsh_da.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)
    Qsh_da.attrs['Salinity Bins'] = str(len(saltbins)-1)
    Qsh_da.attrs['Qsh units'] = '(g/kg) m^3 s^-1'

    print('Saving salinity histograms')
    
    path = str('../outputs/histograms/Divided_histograms/salt/1001/' +
           'Qsh_nested_hourly_xi_50250_eta150350_s1000_' +
           'timesteps_' + str(int(nfiles[item])) +  '_' +  
            str(int(nfiles[item+1])) +'.nc'
           )
    
    Qsh_da.to_netcdf(path)

    Qh_da.attrs['Description'] = 'Volume transport weighted histograms'
    Qh_da.attrs['Author'] = 'Dylan Schlichting'
    Qh_da.attrs['Created'] = datetime.now().isoformat()
    Qh_da.attrs['Grid'] = 'xi points: '+str(xislice)+', eta points: '+str(etaslice)
    Qh_da.attrs['Salinity Bins'] = str(len(saltbins)-1)
    Qh_da.attrs['Q units'] = 'm^3 s^-1'

    #print('Saving volume histograms')
    path = str('../outputs/histograms/Divided_histograms/vol/1001/' +
           'Qh_nested_hourly_xi_50250_eta150350_s1000_' +
           'timesteps_' + str(int(nfiles[item])) +  '_' +  
            str(int(nfiles[item+1])) +'.nc'
           )
    
    Qh_da.to_netcdf(path)
