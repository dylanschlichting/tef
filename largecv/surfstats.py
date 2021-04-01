'''
This script calculates surface statistics for the TXLA model output in the nGOM from summmer 2010. Variables include surface normalized vertical vorticity, lateral divergence, lateral strain, and the magnitude of the horizontal salinity gradients. 
'''
import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from glob import glob 

#Need to subset the netcdf files because xroms will crash if I open them all. 
paths = glob('../../../../dylan.schlichting/TXLA_Outputs/parent/2010/ocean_his_00*.nc')

ds = xroms.open_mfnetcdf(paths, 
                         chunks = {'ocean_time':1})
ds, grid = xroms.roms_dataset(ds, 
                              Vtransform = None)

xislice=slice(260,381) 
etaslice=slice(47,149)
    
months = np.arange(6, 9)
for m in months:

    #Compute vertical relative vorticity
#     rv = xroms.relative_vorticity(ds.u, ds.v, ds.u.attrs['grid'])
#     #Interpolate to the rho points. 
#     rv = grid.interp(rv, 'Z')
#     rv = rv.isel(eta_v = etaslice, xi_u = xislice, s_rho = -1)

#     #Note we're going to need to select the second to last vertical value since its the w-points

#     fx = grid.interp(ds.f, 'X', boundary = 'extend')
#     fxy = grid.interp(fx, 'Y', boundary = 'extend')
#     f = fxy.isel(eta_v = etaslice, xi_u = xislice)

#     RVn = rv/f

#     zetabins = np.linspace(-3,3,200)

#     RVn.name = 'relative_vorticity_n'
#     RVn_slice = RVn.sel(ocean_time = '2010-'+str(m))

#     #We need to remove the grid attributes from the variables. It will generate this annoying error code 
#     #because of an invalid character.
#     RVn_slice.attrs = ''
    
#     print('Computing histogam')
#     zetaf_hist = histogram(RVn_slice, bins = [zetabins], density = True)
    
#     print('Saving histogram')
#     zetaf_hist.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/statistics/Normalized_vvort_2010_%s.nc'% m, engine = 'h5netcdf')
    
    #Compute divergence
#     dudx = xroms.hgrad(ds.u, 
#                        grid, 
#                        which = 'xi')

#     #Interpolate to the vertical rho points, horizontal u and v points
#     dudxrho = grid.interp(dudx, 'Z')
#     dudxrhou = grid.interp(dudxrho, 'X')
#     dudxrhouv = grid.interp(dudxrhou, 'Y')

#     dvdy = xroms.hgrad(ds.v, 
#                        grid, 
#                        which = 'eta')

#     #Interpolate to the vertical rho points, horizontal u and v points
#     dvdyrho = grid.interp(dvdy, 'Z')
#     dvdyrhov = grid.interp(dvdyrho, 'Y')
#     dvdyrhovu = grid.interp(dvdyrhov, 'X')

#     divergence = dudxrhouv+dvdyrhovu

#     fx = grid.interp(ds.f, 'X', boundary = 'extend')
#     fxy = grid.interp(fx, 'Y', boundary = 'extend')

#     divnorm = divergence/fxy

#     divnormslice = divnorm.isel(eta_v = etaslice, 
#                                 xi_u = xislice, 
#                                 s_rho = -1).sel(ocean_time = '2010-'+str(m))
    
#     divbins = np.linspace(-3,3,200)
#     divnormslice.attrs = ''
#     divnormslice.name = 'divergence'
#     div_hist = histogram(divnormslice, bins = [divbins], density = True)
    
#     print('saving divergence histograms')
#     div_hist.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/statistics/Normalized_divergence_2010_%s.nc'% m, engine = 'h5netcdf')
    
#     #Lateral Strain
#     dudy = xroms.hgrad(ds.u, 
#                    grid, 
#                    which = 'eta')
#     dudyrho = grid.interp(dudy, 'Z')

#     dvdx = xroms.hgrad(ds.v, 
#                        grid, 
#                        which = 'xi')
#     dvdxrho = grid.interp(dvdx, 'Z')

#     strain = ((dudxrhouv-dvdyrhovu)**2+(dvdxrho+dudyrho)**2)**(1/2)

#     strainnorm = strain/fxy
#     strainnorm_slice = strainnorm.isel(eta_v = etaslice, 
#                                        xi_u = xislice, 
#                                        s_rho = -1).sel(ocean_time = '2010-'+str(m))
     
#     strainbins = np.linspace(0,3,200)
#     strainnorm_slice.attrs = ''
#     strainnorm_slice.name = 'strain'
#     strain_hist = histogram(strainnorm_slice, bins = [strainbins], density = True)
    
#     strain_hist.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/statistics/Normalized_strain_2010_%s.nc'% m, engine = 'h5netcdf')
    
    #Calculate magnitude of surface salinity gradients 
    dsdx = grid.interp(grid.diff(ds.salt, 'X'),'X', boundary = 'extend')/ds.dx
    dsdy = grid.interp(grid.diff(ds.salt, 'Y'),'Y', boundary = 'extend')/ds.dy
    
    dsdx.attrs = ''
    dsdy.attrs = ''
    
    dsdx_slice = dsdx.isel(eta_rho = etaslice, 
                           xi_rho = xislice, 
                           s_rho = -1).sel(ocean_time = '2010-'+str(m))
    dsdy_slice = dsdy.isel(eta_rho = etaslice, 
                           xi_rho = xislice, 
                           s_rho = -1).sel(ocean_time = '2010-'+str(m))
     

    sgradbins = np.linspace(0,0.002,200)

    dsdx_slice.name = 'dsdx'
    dsdy_slice.name = 'dsdy'
    
    #We need to remove the grid attributes from the variables. It will generate this annoying error code 
    #because of an invalid character.
    dsdx_slice.attrs = ''
    dsdy_slice.attrs = ''
    
    gradmag = ((dsdx_slice**2)+(dsdy_slice**2))**(1/2)
    gradmag.name = 'salgrad_mag'
    
    gradmag_hist = histogram(gradmag, bins = [sgradbins], density = True) 
    gradmag_hist.attrs = ''
    gradmag_hist.to_netcdf('/scratch/user/dylan.schlichting/tef/largecv/statistics/Normalized_sgradmag_2010_%s.nc'% m, engine = 'h5netcdf')