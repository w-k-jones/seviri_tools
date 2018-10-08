"""
SEVIRI_tools.py
module containing various tools for SEVIRI data analysis
"""

import numpy as np
import numpy.ma as ma
import scipy as sp
import netCDF4 as nc
from datetime import datetime,timedelta
from glob import glob

def era_interim_file_search(year, month, day, type, quiet=False):
    date = datetime(year=year, month=month, day=day)
    base_path = r'/badc/ecmwf-era-interim/data/'
    path_to_file = (base_path + type[:2] + '/' + type[2:] + '/' 
                    + str(date.year).zfill(4) + '/' 
                    + str(date.month).zfill(2) + '/' 
                    + str(date.day).zfill(2) + '/')
    try: 
        files = glob(path_to_file + type + '*.nc')
        files.sort()
    except:
        raise Exception('Cannot find ERA-interim directory: ' + path_to_file)
    if ~quiet:
        print 'Files found: ' + str(len(files))
    return files
    
def imerg_file_search(year, month, day, quiet=False):
    date = datetime(year=year, month=month, day=day)
    base_path = r'/group_workspaces/jasmin/acpc/Data/IMERG/'
    path_to_file = (base_path + str(date.year).zfill(4) + '/' 
                    + str(date.month).zfill(2) + '/')
    try: 
        files = glob(path_to_file + '/3B-HHR-E.MS.MRG.3IMERG.'
                     + str(date.year).zfill(4) 
                     + str(date.month).zfill(2)
                     + str(date.day).zfill(2) + '*.nc')
        files.sort()
    except:
        raise Exception('Cannot find IMERG directory: ' + path_to_file)
    if ~quiet:
        print 'Files found: ' + str(len(files))
    return files
    
def orac_merged_file_search(year, month, day, quiet=False):
    date = datetime(year=year, month=month, day=day)
    path_to_file = r'/group_workspaces/jasmin/acpc/Data/ORAC/clarify/merged/'
    try: 
        files = glob(path_to_file + '*' + str(date.year).zfill(4)
                        + str(date.month).zfill(2) 
                        + str(date.day).zfill(2)
                        + '*_fv2.0.merged.nc')
        files.sort()
    except:
        raise Exception('Cannot find ORAC-SEVIRI directory: ' + path_to_file)
    if ~quiet:
        print 'Files found: ' + str(len(files))
    return files
    
def sev_area_mean(data_in, n):
    """
    Finds the mean values over nxn pixels of a 2d array.
    Returns a 2d array of shape (x/n,y/n)
    """
    n = int(n)
    crop_size = (np.array(data_in.shape) // n) * n
    data_in = data_in[:crop_size[0], :crop_size[1]]
    data_out = data_in.reshape([crop_size[0] / n, n, crop_size[1] / n, n]).mean(3).mean(1)
    return data_out

def map_ll_to_seviri(lon, lat):
    """
    This function maps lat/lon points onto pixel location on the SEVIRI imager
    grid. Return is a tuple of masked arrays of the x and y imager grid
    locations.

    (SEVIRI pixel locations) = map_LL_to_SEV(lon, lat)

    This mapping can then be used to find NN or interpolate values much faster
    as bilinear methods can be used directly.
    e.g:
    x, y = map_LL_to_SEV(lon,lat)
    Sindx,Sindy=np.meshgrid(np.arange(3712),np.arange(3712))
    data_grid = interpolate.griddata((x.compressed(), y.compressed()), data,
                                     (Sindx,Sindy), method='linear')

    The function will also screen for input points that are outside the SEVIRI
    instrument field of view by calculating the effective instrument zenith
    angle.
    """
    # Define Earth radius and geostationary orbit height in km and calucalte max
    #  viewer angle
    r_sat = 42164.
    r_earth = 6378.
    zenith_max = np.arcsin(r_earth/r_sat)
    # convert lat/lon to cartesian coordinates
    x = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    y = np.sin(np.radians(lat))
    z = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    # x,y vector magnitude
    d = np.sqrt(x**2 + y**2)
    # Calculate footprint SEVIRI effective zenith angle and mask for > pi/2
    #  values
    zenith = np.arctan2(d, z) + np.arctan2(r_earth*d, r_sat-r_earth*z)
    zenith_mask = np.abs(zenith) >= (0.5 * np.pi)
    # Calculate x and y viewer angles
    theta_x = np.arctan2(r_earth*x, r_sat-r_earth*z)
    theta_y = np.arctan2(r_earth*y, r_sat-r_earth*z)
    # Define SEVIRI global index range and offset
    # These should be the same on all files, but may need to check
    x_irange = 3623
    x_ioffset = 44
    y_irange = 3611
    y_ioffset = 51
    # Remap viewer angles to indexes using max viewer angle, index range and
    #  offset. Note -ve theta_y as SEVIRI indexes the x-axis right to left(E-W)
    x_out = (1 - theta_x / zenith_max) * 0.5 * x_irange + x_ioffset
    y_out = (1 + theta_y / zenith_max) * 0.5 * y_irange + y_ioffset
    # Return masked arrays using the zenith angle mask
    return ma.array(x_out, mask=zenith_mask), ma.array(y_out, mask=zenith_mask)

def read_ecmwf_gafs_variable(vname,year,month,dom):
    """
    Read Accumulated (gafs) ECWMF variable from /badc/ on JASMIN
    Inputs
    variable name (e.g. 'TP' total precipitation)
    year [int]
    month [int]
    dom day of the month [int]
    Output
    ARR [accumulated field]
    """
    path_ecmwf = '/badc/ecmwf-era-interim/data/ga/fs/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(dom).zfill(2)+'/'
    print('fetching: '+path_ecmwf)
    prefix = str(year).zfill(4)+str(month).zfill(2)+str(dom).zfill(2)
    ftimes = ['0003','0006','0009','0012','1203','1206','1209','1212']
    ARR = np.empty( [8,256,512] )
    for i in range(len(ftimes)):
        ecmwf_file = path_ecmwf+'gafs'+prefix+ftimes[i]+'.nc'
        ncfile = nc.Dataset( ecmwf_file, mode='r')
        unitStr = ncfile.variables[vname].units
        ARR[i,:,:] = (ncfile.variables[vname][:])[0,0,:,:]

    #Express quantity as instantaneous
    tstep    = 3. * 3600. #3 hours converted to seconds
    aARR     = np.empty( [8,256,512] )
    aARR[0,:,:] = ( ARR[0,:,:]-0.        ) / tstep
    aARR[1,:,:] = ( ARR[1,:,:]-ARR[0,:,:]) / tstep
    aARR[2,:,:] = ( ARR[2,:,:]-ARR[1,:,:]) / tstep
    aARR[3,:,:] = ( ARR[3,:,:]-ARR[2,:,:]) / tstep
    aARR[4,:,:] = ( ARR[4,:,:]-0.        ) / tstep
    aARR[5,:,:] = ( ARR[5,:,:]-ARR[4,:,:]) / tstep
    aARR[6,:,:] = ( ARR[6,:,:]-ARR[5,:,:]) / tstep
    aARR[7,:,:] = ( ARR[7,:,:]-ARR[6,:,:]) / tstep

    #Daily means
    bARR = np.mean(aARR, axis=0)

    lat = ncfile.variables['latitude'][:]
    lon = ncfile.variables['longitude'][:]
    OUT = {'lat':lat, 'lon':lon, 'data':aARR, 'unit':unitStr+'/s', 'daily':bARR}
    return OUT

def latent_heat(T,phase=None):
    """
    Calcualtes the latent heat of condensation/deposition for liquid/ice water
    respectively using two empirical equations (Table 2.1. R. R. Rogers; M. K.
    Yau (1989). A Short Course in Cloud Physics (3rd ed.). Pergamon Press)
    Returns latent heat in J/g
    """
    scl_flag=False
    if np.isscalar(T):
        T = np.array([T])
        scl_flag = True
    # Check if in Kelvin and convert to Celsius
    T[T>100] -= 273.15


    # Initialise output array in shape of the temperature input
    out = np.full(T.shape, np.nan)

    # Find where the phase is liquid or T>0, and where the phase is ice or T<0
    wh_liq = np.logical_or(np.logical_and(phase=='liquid',T >= -25), T>0)
    wh_ice = np.logical_or(np.logical_or(np.logical_and(phase=='ice', T<=0), np.logical_and(phase!='liquid', T<=0)), T<-25)

    # Use empirical equations for liquid/ice latent heat of condensation/deposition
    out[wh_liq] = 2500.8-2.36*T[wh_liq]+0.0016*T[wh_liq]**2-0.00006*T[wh_liq]**3
    out[wh_ice] = 2834.1-0.29*T[wh_ice]-0.004*T[wh_ice]**2

    if scl_flag:
        out = out[0]

    return out
