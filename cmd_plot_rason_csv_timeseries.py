# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 13:57:00 2020

@author: donaldi permana
Indonesia Agency for Meteorology Climatology and Geophysics (BMKG)
donaldi.permana@bmkg.go.id
"""
#from IPython import get_ipython
#get_ipython().magic('clear') # clear screen
#get_ipython().magic('reset -sf')

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as pl
import math
import pandas as pd
# from skewt import SkewT
import copy
#def dependencies_for_myprogram():
    #import xmltodict

# just making sure that the plots immediately pop up
#pl.interactive(True)  # noqa
#import pylab as pyl
import datetime as dt
import glob
import sys, getopt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data
    
def windcalc(u,v):
    ws = np.sqrt(u*u + v*v)
    if (ws != 0):    
        wd= (270-math.atan2(v,u)*180/math.pi)%360
    else:
        wd = np.nan # calm
    return ws,wd
    
def uvcalc(ws,wd):
    wd = 270-wd    
    u = ws * math.cos(math.radians(wd))
    v = ws * math.sin(math.radians(wd)) 
    
    return u,v

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
       opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
       print ('test.py -i <inputfile> -o <outputfile>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print ('test.py -i <inputfile> -o <outputfile>')
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg
       elif opt in ("-o", "--ofile"):
          outputfile = arg
    print ('Input file is "', inputfile)
#    print 'Output file is "', outputfile

#    path = 'L:\Radar2016\sample_data\pky1.vol'
    path = inputfile
    dirout = '.\\'
    # path = 'I:\Rason\*.CSV'
    files = sorted(glob.glob(path))
    
    tstart = dt.datetime.now()

    idx = 0
    ts_dt_rason = np.array([])
    ts_height = np.array([])
    ts_press = np.array([])
    ts_dew_temp = np.array([])
    ts_temp = np.array([])
    ts_CAPE = np.array([])
    ts_CIN = np.array([])
    ts_TPW = np.array([])
    ts_pot_temp = np.array([])
    ts_rh = np.array([])    
    ts_u = np.array([])
    ts_v = np.array([])
    ts_size = np.array([])
    ts_lat = np.array([])
    ts_lon = np.array([])
    
    max_press = 1000 # hPa
    min_press = 50 # hPa
    diff_press = 0.1 # hPa
    new_press = np.linspace(min_press,max_press,int(((max_press-min_press)/diff_press))+1)
    new_press = new_press[::-1] # reverse the array

    max_height_recorded = 0    
    min_height_recorded = 20000
    
    dts = np.arange(dt.datetime(2019,1,1,0), dt.datetime(2019,1,31,12), dt.timedelta(hours=12)).astype(dt.datetime)
    
    for dtime in dts:
        strdtime = dtime.strftime('%Y%m%d%H')
    
        filename = [x for x in files if strdtime in x]
        
        if filename == []:
            print ('Datetime '+strdtime+ ' is not found')
            new_nan = np.arange(0,np.size(new_press)).astype(float)
            new_nan[:] = np.nan
            try:
                ts_temp = np.column_stack([ts_temp, new_nan])
                ts_pot_temp = np.column_stack([ts_pot_temp, new_nan])
                ts_height = np.column_stack([ts_height, new_nan])
            except:
                ts_temp = np.hstack((ts_temp,new_nan))
                ts_pot_temp = np.hstack((ts_pot_temp,new_nan))
                ts_height = np.hstack((ts_height,new_nan))
                
            try:
                ts_rh = np.column_stack([ts_rh, new_nan])
            except:
                ts_rh = np.hstack((ts_rh,new_nan))
                
            try:
                ts_u = np.column_stack([ts_u, new_nan])
            except:
                ts_u = np.hstack((ts_u,new_nan))
                
            try:
                ts_v = np.column_stack([ts_v, new_nan])
            except:
                ts_v = np.hstack((ts_v,new_nan))
                
            # ts_CAPE = np.append(ts_CAPE,np.nan)
            # ts_CIN = np.append(ts_CIN,np.nan)
            # ts_TPW = np.append(ts_TPW,np.nan)
        else:
            filename = filename[0]
            #for filename in files:
            print ('Reading file '+filename+' ...')
            
            #filename = 'n:\\rason1\F2017111606S6008157.csv'
            
            #StaNo = filename[-20:-15]
            # StaNo = '97180'
            
            rason_datetime = filename[-14:-4]
            #rason_datetime = filename[-22:-12]
            YY = str(rason_datetime[0:4])
            MM = str(rason_datetime[4:6])
            DD = str(rason_datetime[6:8])
            HH = str(rason_datetime[8:10])
            dt_rason = dt.datetime.strptime(YY+'-'+MM+'-'+DD+'_'+HH,'%Y-%m-%d_%H')
            #dt_rason = dt_rason - dt.timedelta(hours=6) # 6 hours time difference
            strdt_rason = dt_rason.strftime('%Y-%m-%d_%H')        
            
            df = pd.read_csv(filename, nrows=0)
            rason_type = df.columns[0]
            rason_release_date_local = df.columns[4]        
            rason_release_time_local = df.columns[5]
            rason_sounding_length = df.columns[7]
            rason_lat = str(round(float(df.columns[9]),2))
            rason_lon = str(round(float(df.columns[10]),2))
            rason_observer = df.columns[13]
            
            try:
                dt_local_rason = dt.datetime.strptime(rason_release_date_local+'_'+rason_release_time_local[:-6],'%Y/%m/%d_%H')
            except:
                dt_local_rason = dt.datetime.strptime(rason_release_date_local+'_'+rason_release_time_local[:-6],'%m/%d/%Y_%H')
            print ('local and utc time difference : '+str((dt_local_rason-dt_rason).seconds/3600) + ' hours')
            
    #        newdir = filename+'.dir'
    #        try:
    #            os.mkdir(newdir, 0755 );
    #        except:
    #            print newdir+' already exists'
            
            df = pd.read_csv(filename, skiprows=6)
                    
            #obstime = df[df.columns[0]].map(lambda x: dt.datetime.strptime(str(x), '%H:%M:%S')) #ObsTime
            
            press = df[df.columns[20]].replace('-----',np.nan) #Pressure
            press = press.replace('------',np.nan).astype(float)
            
            height = df[df.columns[11]].replace('-----',np.nan).astype(float) #Height
            
            if(max_height_recorded < np.nanmax(height)):
                max_height_recorded = np.nanmax(height)
                max_height_recorded_dt = strdt_rason
            
            if(min_height_recorded > np.nanmax(height)):
                min_height_recorded = np.nanmax(height)
                min_height_recorded_dt = strdt_rason
            
            lat = df[df.columns[17]].replace('-----',np.nan).astype(float) #GeodetLat
            lon = df[df.columns[18]].replace('-----',np.nan).astype(float) #GeodetLon
            wd = df[df.columns[9]].replace('-----',np.nan).astype(float) #Wind Direction
            ws = df[df.columns[10]].replace('-----',np.nan).astype(float) # Wind Speed
            temp = df[df.columns[21]].replace('-----',np.nan).astype(float) #Temperature
            rh = df[df.columns[22]].replace('-----',np.nan).astype(float) #Relative Humidity
            dewtemp = temp - ((100 - rh)/5.) # Dewpoint temperature
            # calculating potential temperature
            # pot_temp = (temp+273.15)*np.power((1000./press),0.286) (Kelvin), 0.286 is poisson constant
            pot_temp = (temp+273.15)*np.power((1000./press),0.286)

            ## 'linear' has no effect since it's the default, but I'll plot it too:
            #set_interp = interp1d(press,temp, kind='linear')
            #new_temp = set_interp(new_press)
    
            # temperature, potential temperature and height
            try:
                new_temp = interp1d(press,temp)(new_press)
                new_pot_temp = interp1d(press,pot_temp)(new_press)
                new_height = interp1d(press,height)(new_press)
            except:
                max_press0 = max_press
                min_press0 = min_press                
                if max_press > np.max(press):
                    max_press0 = np.max(press)
                if min_press < np.min(press):
                    min_press0 = np.min(press)
                
                new_press0 = np.linspace(min_press0,max_press0,int(((max_press0-min_press0)/diff_press))+1)
                new_press0 = new_press0[::-1] # reverse the array
        
                new_temp = interp1d(press,temp)(new_press0)
                new_temp = np.append(new_temp,np.repeat(np.nan,np.floor((min_press0-min_press)/diff_press)))
                new_temp = np.append(np.repeat(np.nan,np.floor((max_press-max_press0)/diff_press)),new_temp)
                new_temp = np.append(new_temp,np.repeat(np.nan,np.size(new_press)-np.size(new_temp)))
                
                new_pot_temp = interp1d(press,pot_temp)(new_press0)
                new_pot_temp = np.append(new_pot_temp,np.repeat(np.nan,np.floor((min_press0-min_press)/diff_press)))
                new_pot_temp = np.append(np.repeat(np.nan,np.floor((max_press-max_press0)/diff_press)),new_pot_temp)
                new_pot_temp = np.append(new_pot_temp,np.repeat(np.nan,np.size(new_press)-np.size(new_pot_temp)))                
                
                new_height = interp1d(press,height)(new_press0)
                new_height = np.append(new_height,np.repeat(np.nan,np.floor((min_press0-min_press)/diff_press)))
                new_height = np.append(np.repeat(np.nan,np.floor((max_press-max_press0)/diff_press)),new_height)
                new_height = np.append(new_height,np.repeat(np.nan,np.size(new_press)-np.size(new_height)))
                
            try:
                ts_temp = np.column_stack([ts_temp, new_temp])
                ts_pot_temp = np.column_stack([ts_pot_temp, new_pot_temp])                
                ts_height = np.column_stack([ts_height, new_height])
            except:
                ts_temp = np.hstack((ts_temp,new_temp))
                ts_pot_temp = np.hstack((ts_pot_temp,new_pot_temp))                
                ts_height = np.hstack((ts_height,new_height))
            
            #ts_size = np.append(ts_size, np.size(new_pot_temp))
            
            # relative humidity
            try:
                new_rh = interp1d(press,rh)(new_press)
                new_rh = pad(new_rh)
            except:
                if max_press > np.max(press):
                    max_press0 = np.max(press)
                if min_press < np.min(press):
                    min_press0 = np.min(press)
                
                new_press0 = np.linspace(min_press0,max_press0,int(((max_press0-min_press0)/diff_press))+1)
                new_press0 = new_press0[::-1] # reverse the array
        
                new_rh = interp1d(press,rh)(new_press0)
                new_rh = pad(new_rh)
                new_rh = np.append(new_rh,np.repeat(np.nan,np.floor((min_press0-min_press)/diff_press)))
                new_rh = np.append(np.repeat(np.nan,np.floor((max_press-max_press0)/diff_press)),new_rh)
                new_rh = np.append(new_rh,np.repeat(np.nan,np.size(new_press)-np.size(new_rh)))
                   
            try:
                ts_rh = np.column_stack([ts_rh, new_rh])
            except:
                ts_rh = np.hstack((ts_rh,new_rh))
                
            u = np.arange(0,np.size(ws)).astype(float)
            #u[:] = np.nan
            v = np.arange(0,np.size(ws)).astype(float)
            #v[:] = np.nan        
            for i in range(0, np.size(u)):
                [u[i],v[i]] = uvcalc(ws[i],wd[i])
            
            # u-component - zonal wind
            try:
                new_u = interp1d(press,u)(new_press)
            except:
                if max_press > np.max(press):
                    max_press0 = np.max(press)
                if min_press < np.min(press):
                    min_press0 = np.min(press)
                
                new_press0 = np.linspace(min_press0,max_press0,int(((max_press0-min_press0)/diff_press))+1)
                new_press0 = new_press0[::-1] # reverse the array
        
                new_u = interp1d(press,u)(new_press0)
                new_u = np.append(new_u,np.repeat(np.nan,np.floor((min_press0-min_press)/diff_press)))
                new_u = np.append(np.repeat(np.nan,np.floor((max_press-max_press0)/diff_press)),new_u)
                new_u = np.append(new_u,np.repeat(np.nan,np.size(new_press)-np.size(new_u)))
                   
            try:
                ts_u = np.column_stack([ts_u, new_u])
            except:
                ts_u = np.hstack((ts_u,new_u))
                
            # v-component - meridional wind
            try:
                new_v = interp1d(press,v)(new_press)
            except:
                if max_press > np.max(press):
                    max_press0 = np.max(press)
                if min_press < np.min(press):
                    min_press0 = np.min(press)
                
                new_press0 = np.linspace(min_press0,max_press0,int(((max_press0-min_press0)/diff_press))+1)
                new_press0 = new_press0[::-1] # reverse the array
        
                new_v = interp1d(press,v)(new_press0)
                new_v = np.append(new_v,np.repeat(np.nan,np.floor((min_press0-min_press)/diff_press)))
                new_v = np.append(np.repeat(np.nan,np.floor((max_press-max_press0)/diff_press)),new_v)
                new_v = np.append(new_v,np.repeat(np.nan,np.size(new_press)-np.size(new_v)))
                   
            try:
                ts_v = np.column_stack([ts_v, new_v])
            except:
                ts_v = np.hstack((ts_v,new_v))
                
            #_AllowedKeys=['pres','hght','temp','dwpt','relh','mixr','drct','sknt','thta','thte','thtv']
            # mydata=dict(zip(('StationNumber','SoundingDate','hght','pres','temp','dwpt','relh','drct','sknt'),\
            # ('MAKASSAR ('+StaNo+')', strdt_rason +'UTC', \
            # pad(height), pad(press),pad(temp),pad(dewtemp),pad(rh),pad(wd),pad(ws))))
             
            # S=SkewT.Sounding(soundingdata=mydata)
             
            # #Calculate CAPE and TPW
            # parcel = S.get_parcel(method='sb')
            # Ps,TCs,TDs,method = parcel
            # P_lcl,P_lfc,P_el,CAPE,CIN=S.get_cape(*parcel)
            # TPW = S.precipitable_water()            
            # ts_CAPE = np.append(ts_CAPE,CAPE)
            # ts_CIN = np.append(ts_CIN,CIN)
            # ts_TPW = np.append(ts_TPW,TPW)
             
            #plot skew T diagram            
            #S.plot_skewt(tmin=-40.,tmax=40.,pmin=100.,pmax=1050.,parcel_type='sb')
    
            #S.make_skewt_axes(tmin=-40.,tmax=40.,pmin=100.,pmax=1050.)
            #S.add_profile(lw=1)
   
            idx = idx + 1

    #calculate mean height    
    mean_height = np.nanmean(ts_height,axis=1)
    # calculate temperature mean for each pressure level
    mean_temp = np.nanmean(ts_temp,axis=1)
    # calculate temperature anomaly for each pressure level
    ts_temp_anomaly = ts_temp - mean_temp[:,None]
    # calculate potential temperature mean for each pressure level
    mean_pot_temp = np.nanmean(ts_pot_temp,axis=1)
    # calculate potential temperature anomaly for each pressure level
    ts_pot_temp_anomaly = ts_pot_temp - mean_pot_temp[:,None]
    # calculate the freezing level line ( temp = 0 C)
    ts_height_temp0C = np.array([])
    for j in range(len(np.transpose(ts_temp))):
        try:        
            idx0 = np.where(np.abs(ts_temp[:,j]-0)<0.02)
            idx0 = int(np.floor(np.nanmean(idx0)))            
            ts_height_temp0C = np.append(ts_height_temp0C,mean_height[idx0])
        except:
            ts_height_temp0C = np.append(ts_height_temp0C,np.nan)
    
    #set TPW = Nan when < 1 mm
    # ts_TPW[np.where(ts_TPW<1)] = np.nan
    
    # ts_ws = np.zeros((len(ts_u),len(np.transpose(ts_u)))) 
    # ts_wd = np.zeros((len(ts_u),len(np.transpose(ts_u)))) 
    # for i in range(0, len(ts_u)):
    #     for j in range(0, len(np.transpose(ts_u))):
    #         [ts_ws[i,j],ts_wd[i,j]] = windcalc(ts_u[i,j], ts_v[i,j])
        
    # convert from pressure to height interpolation
    max_height = 20500. # meter or ~20 hPa
    min_height = 100 # meter
    diff_height = 5 # meter
    new_height = np.linspace(min_height,max_height,int(((max_height-min_height)/diff_height))+1)
    
    tsh_temp = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_temp_anomaly = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_pot_temp = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_pot_temp_anomaly = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_rh = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_u = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_v = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_ws = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    tsh_wd = np.zeros((len(new_height),len(np.transpose(ts_temp)))) 
    for j in range(len(np.transpose(ts_temp))):
        tsh_temp[:,j] = interp1d(mean_height,ts_temp[:,j])(new_height)
        tsh_temp_anomaly[:,j] = interp1d(mean_height,ts_temp_anomaly[:,j])(new_height)
        tsh_pot_temp[:,j] = interp1d(mean_height,ts_pot_temp[:,j])(new_height)
        tsh_pot_temp_anomaly[:,j] = interp1d(mean_height,ts_pot_temp_anomaly[:,j])(new_height)
        tsh_rh[:,j] = interp1d(mean_height,ts_rh[:,j])(new_height)
        tsh_u[:,j] = interp1d(mean_height,ts_u[:,j])(new_height)
        tsh_v[:,j] = interp1d(mean_height,ts_v[:,j])(new_height)
        for i in range(0, np.size(tsh_u[:,j])):    
            [tsh_ws[i,j],tsh_wd[i,j]] = windcalc(tsh_u[i,j], tsh_v[i,j])
            
    #pl.contourf(tsh_v)
    #pl.colorbar()
    
    #calculate diurnal mean for each parameter (temp,pot_temp,rh,u,v,height)
    dmean_temp = np.zeros((8,len(ts_temp)))
    dmean_temp_anomaly = np.zeros((8,len(ts_temp)))    
    dmean_pot_temp = np.zeros((8,len(ts_temp)))
    dmean_pot_temp_anomaly = np.zeros((8,len(ts_temp)))
    dmean_rh = np.zeros((8,len(ts_temp)))
    dmean_u = np.zeros((8,len(ts_temp)))
    dmean_v = np.zeros((8,len(ts_temp)))
    dmean_ws = np.zeros((8,len(ts_temp)))
    dmean_wd = np.zeros((8,len(ts_temp)))    
    dmean_height = np.zeros((8,len(ts_temp)))
    dmeanh_temp = np.zeros((8,len(tsh_temp)))
    dmeanh_temp_anomaly = np.zeros((8,len(tsh_temp)))    
    dmeanh_pot_temp = np.zeros((8,len(tsh_temp)))
    dmeanh_pot_temp_anomaly = np.zeros((8,len(tsh_temp)))
    dmeanh_rh = np.zeros((8,len(tsh_temp)))
    dmeanh_u = np.zeros((8,len(tsh_temp)))
    dmeanh_v = np.zeros((8,len(tsh_temp)))
    dmeanh_ws = np.zeros((8,len(tsh_temp)))
    dmeanh_wd = np.zeros((8,len(tsh_temp)))
    dmean_height_temp0C = np.zeros((8))
    # dmean_CAPE = np.zeros((8))
    # dmean_CIN = np.zeros((8))
    # dmean_TPW = np.zeros((8))
    
    for i in np.arange(8):
        dmean_temp[i,:] = np.nanmean(ts_temp[:,i::8],axis=1)
        dmean_temp_anomaly[i,:] = np.nanmean(ts_temp_anomaly[:,i::8],axis=1)
        dmean_pot_temp[i,:] = np.nanmean(ts_pot_temp[:,i::8],axis=1)
        dmean_pot_temp_anomaly[i,:] = np.nanmean(ts_pot_temp_anomaly[:,i::8],axis=1)
        dmean_rh[i,:] = np.nanmean(ts_rh[:,i::8],axis=1)
        dmean_u[i,:] = np.nanmean(ts_u[:,i::8],axis=1)
        dmean_v[i,:] = np.nanmean(ts_v[:,i::8],axis=1)
        for j in range(0, np.size(dmean_u[i,:])):    
            [dmean_ws[i,j],dmean_wd[i,j]] = windcalc(dmean_u[i,j], dmean_v[i,j])
        dmean_height[i,:] = np.nanmean(ts_height[:,i::8],axis=1)
        
        dmeanh_temp[i,:] = np.nanmean(tsh_temp[:,i::8],axis=1)
        dmeanh_temp_anomaly[i,:] = np.nanmean(tsh_temp_anomaly[:,i::8],axis=1)
        dmeanh_pot_temp[i,:] = np.nanmean(tsh_pot_temp[:,i::8],axis=1)
        dmeanh_pot_temp_anomaly[i,:] = np.nanmean(tsh_pot_temp_anomaly[:,i::8],axis=1)
        dmeanh_rh[i,:] = np.nanmean(tsh_rh[:,i::8],axis=1)
        dmeanh_u[i,:] = np.nanmean(tsh_u[:,i::8],axis=1)
        dmeanh_v[i,:] = np.nanmean(tsh_v[:,i::8],axis=1)
        for j in range(0, np.size(dmeanh_u[i,:])):    
            [dmeanh_ws[i,j],dmeanh_wd[i,j]] = windcalc(dmeanh_u[i,j], dmeanh_v[i,j])
        
        dmean_height_temp0C[i] = np.nanmean(ts_height_temp0C[i::8])
        # dmean_CAPE[i] = np.nanmean(ts_CAPE[i::8])
        # dmean_CIN[i] = np.nanmean(ts_CIN[i::8])
        # dmean_TPW[i] = np.nanmean(ts_TPW[i::8])

    # plotting part
    nn = 1 
    idxtop = 3980
    interval = 200 # *5 = 1000 meter
    idx = np.append(0,np.arange(180,idxtop+1,interval))

    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(121)
    cs = pl0.contourf(np.transpose(dmeanh_temp[:,0:idxtop]), cmap="nipy_spectral")        
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl0.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(8))
    pl0.set_xticklabels(np.arange(8)*3)
    pl0.set_xlabel('Time [UTC]')     
    pl.colorbar(cs)
    pl0.set_title('Diurnal Mean Temp. [C] - BMKG')
    
    pl1 = fig.add_subplot(122)
    cs = pl1.contourf(np.transpose(dmeanh_temp_anomaly[:,0:idxtop]), cmap="nipy_spectral")        
    pl1.contour(np.transpose(dmeanh_temp_anomaly[:,0:idxtop]), levels=[0],colors='black',linewidths=[0.5],linestyles='solid')     
    pl1.set_yticks(idx)
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl1.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl1.set_ylabel('Height [km]')    
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(8))
    pl1.set_xticklabels(np.arange(8)*3)
    pl1.set_xlabel('Time [UTC]')     
    pl.colorbar(cs)
    pl1.set_title('Diurnal Mean Temp. Anom [C] - BMKG')
    fig.savefig(dirout+'diurnal_t_anomt.png', format='png', dpi=300)

    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(121)
    cs = pl0.contourf(np.transpose(dmeanh_pot_temp[:,0:idxtop]), cmap="nipy_spectral")        
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl0.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(8))
    pl0.set_xticklabels(np.arange(8)*3)
    pl0.set_xlabel('Time [UTC]')     
    pl.colorbar(cs)
    pl0.set_title('Diurnal Mean Pot. Temp. [K] - BMKG')
    
    pl1 = fig.add_subplot(122)
    cs = pl1.contourf(np.transpose(dmeanh_pot_temp_anomaly[:,0:idxtop]), cmap="nipy_spectral")        
    pl1.contour(np.transpose(dmeanh_pot_temp_anomaly[:,0:idxtop]), levels=[0],colors='black',linewidths=[0.5],linestyles='solid')     
    pl1.set_yticks(idx)
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl1.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl1.set_ylabel('Height [km]')    
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(8))
    pl1.set_xticklabels(np.arange(8)*3)
    pl1.set_xlabel('Time [UTC]')     
    pl.colorbar(cs)
    pl1.set_title('Diurnal Mean Pot. Temp. Anom [K] - BMKG')
    fig.savefig(dirout+'diurnal_pot_t_anom_pot_t.png', format='png', dpi=300)
   
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(121)
    cs = pl0.contourf(np.transpose(dmeanh_u[:,0:idxtop]), cmap="nipy_spectral")
    pl0.contour(np.transpose(dmeanh_u[:,0:idxtop]), levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl0.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(8))
    pl0.set_xticklabels(np.arange(8)*3)
    pl0.set_xlabel('Time [UTC]')     
    pl.colorbar(cs)
    pl0.set_title('Diurnal Mean Zonal Wind [m/s] - BMKG')
    
    pl1 = fig.add_subplot(122)
    cs = pl1.contourf(np.transpose(dmeanh_v[:,0:idxtop]), cmap="nipy_spectral")        
    pl1.contour(np.transpose(dmeanh_v[:,0:idxtop]), levels=[0],colors='black',linewidths=[0.5],linestyles='solid')     
    pl1.set_yticks(idx)
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl1.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl1.set_ylabel('Height [km]')    
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(8))
    pl1.set_xticklabels(np.arange(8)*3)
    pl1.set_xlabel('Time [UTC]')     
    pl.colorbar(cs)
    pl1.set_title('Diurnal Mean Meridional Wind [m/s] - BMKG')
    fig.savefig(dirout+'diurnal_u_v.png', format='png', dpi=300)

    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(121)
    cs = pl0.contourf(np.transpose(dmeanh_ws[:,0:idxtop]), cmap="nipy_spectral")
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl0.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(8))
    pl0.set_xticklabels(np.arange(8)*3)
    pl0.set_xlabel('Time [UTC]')     
    pl.colorbar(cs)
    pl0.set_title('Diurnal Mean Wind Speed [m/s] - BMKG')
    
    pl1 = fig.add_subplot(122)
    clevs = [0,45,90,135,180,225,270,315,360]
    cs = pl1.contourf(np.transpose(dmeanh_wd[:,0:idxtop]), clevs, cmap="nipy_spectral")        
    pl1.set_yticks(idx)
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl1.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl1.set_ylabel('Height [km]')    
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(8))
    pl1.set_xticklabels(np.arange(8)*3)
    pl1.set_xlabel('Time [UTC]')     
    cbar = pl.colorbar(cs)
    cbar.ax.set_yticklabels(['N','NE','E','SE','S','SW','W','NW','N'])
    pl1.set_title('Diurnal Mean Wind Direction - BMKG')
    fig.savefig(dirout+'diurnal_ws_wd.png', format='png', dpi=300)
      
   
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(121)
    pl0.barbs(np.transpose(dmeanh_u[:,idx]),np.transpose(dmeanh_v[:,idx]))    
    pl0.set_yticks((new_height[idx]/1000))
    pl0.set_ylim(0,21)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(8))
    pl0.set_xticklabels(np.arange(8)*3)
    pl0.set_xlabel('Time [UTC]')     
    pl0.set_title('Diurnal Mean Wind Speed and Direction - BMKG')
    
    pl1 = fig.add_subplot(122)
    clevs = np.arange(0,101,5)    
    cs = pl1.contourf(np.transpose(dmeanh_rh[:,0:idxtop]), clevs, cmap="nipy_spectral")        
    pl1.set_yticks(idx)
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels((new_height[idx]/1000).astype(float))
    #pl1.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl1.set_ylabel('Height [km]')
    pl1.tick_params(axis='x',direction='out')
    pl1.set_xticks(np.arange(8))
    pl1.set_xticklabels(np.arange(8)*3)
    pl1.set_xlabel('Time [UTC]')     
    pl.colorbar(cs,ax=pl1)
    pl1.set_title('Diurnal Mean Rel. Humidity [%] - BMKG')
    fig.savefig(dirout+'diurnal_windbarb_rh.png', format='png', dpi=300)
   
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(221)
    pl0.plot(dmean_height_temp0C,'ob-')    
    pl0.tick_params(axis='y',direction='out')
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(8))
    pl0.set_xticklabels(np.arange(8)*3)
    pl0.set_xlabel('Time [UTC]')     
    pl0.set_title('Diurnal Mean Freezing level Height - BMKG')
    
    # pl0 = fig.add_subplot(222)
    # pl0.plot(dmean_CAPE/1000.,'ob-')    
    # pl0.tick_params(axis='y',direction='out')
    # pl0.set_ylabel('CAPE [KJ]')    
    # pl0.tick_params(axis='x',direction='out')
    # pl0.set_xticks(np.arange(8))
    # pl0.set_xticklabels(np.arange(8)*3)
    # pl0.set_xlabel('Time [UTC]')     
    # pl0.set_title('Diurnal Mean CAPE - BMKG')
    
    # pl0 = fig.add_subplot(223)
    # pl0.plot(dmean_CIN/1000.,'ob-')    
    # pl0.tick_params(axis='y',direction='out')
    # pl0.set_ylabel('CIN [KJ]')    
    # pl0.tick_params(axis='x',direction='out')
    # pl0.set_xticks(np.arange(8))
    # pl0.set_xticklabels(np.arange(8)*3)
    # pl0.set_xlabel('Time [UTC]')     
    # pl0.set_title('Diurnal Mean CIN - BMKG')
    
    # pl0 = fig.add_subplot(224)
    # pl0.plot(dmean_TPW,'ob-')    
    # pl0.tick_params(axis='y',direction='out')
    # pl0.set_ylabel('TPW [mm]')    
    # pl0.tick_params(axis='x',direction='out')
    # pl0.set_xticks(np.arange(8))
    # pl0.set_xticklabels(np.arange(8)*3)
    # pl0.set_xlabel('Time [UTC]')     
    # pl0.set_title('Diurnal Mean TPW - BMKG')
    fig.savefig(dirout+'diurnal_freezelevel_cape_cin_tpw.png', format='png', dpi=300)    
    
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(411)
    pl0.plot(ts_height_temp0C,'.k-')
    #pl0.set_yticks(np.arange(0,5.1,0.5))
    pl0.tick_params(axis='y',direction='out')
    pl0.set_ylabel('Height [m]')
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    #pl0.set_xticklabels(strdts)    
    pl0.set_xticklabels([])
    pl0.set_title('Freezing level (0 degC) Height [m] - BMKG')
    
    pl0 = fig.add_subplot(412)
    pl0.plot(ts_CAPE/1000.,'.b-')
    pl0.set_yticks(np.arange(0,5.1,0.5))
    pl0.tick_params(axis='y',direction='out')
    pl0.set_ylabel('CAPE [KJ]')
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    #pl0.set_xticklabels(strdts)
    pl0.set_xticklabels([])    
    pl0.set_title('Convective Available Potential Energy [KJ] - BMKG')
    
    pl1 = fig.add_subplot(413)
    pl1.plot(ts_CIN/1000.,'.k-')
    pl1.set_yticks(np.arange(-0.21,0.01,0.03))
    pl1.tick_params(axis='y',direction='out')
    pl1.set_ylabel('CIN [KJ]')
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    #pl1.set_xticklabels(strdts)
    pl1.set_xticklabels([])
    pl1.set_title('Convective Inhibition [KJ] - BMKG')
            
    pl2 = fig.add_subplot(414)
    pl2.plot(ts_TPW,'.r-')
    pl2.set_yticks(np.arange(40,71,5))
    pl2.tick_params(axis='y',direction='out')
    pl2.set_ylabel('TPW [mm]')
    pl2.tick_params(axis='x',direction='out')    
    pl2.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl2.set_xticklabels(strdts)
    pl2.set_title('Total Precipitable Water [mm] - BMKG')
    fig.savefig(dirout+'ts_freezelevel_cape_cin_tpw.png', format='png', dpi=300)    

#_-------------
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(411)
    pl0.plot(pad(ts_height_temp0C),'.k-')
    #pl0.set_yticks(np.arange(0,5.1,0.5))
    pl0.tick_params(axis='y',direction='out')
    pl0.set_ylabel('Height [m]')
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    #pl0.set_xticklabels(strdts)    
    pl0.set_xticklabels([])
    pl0.set_title('Freezing level (0 degC) Height [m] - BMKG')
    
    # pl0 = fig.add_subplot(412)
    # pl0.plot(pad(ts_CAPE/1000.),'.b-')
    # pl0.set_yticks(np.arange(0,5.1,0.5))
    # pl0.tick_params(axis='y',direction='out')
    # pl0.set_ylabel('CAPE [KJ]')
    # pl0.tick_params(axis='x',direction='out')    
    # pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    # strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    # #pl0.set_xticklabels(strdts)
    # pl0.set_xticklabels([])    
    # pl0.set_title('Convective Available Potential Energy [KJ] - BMKG')
    
    # pl1 = fig.add_subplot(413)
    # pl1.plot(pad(ts_CIN/1000.),'.k-')
    # pl1.set_yticks(np.arange(-0.21,0.01,0.03))
    # pl1.tick_params(axis='y',direction='out')
    # pl1.set_ylabel('CIN [KJ]')
    # pl1.tick_params(axis='x',direction='out')    
    # pl1.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    # strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    # #pl1.set_xticklabels(strdts)
    # pl1.set_xticklabels([])
    # pl1.set_title('Convective Inhibition [KJ] - BMKG')
            
    # pl2 = fig.add_subplot(414)
    # pl2.plot(pad(ts_TPW),'.r-')
    # pl2.set_yticks(np.arange(40,71,5))
    # pl2.tick_params(axis='y',direction='out')
    # pl2.set_ylabel('TPW [mm]')
    # pl2.tick_params(axis='x',direction='out')    
    # pl2.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))    
    # strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    # pl2.set_xticklabels(strdts)
    # pl2.set_title('Total Precipitable Water [mm] - BMKG')
    fig.savefig(dirout+'ts_freezelevel_cape_cin_tpw_interp.png', format='png', dpi=300) 

#--------------------------

    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-10,10.5,1)
    cs = pl0.contourf(ts_pot_temp_anomaly[0::nn,:],clevs, cmap="nipy_spectral", extend='both')
    pl0.contour(ts_pot_temp_anomaly[0::nn,:], levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    #pl0.plot(ts_height_temp0C,color='black',linewidth=1,linestyle='solid')
    pl0.set_yticks(np.arange(0,len(ts_temp)+1,500))
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels(new_press[0::500].astype(int))
    #pl0.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl0.set_ylabel('Pressure [mb]')    
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))
#    strdts = ['']*62
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)
    #pl0.set_xlabel('Datetime')
    pl0.set_title('Potential Temperature Anomaly [K] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs,orientation='vertical',ax=pl0)
    cbar.set_label('Potential Temperature Anomaly [K]')
    
    pl1 = fig.add_subplot(212)    
    clevs = np.arange(10,100.5,5)
    cs = pl1.contourf(ts_rh[0::nn,:],clevs, cmap="nipy_spectral", extend='min')
    #pl1.contour(ts_rh[0::nn,:], colors='black',linewidths=[0.5],linestyles='solid')
    pl1.set_yticks(np.arange(0,9600,500))
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels(new_press[0::500].astype(int))
    pl1.set_ylabel('Pressure [mb]')
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(0,len(np.transpose(ts_rh))+1,3*2))
#    strdts = ['']*62    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl1.set_xticklabels(strdts)
    #pl1.set_xlabel('Datetime')
    pl1.set_title('Relative Humidity [%] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl1)
    cbar.set_label('Relative Humidity [%]')
    fig.savefig(dirout+'ts_anom_pot_t_rh.png', format='png', dpi=300)         

    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-10,10.5,1)
    cs = pl0.contourf(tsh_pot_temp_anomaly[0::nn,:],clevs, cmap="nipy_spectral", extend='both')
    pl0.contour(tsh_pot_temp_anomaly[0::nn,:], levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl0.set_ylabel('Height [km]')
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))
#    strdts = ['']*62
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)
    #pl0.set_xlabel('Datetime')
    pl0.set_title('Potential Temperature Anomaly [K] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs,orientation='vertical',ax=pl0)
    cbar.set_label('Potential Temperature Anomaly [K]')
    
    pl1 = fig.add_subplot(212)    
    clevs = np.arange(10,100.5,5)
    cs = pl1.contourf(tsh_rh[0::nn,:],clevs, cmap="nipy_spectral", extend='min')
    pl1.set_yticks(idx)
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl1.set_ylabel('Height [km]')
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(0,len(np.transpose(ts_rh))+1,3*2))
#    strdts = ['']*62    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl1.set_xticklabels(strdts)
    #pl1.set_xlabel('Datetime')
    pl1.set_title('Relative Humidity [%] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl1)
    cbar.set_label('Relative Humidity [%]')
    fig.savefig(dirout+'tsh_anom_pot_t_rh.png', format='png', dpi=300)

    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-30,31,3)
    cs = pl0.contourf(ts_u[0::nn,:],clevs, cmap="nipy_spectral", extend='both')
    pl0.contour(ts_u[0::nn,:], levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(np.arange(0,9600,500))
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels(new_press[0::500].astype(int))
    pl0.set_ylabel('Pressure [mb]')
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_u))+1,3*2))
#    strdts = ['']*62    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)    
    #pl0.xlabel('Datetime')
    pl0.set_title('Zonal Wind [m/s] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl0)
    cbar.set_label('Zonal Wind [m/s]')
    #fig.savefig(dirout+'ts_u_wind.png', format='png', dpi=300) 
    
    #fig = pl.figure(figsize=(10,8))
    pl1 = fig.add_subplot(212)    
    clevs = np.arange(-30,31,3)
    cs = pl1.contourf(ts_v[0::nn,:],clevs, cmap="nipy_spectral", extend='both')
    pl1.contour(ts_v[0::nn,:], levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl1.set_yticks(np.arange(0,9600,500))
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels(new_press[0::500].astype(int))
    pl1.set_ylabel('Pressure [mb]')
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(0,len(np.transpose(ts_v))+1,3*2))
#    strdts = ['']*62 # 62 days
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl1.set_xticklabels(strdts)    
    #pl1.set_xlabel('Datetime')
    pl1.set_title('Meridional Wind [m/s] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl1)
    cbar.set_label('Meridional Wind [m/s]')
    #fig.savefig(dirout+'ts_v_vwind.png', format='png', dpi=300) 
    fig.savefig(dirout+'ts_u_v_wind.png', format='png', dpi=300)
    
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-30,31,3)
    cs = pl0.contourf(tsh_u[0::nn,:], clevs, cmap="nipy_spectral",extend='both')
    pl0.contour(tsh_u[0::nn,:], levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_u))+1,3*2))
#    strdts = ['']*62 # 62 days
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)
    pl0.set_title('Zonal Wind [m/s] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl0)
    cbar.set_label('Zonal Wind [m/s]')
    
    pl0 = fig.add_subplot(212)
    clevs = np.arange(-30,31,3)
    cs = pl0.contourf(tsh_v[0::nn,:],clevs, cmap="nipy_spectral",extend='both')
    pl0.contour(tsh_v[0::nn,:], levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_u))+1,3*2))
#    strdts = ['']*62 # 62 days
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)    
    #pl0.set_xlabel('Datetime')     
    pl0.set_title('Meridional Wind [m/s] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl0)
    cbar.set_label('Meridional Wind [m/s]')
    fig.savefig(dirout+'tsh_u_v_wind.png', format='png', dpi=300)
   
    # interpolated data
    ts_pot_temp_anomaly_interp = copy.deepcopy(ts_pot_temp_anomaly)
    ts_rh_interp = copy.deepcopy(ts_rh)
    tsh_pot_temp_anomaly_interp = copy.deepcopy(tsh_pot_temp_anomaly)
    tsh_rh_interp = copy.deepcopy(tsh_rh)
    ts_u_interp = copy.deepcopy(ts_u)
    ts_v_interp = copy.deepcopy(ts_v)
    tsh_u_interp = copy.deepcopy(tsh_u)
    tsh_v_interp = copy.deepcopy(tsh_v)
    
    ts_pot_temp_anomaly_interp = [pad(row) for row in ts_pot_temp_anomaly_interp[:]]
    ts_rh_interp = [pad(row) for row in ts_rh_interp[:]]
    tsh_pot_temp_anomaly_interp = [pad(row) for row in tsh_pot_temp_anomaly_interp[:]]
    tsh_rh_interp = [pad(row) for row in tsh_rh_interp[:]]
    ts_u_interp = [pad(row) for row in ts_u_interp[:]]
    ts_v_interp = [pad(row) for row in ts_v_interp[:]]
    tsh_u_interp = [pad(row) for row in tsh_u_interp[:]]
    tsh_v_interp = [pad(row) for row in tsh_v_interp[:]]
    
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-10,10.5,1)
    cs = pl0.contourf(ts_pot_temp_anomaly_interp,clevs, cmap="nipy_spectral", extend='both')
    pl0.contour(ts_pot_temp_anomaly_interp, levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    #pl0.plot(ts_height_temp0C,color='black',linewidth=1,linestyle='solid')
    pl0.set_yticks(np.arange(0,len(ts_temp)+1,500))
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels(new_press[0::500].astype(int))
    #pl0.set_yticklabels([round(h/1000.,1) for h in new_height[0::500]])
    pl0.set_ylabel('Pressure [mb]')    
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))
#    strdts = ['']*62
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)
    #pl0.set_xlabel('Datetime')
    pl0.set_title('Potential Temperature Anomaly [K] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs,orientation='vertical',ax=pl0)
    cbar.set_label('Potential Temperature Anomaly [K]')
    
    pl1 = fig.add_subplot(212)    
    clevs = np.arange(10,100.5,5)
    cs = pl1.contourf(ts_rh_interp,clevs, cmap="nipy_spectral", extend='min')
    #pl1.contour(ts_rh[0::nn,:], colors='black',linewidths=[0.5],linestyles='solid')
    pl1.set_yticks(np.arange(0,9600,500))
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels(new_press[0::500].astype(int))
    pl1.set_ylabel('Pressure [mb]')
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(0,len(np.transpose(ts_rh))+1,3*2))
#    strdts = ['']*62    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl1.set_xticklabels(strdts)
    #pl1.set_xlabel('Datetime')
    pl1.set_title('Relative Humidity [%] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl1)
    cbar.set_label('Relative Humidity [%]')
    fig.savefig(dirout+'ts_anom_pot_t_rh_interp.png', format='png', dpi=300)         

    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-10,10.5,1)
    cs = pl0.contourf(tsh_pot_temp_anomaly_interp,clevs, cmap="nipy_spectral", extend='both')
    pl0.contour(tsh_pot_temp_anomaly_interp, levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl0.set_ylabel('Height [km]')
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_temp))+1,3*2))
#    strdts = ['']*62
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)
    #pl0.set_xlabel('Datetime')
    pl0.set_title('Potential Temperature Anomaly [K] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs,orientation='vertical',ax=pl0)
    cbar.set_label('Potential Temperature Anomaly [K]')
    
    pl1 = fig.add_subplot(212)    
    clevs = np.arange(10,100.5,5)
    cs = pl1.contourf(tsh_rh_interp,clevs, cmap="nipy_spectral", extend='min')
    pl1.set_yticks(idx)
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl1.set_ylabel('Height [km]')
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(0,len(np.transpose(ts_rh))+1,3*2))
#    strdts = ['']*62    
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl1.set_xticklabels(strdts)
    #pl1.set_xlabel('Datetime')
    pl1.set_title('Relative Humidity [%] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl1)
    cbar.set_label('Relative Humidity [%]')
    fig.savefig(dirout+'tsh_anom_pot_t_rh_interp.png', format='png', dpi=300)

    pl.rcParams.update({'font.size': 22})
    fig = pl.figure(figsize=(20,15))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-30,31,5)
    # cs = pl0.contourf(ts_u_interp,clevs, cmap="nipy_spectral", extend='both')
    cs = pl0.contourf(ts_u_interp,clevs, cmap="RdBu_r", extend='both')
    pl0.contour(ts_u_interp, levels=[0],colors='black',linewidths=[1.0],linestyles='solid')
    pl0.set_yticks(np.arange(0,9600,1000))
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels(new_press[0::1000].astype(int))
    pl0.set_ylabel('Pressure [mb]')
    pl0.tick_params(axis='x',direction='out')    
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_u))+1,3*2))
#    strdts = ['']*62    
    strdts = [dt.datetime.strftime(date,'%d%b') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)    
    #pl0.xlabel('Datetime')
    # pl0.set_title('Zonal Wind [m/s] - BMKG')
    pl0.set_title('Zonal Wind')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl0)
    cbar.set_label('[m $s^{-1}$]')
    #fig.savefig(dirout+'ts_u_wind.png', format='png', dpi=300) 
    fig.savefig(dirout+'ts_u_wind_interp.png', format='png', dpi=300)
    
    #fig = pl.figure(figsize=(10,8))    
    # fig = pl.figure(figsize=(20,15))
    pl1 = fig.add_subplot(212)    
    clevs = np.arange(-30,31,5)
    # cs = pl1.contourf(ts_v_interp,clevs, cmap="nipy_spectral", extend='both')
    cs = pl1.contourf(ts_v_interp,clevs, cmap="RdBu_r", extend='both')
    pl1.contour(ts_v_interp, levels=[0],colors='black',linewidths=[1.0],linestyles='solid')
    pl1.set_yticks(np.arange(0,9600,1000))
    pl1.tick_params(axis='y',direction='out')
    pl1.set_yticklabels(new_press[0::1000].astype(int))
    pl1.set_ylabel('Pressure [mb]')
    pl1.tick_params(axis='x',direction='out')    
    pl1.set_xticks(np.arange(0,len(np.transpose(ts_v))+1,3*2))
#    strdts = ['']*62 # 62 days
    strdts = [dt.datetime.strftime(date,'%d%b') for date in dts[0::3*2]]
    pl1.set_xticklabels(strdts)    
    #pl1.set_xlabel('Datetime')
    # pl1.set_title('Meridional Wind [m/s] - BMKG')
    pl1.set_title('Meridional Wind')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl1)
    cbar.set_label('[m $s^{-1}$]')
    #fig.savefig(dirout+'ts_v_vwind.png', format='png', dpi=300) 
    fig.savefig(dirout+'ts_u_v_wind_interp.png', format='png', dpi=300, bbox_inches='tight')
    
    fig = pl.figure(figsize=(20,10))
    pl0 = fig.add_subplot(211)
    clevs = np.arange(-30,31,3)
    cs = pl0.contourf(tsh_u_interp, clevs, cmap="nipy_spectral",extend='both')
    pl0.contour(tsh_u_interp, levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_u))+1,3*2))
#    strdts = ['']*62 # 62 days
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)
    pl0.set_title('Zonal Wind [m/s] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl0)
    cbar.set_label('Zonal Wind [m/s]')
    
    pl0 = fig.add_subplot(212)
    clevs = np.arange(-30,31,3)
    cs = pl0.contourf(tsh_v_interp,clevs, cmap="nipy_spectral",extend='both')
    pl0.contour(tsh_v_interp, levels=[0],colors='black',linewidths=[0.5],linestyles='solid')
    pl0.set_yticks(idx)
    pl0.tick_params(axis='y',direction='out')
    pl0.set_yticklabels((new_height[idx]/1000).astype(float))    
    pl0.set_ylabel('Height [km]')    
    pl0.tick_params(axis='x',direction='out')
    pl0.set_xticks(np.arange(0,len(np.transpose(ts_u))+1,3*2))
#    strdts = ['']*62 # 62 days
    strdts = [dt.datetime.strftime(date,'%d%b%y') for date in dts[0::3*2]]
    pl0.set_xticklabels(strdts)    
    #pl0.set_xlabel('Datetime')     
    pl0.set_title('Meridional Wind [m/s] - BMKG')
    cbar = pl.colorbar(cs, ticks=clevs, orientation='vertical',ax=pl0)
    cbar.set_label('Meridional Wind [m/s]')
    fig.savefig(dirout+'tsh_u_v_wind_interp.png', format='png', dpi=300)
     
    
if __name__ == "__main__":
   main(sys.argv[1:])
