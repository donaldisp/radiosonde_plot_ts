# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:44:59 2017

@author: donaldi permana
Puslitbang BMKG
donaldi.permana@bmkg.go.id
"""
#from IPython import get_ipython
#get_ipython().magic('clear') # clear screen
#get_ipython().magic('reset -sf')
 
from mpl_toolkits.basemap import Basemap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from skewt import SkewT
#from __future__ import print_function, division
#from SkewTplus.skewT import figure
#from SkewTplus.sounding import sounding
from numpy import interp
import wradlib as wrl
import numpy as np
import matplotlib.pyplot as pl
import math
import csv
import pandas as pd
#def dependencies_for_myprogram():
    #import xmltodict

# just making sure that the plots immediately pop up
#pl.interactive(True)  # noqa
#import pylab as pyl
import datetime as dt
from osgeo import osr
from netCDF4 import Dataset, date2num
import glob
import os
import gc
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

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
       opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
       print 'test.py -i <inputfile> -o <outputfile>'
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print 'test.py -i <inputfile> -o <outputfile>'
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg
       elif opt in ("-o", "--ofile"):
          outputfile = arg
    print 'Input file is "', inputfile
#    print 'Output file is "', outputfile

#    path = 'L:\Radar2016\sample_data\pky1.vol'
    path = inputfile
    files = sorted(glob.glob(path))
    
    tstart = dt.datetime.now()
    for filename in files: 
        print 'Creating folder '+filename+'.dir ...'
        
        #filename = 'n:\\rason1\F2017111606S6008157.csv'
        
        #StaNo = filename[-20:-15]
        StaNo = '96253'
        
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
        
        newdir = filename+'.dir'
        try:
            os.mkdir(newdir, 0755 );
        except:
            print newdir+' already exists'
        
        df = pd.read_csv(filename, skiprows=6)
                
        obstime = df[df.columns[0]].map(lambda x: dt.datetime.strptime(str(x), '%H:%M:%S')) #ObsTime
        height = pad(df[df.columns[11]].replace('-----',np.nan).astype(float)) #Height
        lat = pad(df[df.columns[17]].replace('-----',np.nan).astype(float)) #GeodetLat
        lon = pad(df[df.columns[18]].replace('-----',np.nan).astype(float)) #GeodetLon
        wd = pad(df[df.columns[9]].replace('-----',np.nan).astype(float)) #Wind Direction
        ws = pad(df[df.columns[10]].replace('-----',np.nan).astype(float)) # Wind Speed
        press = df[df.columns[20]].replace('-----',np.nan) #Pressure
        press = pad(press.replace('------',np.nan).astype(float))
        temp = pad(df[df.columns[21]].replace('-----',np.nan).astype(float)) #Temperature
        rh = pad(df[df.columns[22]].replace('-----',np.nan).astype(float)) #Relative Humidity
        dewtemp = temp - ((100 - rh)/5.) # Dewpoint temperature
        
        #_AllowedKeys=['pres','hght','temp','dwpt','relh','mixr','drct','sknt','thta','thte','thtv']
        
        mydata=dict(zip(('StationNumber','SoundingDate','hght','pres','temp','dwpt','relh','drct','sknt'),\
        ('Bengkulu ('+StaNo+')', strdt_rason +'UTC', height, press,temp,dewtemp,rh,wd,ws)))
        S=SkewT.Sounding(soundingdata=mydata)
        
        #S.do_thermodynamics()
        
        #S.plot_skewt(tmin=-40.,tmax=40.,pmin=100.,pmax=1050.,parcel_type='sb')
        
        try:
            #S.plot_skewt(tmin=-40.,tmax=40.,pmin=100.,pmax=1050.,parcel_type='mu')
            ##S.plot_skewt(parcel_type='sb',imagename='f:/rason/test.png')
            
            #S.fig.savefig(newdir+'\\'+StaNo+'_skewt-mu_'+strdt_rason+'.png', format='png', dpi=500)
            
            #S.plot_skewt(tmin=-40.,tmax=40.,pmin=100.,pmax=1050.,parcel_type='ml')
            ##S.plot_skewt(parcel_type='sb',imagename='f:/rason/test.png')
            
            #S.fig.savefig(newdir+'\\'+StaNo+'_skewt-ml_'+strdt_rason+'.png', format='png', dpi=500) 
            
            S.plot_skewt(tmin=-40.,tmax=40.,pmin=100.,pmax=1050.,parcel_type='sb')
            #S.plot_skewt(parcel_type='sb',imagename='f:/rason/test.png')
            
            S.fig.savefig(newdir+'\\'+StaNo+'_skewt-sb_'+strdt_rason+'.png', format='png', dpi=500) 
        except:            
            #S.make_skewt_axes()
            S.make_skewt_axes(tmin=-40.,tmax=40.,pmin=100.,pmax=1050.)
            #S.make_skewt_axes(tmin=-50,tmax=40,pmin=100)
            S.add_profile(lw=1)
            #pc=S.most_unstable_parcel()
            #sounding.lift_parcel(*pc,totalcape=True)
        
                        
        #S.fig.savefig('f:/rason/test.png', format='png', dpi=500) 
        #S.fig.savefig(newdir+'\\'+strdt_rason+'UTC-bengkulu-skewt.png', format='png', dpi=500)         
               
        #parcel=S.get_parcel()
        #S.lift_parcel(*parcel)
        #S.get_cape(*parcel)
        
        # Create a Figure Manager 
        #mySkewT_Figure = figure()

        # Add an Skew-T axes to the Figure
        #mySkewT_Axes = mySkewT_Figure.add_subplot(111, projection='skewx')
        
        # Add a profile to the Skew-T diagram
        #mySkewT_Axes.addProfile(press.astype(float),temp.astype(float), dewtemp.astype(float) ,
        #                hPa=True, celsius=True, method=1, diagnostics=True, useVirtual=0)
 
        # Show the figure
        #mySkewT_Figure.show()

        print 'Plotting figures in '+filename+'.dir ...'
        
        # draw figure
        fig = pl.figure(figsize=(10,8))        
        
        pl.plot(temp.astype(float),height.astype(float)/1000,'b')
        pl.plot(dewtemp.astype(float),height.astype(float)/1000,'r')
        #pl.xlabel('Temperature (C)')
        pl.ylabel('Height (km)',fontsize=12)
        pl.ylim(0,30)
        pl.xlim(-120,40)
        pl.text(-100,-3,'Temperature (C)',color='b',fontsize=12)
        pl.text(-50,-3,'Dew Point Temperature (C)',color='r',fontsize=12)
        pl.title('Temperature '+HH+'UTC '+DD+'-'+MM+'-'+YY+' LON='+rason_lon+',LAT='+rason_lat,fontsize=12)
        #cbar = pl.colorbar(pm, shrink=0.75)
        fig.savefig(newdir+'\\'+StaNo+'_t_'+strdt_rason+'.png', format='png', dpi=500) 
            
        
        fig = pl.figure(figsize=(10,8))        
        
        pl.plot(rh.astype(float),height.astype(float)/1000,'b')
        #pl.plot(dewtemp.astype(float),height.astype(float)/1000,'r')
        #pl.xlabel('Relative Humidity (%)')
        pl.ylabel('Height (km)',fontsize=12)
        pl.ylim(0,30)
        pl.xlim(0,100)
        pl.text(10,-3,'Relative Humidity (%)',color='b',fontsize=12)
        #pl.text(-100,4000,'Dew point temperature',color='r')
        pl.title('Humidity '+HH+'UTC '+DD+'-'+MM+'-'+YY+' LON='+rason_lon+',LAT='+rason_lat,fontsize=12)
        #cbar = pl.colorbar(pm, shrink=0.75)
        fig.savefig(newdir+'\\'+StaNo+'_rh_'+strdt_rason+'.png', format='png', dpi=500) 
            

        fig = pl.figure(figsize=(10,8))        
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        
        ax1.plot(wd.astype(float),height.astype(float)/1000,'y')
        ax1.set_xlim(0,360)
        ax1.set_xticks(np.arange(0,361,45))
        ax1.set_ylim(0,30)
        
        ax2.plot(ws.astype(float),height.astype(float)/1000,'g')
        ax2.set_xlim(0,50)
        ax2.set_ylim(0,30)
        #pl.xlabel('Temperature (C)')
        ax1.set_ylabel('Height (km)',fontsize=12)
        ax1.text(45,-3,'Wind Direction (deg)',color='y',fontsize=12)
        ax2.text(5,32,'Wind Speed (m/s)',color='g',fontsize=12)
        ax2.text(23,32,'Wind '+HH+'UTC '+DD+'-'+MM+'-'+YY+' LON='+rason_lon+',LAT='+rason_lat,fontsize=12)
        #pl.title('Wind',fontsize=12)
        #cbar = pl.colorbar(pm, shrink=0.75)
        
        fig.savefig(newdir+'\\'+StaNo+'_wind_'+strdt_rason+'.png', format='png', dpi=500) 
            
        

if __name__ == "__main__":
   main(sys.argv[1:])
