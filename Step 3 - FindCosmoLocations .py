import pandas as pd
import numpy as np
import dataframe_image as dfi
import os
import re
from heapq import merge
from operator import index
import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import numpy as np
import sklearn.neighbors
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


# Helper function to sort the files.
def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


# Extract cosmo location of all the stations.
def readindividualstation(station,filenames):
    columns = ['HH' ,  'PMSL' ,   'DF10M'  ,'DF500M' ,  'DF850'  , 'DF700'  , 'DF500' ,  'TG'  , 'T2M'  , 'TD2M' , 'T30M' , 'T850' , 'T700' , 'T500' ,'HML' ,'fog'  ,'HBAS' ,'HTOP'  ,'RR'    ,'RS',     'WS','h',   'hpa',  'dgr','degree', 'centrigrade','octas','10*m','mm','m']
    numberoffilesmerged = 0
    stationmerge = pd.DataFrame()
    # Skip the first 41 rows as they contain the header information.
    for f in filenames:
        data = pd.read_csv("./Cosmo/COSMO_UAE_025_2021_00/"+f,skiprows=41,header=None,sep="\s+")
        if len(data.columns)>30:
            del data[30]
        data.columns =columns
        for col in columns:
            if data[col].dtype == "object":
                data[col] = data[col].str.replace('/','')
                data[col] = data[col].astype(float)
        ff=open("./Cosmo/COSMO_UAE_025_2021_00/"+f)
        lines=ff.readlines()
        I = float(lines[31].split("I:")[1].split("J:")[0])
        J = float(lines[31].split("J:")[1])
        HSURF =  float(lines[33].split(":")[1])
        FR_LAND =float(lines[34].split(":")[1])
        LAT =float(lines[35].split(":")[1])
        LON = float(lines[36].split(":")[1])
    return LAT, LON

cosmolat = []
cosmolon = []
cosmostation = []

# Read all files for all cosmo location.
files = os.listdir("./Cosmo/COSMO_UAE_025_2021_00/")
files = sorted_nicely(files)
newfiles = []
print("Total number of file :",len(files))
stations = {}
count=0

# Extract the cosmo station name from all the files.
for f in files:    
    if "2021" in f:
        x = f.split("M_")[1]
        x = x.split("2021")[0]
        stations[x] = 1
        newfiles.append(f)        
print("Total number of stations in COSMO:",len(stations))

stations_with_complete_data = 0

# For each station read all the files. There are about 361 files for each station. 

for station  in list(stations.keys())[8:]:
    station_files = list(filter(lambda x: station in x, newfiles))
    
    # This is simple check which is specific this dataset only.
    if len(station_files)>=361:
        if station =="Ajman" and len(station_files)==722:
            station_files = list(filter(lambda x: "_City" not in x, station_files))
        stations_with_complete_data = stations_with_complete_data + 1
        #print("{:<20} {:<20}".format( station, len(station_files)))
        lat,lon = readindividualstation(station,station_files)
    # Add the latitude and longitude information.
    cosmolat.append(lat)
    cosmolon.append(lon)
    cosmostation.append(station)
    print(station,lat,lon)

cosmolocations = pd.DataFrame()
cosmolocations['lat'] = cosmolat
cosmolocations['lon'] = cosmolon
cosmolocations['file'] = cosmostation 

# Save the cosmo location.
cosmolocations.to_csv("./Cosmo/cosmolocations.csv",index=False)

# Find the distance between comso locations and rain locations.

def finddistance(cosmo,rain):
 # Convert the cosmo and rain location to radians.

 cosmo[['lat_radians_A','long_radians_A']] = (np.radians(cosmo.loc[:,['lat','lon']]))
 rain[['lat_radians_B','long_radians_B']] = (  np.radians(rain.loc[:,['Latitude','Longitude']]))
 print(cosmo.head())
 print(rain.head())
 
 # Use sklearn distance calculator function.
 dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')
 
 dist_matrix = (dist.pairwise(rain[['lat_radians_B','long_radians_B']],cosmo[['lat_radians_A','long_radians_A']])*3959)
 
 df_dist_matrix = (pd.DataFrame(dist_matrix,index=rain['Station'], 
                 columns=cosmo['file'])
 )
 
 df_dist_long = (pd.melt(df_dist_matrix.reset_index(),id_vars='Station'))
 
 df_dist_long = df_dist_long.rename(columns={'value':'miles'})
 a = []
 b = []
 for loop in df_dist_long['Station'].values:
  temp = rain[rain['Station']==loop]
  a.append(temp['Latitude'].values[0])
  b.append(temp['Longitude'].values[0])

 df_dist_long['Rainlat']= a
 df_dist_long['Rainlon']= b

 
 a = []
 b = []
 for loop in df_dist_long['file'].values:
  temp = cosmo[cosmo['file']==loop]
  a.append(temp['lat'].values[0])
  b.append(temp['lon'].values[0])
 
 df_dist_long['Cosmolat']= a
 df_dist_long['Cosmolon']= b
 print(df_dist_long.head())
 locationa = []
 locationb = []
 distance = []
 rainlat = []
 rainlon = []
 cosmolat = []
 cosmolon = []

 for station in cosmo['file'].values:
    temp = df_dist_long[df_dist_long['file']==station]
    temp = temp.sort_values('miles').iloc[0]
    locationa.append(temp['file'])
    locationb.append(temp['Station'])
    distance.append(temp['miles'])
    rainlat.append(temp['Rainlat'])
    rainlon.append(temp['Rainlon'])
    cosmolat.append(temp['Cosmolat'])
    cosmolon.append(temp['Cosmolon'])
    




    


 finaldata = pd.DataFrame()
 finaldata['Cosmolocation'] = locationa
 finaldata['Rainlocation'] = locationb
 finaldata['Distance'] = distance
 finaldata['Rainlat'] = rainlat
 finaldata['Rainlon'] =rainlon
 finaldata['Cosmolat'] =cosmolat
 finaldata['Cosmolon'] =cosmolon
 
 x = []
 y = []
 z = []
 rainlat = []
 rainlon = []
 cosmolat = []
 cosmolon = []
 for loop in rain['Station'].values:
  temp = finaldata[finaldata['Rainlocation']==loop]
  temp = temp[temp['Distance'].eq(temp['Distance'].min())].iloc[0]
  x.append(temp['Rainlocation'])
  y.append(temp['Cosmolocation'])
  z.append(temp['Distance'])
  rainlat.append(temp['Rainlat'])
  rainlon.append(temp['Rainlon'])
  cosmolat.append(temp['Cosmolat'])
  cosmolon.append(temp['Cosmolon'])
 finaldata = pd.DataFrame()
 finaldata['Rainlocation'] = x
 finaldata['Cosmolocation'] = y 
 finaldata['Distance'] = z
 finaldata['Rainlat'] = rainlat
 finaldata['Rainlon'] =rainlon
 finaldata['Cosmolat'] =cosmolat
 finaldata['Cosmolon'] =cosmolon

 finaldata.to_csv("Distance.csv",index=False)
 print(finaldata) 

rainlocation = pd.read_csv("./Rain/RainPointswithLocations.csv")
rainlocation = rainlocation[rainlocation['Rain Points']>50]
cosmolocations = pd.read_csv("./Cosmo/cosmolocations.csv")
finddistance(cosmolocations,rainlocation)

# After calculating the distance between rain guages and cosmo stations the next step is to
# read the cosmo stations informations and merge into one file.

def readstation(station,filenames,rainname):
    columns = ['HH' ,  'PMSL' ,   'DF10M'  ,'DF500M' ,  'DF850'  , 'DF700'  , 'DF500' ,  'TG'  , 'T2M'  , 'TD2M' , 'T30M' , 'T850' , 'T700' , 'T500' ,'HML' ,'fog'  ,'HBAS' ,'HTOP'  ,'RR'    ,'RS',     'WS','h',   'hpa',  'dgr','degree', 'centrigrade','octas','10*m','mm','m']
    numberoffilesmerged = 0
    stationmerge = pd.DataFrame()
    # Skip the first 41 rows as they contain the header information.
    for f in filenames:
        data = pd.read_csv("./Cosmo/COSMO_UAE_025_2021_00/"+f,skiprows=41,header=None,sep="\s+")
        if len(data.columns)>30:
            del data[30]
        data.columns =columns
        
        
        
        for col in columns:
            if data[col].dtype == "object":
                data[col] = data[col].str.replace('/','')
                data[col] = data[col].astype(float)
        ff=open("./Cosmo/COSMO_UAE_025_2021_00/"+f)
        lines=ff.readlines()
        I = float(lines[31].split("I:")[1].split("J:")[0])
        J = float(lines[31].split("J:")[1])
        HSURF =  float(lines[33].split(":")[1])
        FR_LAND =float(lines[34].split(":")[1])
        LAT =float(lines[35].split(":")[1])
        LON = float(lines[36].split(":")[1])
        SOILTYPE = str(lines[37].split(":")[1])
        data['I'] = [I]*len(data)
        data['J'] = [J]*len(data)
        data['HSURF'] = [HSURF]*len(data)
        data['FR_LAND'] = [FR_LAND]*len(data)
        data['LAT'] = [LAT]*len(data)
        data['LON'] = [LON]*len(data)
        data['SOILTYPE'] = [SOILTYPE]*len(data)
        # There are about 96*5 rows, but we considered only the 96 readings for each day.
        data  = data.head(96)
        stationmerge = pd.concat([stationmerge,data],axis=0,ignore_index=True)
    
    stationmerge.to_csv(rainname+os.sep+rainname+".csv",sep=",")  
    return LAT, LON

distance = pd.read_csv("Distance.csv")
cosmonames = distance['Cosmolocation'].values
rainnames = distance['Rainlocation'].values

# Cosmo files location.
files = os.listdir("./Cosmo/COSMO_UAE_025_2021_00/")
files = sorted_nicely(files)
files = [s for s in files if any(xs in s for xs in cosmonames)]


newfiles = []
print("Total number of file :",len(files))
stations = {}
count=0
for f in files:    
    if "2021" in f:
        x = f.split("M_")[1]
        x = x.split("2021")[0]
        stations[x] = 1
        newfiles.append(f)  
print("Total number of stations in COSMO:",len(stations))


for loop in range(0,len(cosmonames)):
 XX = [s for s in newfiles if any(xs in s for xs in [cosmonames[loop]])]
 if len(XX)>=361:
  if cosmonames[loop] =="Ajman" and len(XX)==722:
   XX = list(filter(lambda x: "_City" not in x, XX))
  lat,lon = readstation(cosmonames[loop],XX,rainnames[loop])
 print("Station: ",cosmonames[loop]," Done!") 

