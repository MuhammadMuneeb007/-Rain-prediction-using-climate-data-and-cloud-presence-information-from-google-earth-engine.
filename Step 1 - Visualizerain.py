import pandas as pd
import numpy as np
import dataframe_image as dfi


rain_points = []
nonrain_points = []
stationname = []
latitude = []
longitude = []

# The first step is to read the station's rain gauge data containing the rain information.
rainpoints = pd.read_csv('./Rain/2021.csv',index_col=None).iloc[:,6:]

# If the reading exceeds 1, consider it a rain point.
rainpoints[rainpoints>0] =1

print(rainpoints.head())
print("Number of Rain stations: ",len(rainpoints.columns))

# Read the file containing the rain gauge location.
rainlocation = pd.read_csv('./Rain/Preprocessed_location_rain.csv',header=None)
print("Number of Rain stations locations: ",rainlocation.shape[0])
print(rainlocation.head())

# For each rain point, extract the corresponding location.
for gname in rainpoints.columns:
 try:
  # Extract the rain gauge station name.
  station_name = gname[:-1]
  
  # Extract the rain gauge station location and match both names.  
  temp = rainlocation[rainlocation[2]==station_name] 
  
  ones = (rainpoints[gname] == 1).sum()
  zeros =(rainpoints[gname] == 0).sum()
  stationname.append(station_name)
  rain_points.append(ones)
  nonrain_points.append(zeros)
  latitude.append(str(temp[3].values[0]))
  longitude.append(str(temp[4].values[0]))
 except:
  latitude.append(str("NA"))
  longitude.append(str("NA"))
  print("Station: ",gname," does not exist in the Rain location!")

visualize = pd.DataFrame()
visualize['Station'] = stationname
visualize['Rain Points'] = rain_points
visualize['Non-Rain Points'] = nonrain_points
visualize['Latitude'] = latitude
visualize['Longitude'] = longitude
visualize = visualize[visualize['Latitude']!="NA"]
visualize = visualize.sort_values('Rain Points')
print("Number of Rain Stations with location information: ",visualize.shape)
print(visualize.to_markdown())
visualize.to_html("./Rain/X.html")
visualize.to_csv("./Rain/RainPointswithLocations.csv",index=False)



