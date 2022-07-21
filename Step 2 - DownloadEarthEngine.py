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
import time
from multiprocessing import Process
import ee
import pandas as pd
import ee
import ee
import geemap
import os
import rasterio
from rasterio.plot import show
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show_hist
from rasterio.plot import show_hist


ee.Initialize()

# Helper function to convert the square into earth engine polygon.

def findpolygon_earthengine(row):
 return [[row['topleft_long'],row['topleft_lat']],
 [row['topright_long'],row['topright_lat']],
 [row['bottomright_long'],row['bottomright_lat']],
 [row['bottomleft_long'],row['bottomleft_lat']], 
 [row['topleft_long'],row['topleft_lat']]]

# Helper function to convert the square into geopandas polygon.
def findpolygon_geopandas(row):
  return Polygon([( row['topleft_long'],row['topleft_lat'])
    , ( row['topright_long'],row['topright_lat'])
    ,(row['bottomright_long'],row['bottomright_lat'] )
    , ( row['bottomleft_long'],row['bottomleft_lat'])
    , (row['topleft_long'],row['topleft_lat'] )])

def makeboxes(data):
  # This constant value is added to each longitude and latitude value to make a polygon.
  constant = 0.1
  data['topleft_lat'] = data['Latitude'] - constant
  data['topleft_long'] = data['Longitude'] -constant
  data['topright_lat'] = data['Latitude'] - constant
  data['topright_long'] = data['Longitude'] + constant
  data['bottomleft_lat'] = data['Latitude'] + constant
  data['bottomleft_long'] = data['Longitude'] - constant
  data['bottomright_lat'] = data['Latitude'] +constant
  data['bottomright_long'] = data['Longitude'] + constant
  data['p'] = data.apply(findpolygon_geopandas, axis=1)
  data['coords'] = data.apply(findpolygon_earthengine, axis=1)
  return data


def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


# Read the rain points with the location.
rainlocation = pd.read_csv("./Rain/RainPointswithLocations.csv")

# Consider only those rain points for which the rain points are more than 70. 
rainlocation = rainlocation[rainlocation['Rain Points']>70]
rainlocation = makeboxes(rainlocation)

# Here, we defined a specific dataset containing cloud presence information.
# Define the google earth engine dataset and the corresponding band.
names = ['NOAA/NCEP_DOE_RE2/total_cloud_coverage']
bands = ['tcdc']
datasetnames = pd.DataFrame()
datasetnames['name'] = names
datasetnames['bands'] = bands

# Define the starting and the ending range.
i_date = '2021-01-01'
f_date =  '2022-01-01'

# Use this function to download earth engine images for each station separately.
def downloadindividualdataset():
 for index, row in rainlocation.iterrows():
  roi = ee.Geometry.Polygon(row['coords'])
  for index2,row2 in datasetnames.iterrows():
   # Read the earth engine dataset.
   collection = (ee.ImageCollection(row2['name']).select(row2['bands']).filterDate(i_date ,f_date).filterBounds(roi))
   
   # Crop the specific portion of an image.
   collection =  collection.map(lambda image: image.clip(roi))
   
   # Make a directory with a station name to store the images.  
   if not os.path.isdir(row['Station']): 
    os.mkdir(row['Station'])   
   directoryname = row['Station']+os.sep+"images" 
   if not os.path.isdir(directoryname): 
    os.mkdir(directoryname)
   print(collection.aggregate_array('system:index').getInfo())
   geemap.ee_export_image_collection(collection, out_dir=directoryname)
   geemap.ee_export_image_collection_to_drive(collection, folder='export', scale=30)


def worker(row):
 roi = ee.Geometry.Polygon(row['coords'])
 for index2,row2 in datasetnames.iterrows():
  collection = (ee.ImageCollection(row2['name']).select(row2['bands']).filterDate(i_date ,f_date).filterBounds(roi))
  collection =  collection.map(lambda image: image.clip(roi))
  if not os.path.isdir(row['Station']): 
   os.mkdir(row['Station'])
  directoryname = row['Station']+os.sep+"images" 
  if not os.path.isdir(directoryname): 
   os.mkdir(directoryname)
  geemap.ee_export_image_collection(collection, out_dir=directoryname)
  geemap.ee_export_image_collection_to_drive(collection, folder='export', scale=30)  


# Use this function to download images for all the stations in parallel
# The issue with this function is that it may not read all the images for some dates, and you have to re-download them.
def runInParallel():
    proc = []
    for index, row in rainlocation.iterrows():
        p = Process(target=worker, args=(row,))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

runInParallel()























