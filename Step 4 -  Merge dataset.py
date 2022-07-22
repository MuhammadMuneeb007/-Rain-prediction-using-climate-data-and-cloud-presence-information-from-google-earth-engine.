import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import numpy as np
import pandas as pd
import ee
import ee
import geemap
import os
import re
import rasterio
from rasterio.plot import show
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show_hist
from rasterio.plot import show_hist

def filterrainfiledates(XX):
 rain = pd.read_csv("./Rain/2021.csv")
 rain["Date"] = pd.to_datetime(rain[["Year", "Month", "Day"]]).dt.strftime("%Y%m%d") 
 #rain['Date'] = rain['Date'].strftime("%Y%m%d") 
 rain['Date'] = rain['Date'].astype(str)
 rain=rain[rain.Date.isin(XX)]
 print(rain.shape)
 return rain

def filterrainfilestations(XX,rain):
 distance  = pd.read_csv("Distance.csv")
 temp = distance[distance['Cosmolocation']== XX]
 temp = rain[temp['Rainlocation'].values[0]+" "].values 
 return temp

def readcosmofilesdates(XX):
 XX =XX 
 cosmofiles = os.listdir("./Cosmo/COSMO_UAE_025_2021_00/")
 cosmofiles = sorted_nicely(cosmofiles)
 newfiles = []
 stations = {}
 count=0
 
 for f in cosmofiles:    
  if "2021" in f and XX in f:
   x = f.split("M_")[1]
   x = x.split("2021")[0]
   stations[x] = 1
   newfiles.append(f)
 newfiles = sorted_nicely(newfiles)
 
 newimages = []
 from datetime import datetime
 for image in newfiles:
  #newimages.append(datetime.strptime(re.findall(r'\d+',image)[0][:-2],'%Y%m%d'))
  newimages.append(re.findall(r'\d+',image)[0][:-2])

 newimages = list(dict.fromkeys(newimages))
 return newimages


def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def filterimagesdates(XX,dates):
 finalvalues = []
 images = os.listdir(XX+os.sep+"images")
 newimages = []
 for date in dates:
  newimages.extend(filter(lambda x: date in x, images))
 newimages = sorted_nicely(newimages)
 for image in newimages: 
  img = rasterio.open('./'+XX+os.sep+"images"+os.sep+image) 
  finalvalues.extend(np.repeat(img.read()[0][0][0], 24))
 return finalvalues

distance = pd.read_csv("Distance.csv")
rainnames = distance['Rainlocation'].values
cosmonames = distance['Cosmolocation'].values
 
for loop in range(0,len(rainnames)):
 # Extract the cosmo station name.
 XX = cosmonames[loop]

 # Extract the rain gauge name.
 YY = rainnames[loop]

 # Read the cosmo dates, rain dates, and google earth engine images dates to merge the data with common dates.
 cosmodates = readcosmofilesdates(XX)

 # Read comso dates.
 raindata = filterrainfiledates(cosmodates)
 
 # Read the rain gauge values.
 rainvalues = filterrainfilestations(XX, raindata)
 mergeddata = pd.read_csv(YY+os.sep+YY+".csv")
 mergeddata['RAIN'] = rainvalues

 # For cloud data, there are four images for each day, so we extended these four images
 # to 96 readings. 4 * 24 = 96.
 # For rain and cosmo data, there are about 96 readings, so we have to sync the data. 

 cloud = filterimagesdates(YY,cosmodates)
 mergeddata['Cloud'] = cloud
 print(mergeddata.shape)
 print(mergeddata.head())
 mergeddata.to_csv(YY+os.sep+"final_"+YY+".csv")


