import pandas as pd
import numpy as np

distance = pd.read_csv("Distance.csv")
rainnames = distance['Rainlocation'].values
cosmonames = distance['Cosmolocation'].values

def extractresults(XX):
 import os
 files = os.listdir(XX)
 max = 0
 file= ""
 for f in files:
    if "." in f.split(".csv")[0] and "lock" not in f.split(".csv")[0]:
        data = pd.read_csv(XX+os.sep+f)
        if max< data.iloc[0,:]['TestAUC']:
            max = data.iloc[0,:]['TestAUC']
            file = f
 print("\n\nX: ",file,"Accuracy: ", max,"\n\n")
        
for loop in range(0,len(rainnames)):
 print("Results for:", rainnames[loop])
 extractresults(rainnames[loop])
 #exit(0)
