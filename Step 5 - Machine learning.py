import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import numpy as np
import pandas as pd

import os
import re

import os
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay


np.seterr(all="ignore")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sys
from sklearn.model_selection import StratifiedKFold

# Read the rain threshold. For example, if the rain gauge value is above a specific threshold, then consider it rain; otherwise non-rain.
rainthreshold = float(sys.argv[1])

# Specify the algorithm name.
algorithm = sys.argv[2]

# There are five algorithms we considered.
names = [
 
    "LinearSVM",
    "DecisionTree",
    "RandomForest",
    "NeuralNet",
    "AdaBoost",
    "NaiveBayes",
]

 

if algorithm=="xgboost":
 model = XGBClassifier()
elif algorithm=="LinearSVM":
 model =  SVC(kernel="linear", C=0.025)
elif algorithm=="DecisionTree":
 model =  DecisionTreeClassifier()
elif algorithm=="AdaBoost":
 model =  AdaBoostClassifier()
elif algorithm=="NaiveBayes":
 model =  GaussianNB()


 
# Specify the classes.

# No Rain - No Cloud
# No Rain - Cloud
# Rain - No Cloud
# Rain - Cloud

def f(x):
  if x['RAIN'] == 0 and x['Cloud'] == 1: return 0
  elif x['RAIN'] == 1 and x['Cloud'] == 1 : return 1
  else: return 2



def getnumericalvalues(XX,threshold):
 XX =XX
 finaldata = pd.read_csv(XX+os.sep+"final_"+XX+".csv")
 finaldata.fillna(0)
 threshold = threshold
 finaldata['RAIN'].loc[(finaldata['RAIN'] > rainthreshold)] = 1
 finaldata.loc[finaldata['Cloud'] < threshold, 'Cloud'] = 0
 finaldata.loc[finaldata['Cloud'] >= threshold, 'Cloud'] = 1

 temp = finaldata[(finaldata['RAIN'] == 0) & (finaldata['Cloud'] == 0)]
 a = len(temp)
 temp = finaldata[(finaldata['RAIN'] == 0) & (finaldata['Cloud'] == 1)]
 b=len(temp)
 temp = finaldata[(finaldata['RAIN'] == 1) & (finaldata['Cloud'] == 0)]
 c=len(temp)
 temp = finaldata[(finaldata['RAIN'] == 1) & (finaldata['Cloud'] == 1)]
 d = len(temp)
 return a,b,c,d


weights = [1, 10, 25, 50, 75, 99, 100, 200]
param_grid = dict(scale_pos_weight=weights)

def splitdata(XX,threshold):
 XX =XX
 finaldata = pd.read_csv(XX+os.sep+"final_"+XX+".csv")
 finaldata.fillna(0)
 threshold = threshold
 finaldata['RAIN'].loc[(finaldata['RAIN'] > rainthreshold)] = 1
 finaldata.loc[finaldata['Cloud'] < threshold, 'Cloud'] = 0
 finaldata.loc[finaldata['Cloud'] >= threshold, 'Cloud'] = 1
 finaldata = finaldata[~((finaldata['RAIN'] == 0) & (finaldata['Cloud'] == 0))]
 finaldata['Category'] = finaldata.apply(f, axis=1)
 finaldata = finaldata[~(finaldata['Category'] == 2)]
 del finaldata['Unnamed: 0.1']
 del finaldata['Unnamed: 0']
 del finaldata['RAIN']
 del finaldata['Cloud']
 y = finaldata['Category'].values
 del finaldata['Category']
 del finaldata['SOILTYPE']
 del finaldata['HH']
 del finaldata['LAT']
 del finaldata['LON']

 return finaldata.values,y

def predictionfunction(XX):
 testAUC = []
 trainAUC = []
 values = []
 scoress = []
 traincf = []
 testcf = []
 values1 = []
 values2 = []
 values3 = []
 values4 = []
 samplesclassoneintrain = []
 samplesclasstwointrain = []
 samplesclassoneintest = []
 samplesclasstwointest = []
 
 # Iterate through various cloud thresholds.
 for loop in range(0,100,5): 
  x = getnumericalvalues(XX,loop)
  values1.append(x[0])
  values2.append(x[1])
  values3.append(x[2])
  values4.append(x[3])
  data,y = splitdata(XX,loop)
  data = scaler.fit_transform(data)
  X_train, X_test, y_train, y_test = train_test_split(data, y, train_size=0.67)
  samplesclassoneintrain.append(np.count_nonzero(y_train==0))
  samplesclasstwointrain.append(np.count_nonzero(y_train==1))
  samplesclassoneintest.append(np.count_nonzero(y_test==0))
  samplesclasstwointest.append(np.count_nonzero(y_test==1))
  from sklearn.utils import class_weight

  weights = np.ones(len(y_train), dtype = 'float')
  skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
  
  temptest = []
  temptrain = []
  temptraincf_matrix = [[0,0],[0,0]]
  temptestcf_matrix = [[0,0],[0,0]]

  for train_index, test_index in skf.split(data, y):
   X_train, X_test = data[train_index], data[test_index]
   y_train, y_test = y[train_index], y[test_index]
   
   sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
   )
   model.fit(X_train, y_train, sample_weight=sample_weights)

   y_pred = model.predict(X_test)
   predictions = [round(value) for value in y_pred]
   try:
    accuracy = accuracy_score(y_test, predictions)
    temptrain.append(int( (roc_auc_score(y_train, [round(value) for value in model.predict(X_train)]) * 100.0)))
    temptest.append(int((roc_auc_score(y_test, [round(value) for value in model.predict(X_test)]) * 100.0)))

    temptraincf_matrix = temptraincf_matrix + confusion_matrix(y_train, model.predict(X_train))
    temptestcf_matrix = temptestcf_matrix + confusion_matrix(y_test, model.predict(X_test))
   except:
    accuracy = accuracy_score(y_test, predictions)
    temptrain.append(0)
    temptest.append(0)

    temptraincf_matrix = [[0,0],[0,0]]
    temptestcf_matrix = [[0,0],[0,0]]
 

  print(mean(temptrain),mean(temptest))
  try:
   traincf.append(temptraincf_matrix/5)
   testcf.append(temptestcf_matrix/5)
  except:
   traincf.append([[0,0],[0,0]])
   testcf.append([[0,0],[0,0]])
  
  trainAUC.append(mean(temptrain))
 
  testAUC.append(mean(temptest))
  values.append(loop)


 results = pd.DataFrame()
 results['Threshold'] = values
 results['TrainAUC'] = trainAUC
 results['TestAUC'] = testAUC
 results['Test CM'] = testcf
 results['Train CM'] = traincf
 results['NoRain_NoCloud'] = values1
 results['NoRain_Cloud'] = values2
 results['Rain_NoCloud'] = values3
 results['Rain_Cloud'] = values4
 results['S0train'] =  samplesclassoneintrain
 results['S1train'] = samplesclasstwointrain
 results['S0test'] = samplesclassoneintest
 results['S1test'] = samplesclasstwointest 
 
 results.to_csv(XX+os.sep+str(algorithm)+str(rainthreshold)+".csv",index=False)
 print("Station:",XX)
 print(results.to_markdown())

distance = pd.read_csv("Distance.csv")
rainnames = distance['Rainlocation'].values
cosmonames = distance['Cosmolocation'].values

 
for loop in range(0,len(rainnames)):
 # For each rain station repeat the process.
 predictionfunction(rainnames[loop])



