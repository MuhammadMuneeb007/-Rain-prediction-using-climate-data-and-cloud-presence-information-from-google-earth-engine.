# -Rain-prediction-using-climate-data-and-cloud-presence-information-from-google-earth-engine.
 This repository presents a machine learning-based methodology for finding whether a cloud can be seeded or not for rain.  
 
 The dataset associate with repository is owned by someone else.
 Kindly send email to hussam.alhamadi@ku.ac.ae
 Page: https://www.ku.ac.ae/college-people/dr-hussam-al-hammadi

Request here:
Google drive link: https://drive.google.com/drive/folders/1IOE-iYwoT5wy4w15holkkjTh1SADY15t?usp=sharing



The supplementary files are in this format. There are five folders, each corresponds to a station, and each folder contains 15 files. 
The file name represent two things: machine learning algorithm name and rain threshold in this format ALGORITHMRainThreshold.csv (Example: Xgboost0.1.csv)


Threshold: 	Cloud threshold (ranges from 0 to 100 with an interval of 5).
TrainAUC: Training Area under the ROC Curve (With 4-fold stratified cross-validation).
TestAUC: Test Area under the ROC Curve (With 5-fold stratified cross-validation).
Test CM: Test confusion matrix (Average of 5-folds). If the confusion matrix for the particular row is missing, then it means after the cloud threshold, there is only one category left. 
    	
Train CM: Training confusion matrix (Average of 5-folds). If the confusion matrix for the particular row is missing, then it means there is only one category left after the cloud threshold. 
 
NoRain\_NoCloud: Number of samples for no rain and no cloud for a particular cloud threshold.	
NoRain\_Cloud: Number of samples for no rain and cloud for a particular cloud threshold. 	
Rain\_NoCloud: Number of samples for rain and no cloud for a particular cloud threshold.	
Rain\_Cloud: Number of samples for rain and cloud for a particular cloud threshold.	
S0train: Number of instances of class NoRain\_Cloud in training data.
S1train: Number of instances of class Rain\_Cloud in test data. 	
S0test: Number of instances of class NoRain\_Cloud in test data.	
S1test: Number of instances of class Rain\_Cloud in test data.	


Results for: Khatam Al Shaklah
X:  NaiveBayes0.2.csv Accuracy:  72.2

Results for: Kalba
X:  LinearSVM0.0.csv Accuracy:  79.2

Results for: Wadi Al Tuwa
X:  LinearSVM0.0.csv Accuracy:  71.0

Results for: Hatta
X:  LinearSVM0.0.csv Accuracy:  87.6

Results for: Al Heben
X:  LinearSVM0.0.csv Accuracy:  71.6
