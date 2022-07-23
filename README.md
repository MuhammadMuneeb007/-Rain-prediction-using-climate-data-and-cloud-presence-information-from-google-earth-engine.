# -Rain-prediction-using-climate-data-and-cloud-presence-information-from-google-earth-engine.
 This repository presents a machine learning-based methodology for finding whether a cloud can be seeded or not for rain.  
 
 The dataset associate with repository is owned by someone else.
 Kindly send email to hussam.alhamadi@ku.ac.ae
 Page: https://www.ku.ac.ae/college-people/dr-hussam-al-hammadi

Request here:
Google drive link: https://drive.google.com/drive/folders/1IOE-iYwoT5wy4w15holkkjTh1SADY15t?usp=sharing



Each folder is the name of station and contains the results for each station for 5 algorithms.




\begin{enumerate}
    \item Threshold: 	Cloud threshold (ranges from 0 to 100 with an interval of 5).
 \item TrainAUC: Training Area under the ROC Curve (With 4-fold stratified cross-validation).
 \item TestAUC: Test Area under the ROC Curve (With 5-fold stratified cross-validation).
 \item Test CM: Test confusion matrix (Average of 5-folds). If the confusion matrix for the particular row is missing, then it means after the cloud threshold, there is only one category left. 
    	
 \item Train CM: Training confusion matrix (Average of 5-folds). If the confusion matrix for the particular row is missing, then it means there is only one category left after the cloud threshold. 
 
 \item NoRain\_NoCloud: Number of samples for no rain and no cloud for a particular cloud threshold.	
 \item NoRain\_Cloud: Number of samples for no rain and cloud for a particular cloud threshold. 	
 \item Rain\_NoCloud: Number of samples for rain and no cloud for a particular cloud threshold.	
 \item Rain\_Cloud: Number of samples for rain and cloud for a particular cloud threshold.	
 \item S0train: Number of instances of class NoRain\_Cloud in training data.
 \item S1train: Number of instances of class Rain\_Cloud in test data. 	
 \item S0test: Number of instances of class NoRain\_Cloud in test data.	
 \item S1test: Number of instances of class Rain\_Cloud in test data.	
\end{enumerate}

