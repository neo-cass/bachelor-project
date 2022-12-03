import sklearn as sk
import pandas as pd
import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from timeit import default_timer as timer

config="ECLF1" #The name given to the classifier in the CSV file.
dataset = 1 #The dataset we are testing over.
cv=5 #Number of folds in cross validation.
y_probs = []
y_preds = []
y_tests = []
scoreTimes = []
fitTimes = []

#Single classifiers
SVM = SVC(kernel='linear', probability=True)
RF = RandomForestClassifier()
GNB = GaussianNB()

#Ensemble classifiers
eclf = VotingClassifier(estimators=[
       ('svm',SVM),('rf',RF),('gnb',GNB)], voting='soft', n_jobs=-1)

data = pd.read_csv("Datasets/Dataset"+str(dataset)+".csv")
if dataset==1:
    X, y = data.iloc[:,1:49], data.iloc[:,49]
if dataset==2:    
    X, y = data.iloc[:,1:31], data.iloc[:,31]

skf = StratifiedKFold(n_splits = cv) 

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    fitTime1 = timer()
    eclf.fit(X_train, y_train)
    fitTime2=timer()
    fitTime = fitTime2-fitTime1
    fitTimes.append(fitTime)
    scoreTime1=timer()
    y_pred = eclf.predict(X_test)
    scoreTime2=timer()
    scoreTime = scoreTime2-scoreTime1
    scoreTimes.append(scoreTime)
    y_prob = eclf.predict_proba(X_test)
    y_preds+=y_pred.tolist()
    y_tests+=y_test.tolist()
    y_probs+=y_prob[:,1].tolist() #Gets the positive class predictions.


conf_matrix = confusion_matrix(y_tests, y_preds)

TN=conf_matrix[0,0]
FP=conf_matrix[0,1]
FN=conf_matrix[1,0]
TP=conf_matrix[1,1]
#Sensitivity
Sensitivity = TP/(TP+FN)
#Specificity
Specificity = TN/(TN+FP) 
#Precision
Precision = TP/(TP+FP)
#Negative predictive value
NPV = TN/(TN+FN)
#Fall out
Fallout = FP/(FP+TN)
#False negative rate
FNR = FN/(TP+FN)
#False discovery rate
FDR = FP/(TP+FP)
#Error rate
ERR=(FP+FN)/(TP+FP+FN+TN)
#Accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
#F1 Acore
F1Score = TP/(TP+0.5*(FP+FN))
#AUROC score
AUROC = roc_auc_score(y_tests, y_probs)

averageScoreTime = 0
averageFitTime = 0

#Calculates average fit and score times.
for fitTime in fitTimes:
    averageFitTime+=fitTime

for scoreTime in scoreTimes:
    averageScoreTime+=scoreTime

averageScoreTime /=cv
averageFitTime /=cv

plt.clf()
#Creates ROC curve.
FPRs, TPRs, curThresholds = roc_curve(y_tests,y_probs)
plt.plot(FPRs, TPRs, label = "AUC = " + str(AUROC))
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title(config+" (Dataset "+str(dataset)+") Average Receiver Operating Characteristic Curve")
plt.legend()
plt.savefig("Graphs/"+config+"_average_dataset="+str(dataset)+".png", dpi=1000,bbox_inches='tight')

#Writes results to CSV file.
with open('scoring_results_dataset='+str(dataset)+'.csv', 'a', encoding='UTF8', newline='\n') as f:
    writer = csv.writer(f)
    writer.writerow([config,ACC,F1Score,Precision,Sensitivity,Fallout,Specificity,AUROC,NPV,FNR,FDR,averageScoreTime,averageFitTime,ERR])
