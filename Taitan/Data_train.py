import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

Data_train=pd.read_csv("train.csv")
# print(Data_train.head())
Data_train ["Age"] = Data_train ['Age'] . fillna(Data_train['Age'].median())  
# //Age列中的缺失值用Age均值进行填充

Data_train.loc[Data_train["Sex"] == "male","Sex"] = 0  
Data_train.loc[Data_train["Sex"] == "female","Sex"] = 1  

Data_train["Embarked"]=Data_train["Embarked"].fillna("S")

Data_train.loc[Data_train["Embarked"]=="S","Embarked"]=0
Data_train.loc[Data_train["Embarked"]=="C","Embarked"]=1
Data_train.loc[Data_train["Embarked"]=="Q","Embarked"]=2

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

alg=LinearRegression()

kf=KFold(Data_train.shape[0],n_folds=3,random_state=1)

predictions=[]

for train,test in kf:
    train_predictors=(Data_train[predictors].iloc[train,:])
    train_target=Data_train["Survived"].iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictions=alg.predict(Data_train[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions=np.concatenate(predictions,axis=0)
predictions[predictions>0.5]=1
predictions[predictions<0.5]=0
acc=sum(predictions[predictions==Data_train["Survived"]])/len(predictions)
print(acc)

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

alg=LogisticRegression(random_state=1)
scores=cross_validation.cross_val_score(alg,Data_train[predictors],Data_train["Survived"],cv=3)

print (scores.mean())