
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

df=pd.read_csv("breast_cancer.csv")

df=df.iloc[:,:-1]

df.shape

x=df.iloc[:,2:].values
y=df.diagnosis.values

print(x[:2])
print(y[:5])

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=500)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

(y_train == 'M').sum()

(y_train=='B').sum()

278/len(y_train)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

baseline_pred=["B"] *len(y_train) # baseline hoga beningn for sabke liye

accuracy_score(y_train,baseline_pred) #  parameter me actual aur predicted data lenge

confusion_matrix(y_train,baseline_pred)# parameter me actual aur predicted data lege

print(classification_report(y_train,baseline_pred))

from sklearn.naive_bayes import GaussianNB

nb_model=GaussianNB()

nb_model.fit(x_train,y_train)

print(x_train)

nb_model.score(x_train,y_train)

nb_model.score(x_test,y_test)

#confusion_matrix training data ke liye
confusion_matrix(y_train,nb_model.predict(x_train))

#confusion matrix test data ke liye
confusion_matrix(y_test,nb_model.predict(x_test))

print(classification_report(y_train,nb_model.predict(x_train)))

print(classification_report(y_test,nb_model.predict(x_test)))
