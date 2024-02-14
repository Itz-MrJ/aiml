
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("housing_prices.csv")

df.head()

df=df.iloc[:,[0,1,2,4]]

df.head()

x=df.iloc[:,:3].values
y=df.iloc[:,3].values

print(x[:5])
print(y[:5])

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

model= LinearRegression()

model.fit(x_train,y_train)

print(model.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(model.coef_)#y=c+mx


print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
