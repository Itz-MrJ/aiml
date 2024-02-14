
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("housing_prices_SLR.csv",delimiter=',')
df.head()


x=df[['AREA']].values
y=df.PRICE.values
x[:5]
y[:5]




x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

print(x_train.shape)
print(x_test.shape)
print(x_train.shape)
print(x_test.shape)




lr_model= LinearRegression() # / (fit_intercept = False)


lr_model.fit(x_train,y_train)

print(lr_model.intercept_) 
print(lr_model.coef_)

#method 1 (print all)
lr_model.predict(x_train)
r2_score(y_train,lr_model.predict(x_train)) 
r2_score(y_test,lr_model.predict(x_test))

#method 2
print(lr_model.predict(x_train))
print(lr_model.score(x_test,y_test))
# print(lr_model.score(x_train,y_train))   
plt.scatter(x_train,y_train,c='red')
plt.scatter(x_test,y_test,c='blue')
plt.plot(x_train,lr_model.predict(x_train),c='green')
