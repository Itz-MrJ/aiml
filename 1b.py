
# Step1:importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Step2:load dataset
df=pd.read_csv("housing_prices_SLR.csv",delimiter=',')
df.head()


x=df[['AREA']].values#feature Matrix
y=df.PRICE.values#Target Matrix
x[:5] #slicing
y[:5]


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

print(x_train.shape)
print(x_test.shape)
print(x_train.shape)
print(x_test.shape)


from sklearn.linear_model import LinearRegression


lr_model= LinearRegression()


lr_model.fit(x_train,y_train)

print(lr_model.intercept_) 
print(lr_model.coef_)

from sklearn.metrics import r2_score
print(lr_model.predict(x_train))
print(lr_model.score(x_test,y_test))
print(lr_model.score(x_train,y_train))   
plt.scatter(x_train,y_train,c='red')
plt.scatter(x_test,y_test,c='blue')
plt.plot(x_train,lr_model.predict(x_train),c='green')
