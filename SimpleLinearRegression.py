# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:12:49 2020

@author: User
"""
#SimpleLinear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_excel(r'C:\Users\User\Desktop\Project\SimpleRegression.xlsx')
x=data.iloc[:,1].values
y=data.iloc[:,2].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=380)
from sklearn.linear_model import LinearRegression
x_train=x_train.reshape(-1,1)
regression=LinearRegression()
regression.fit(x_train,y_train)
regression.coef_
regression.intercept_
x_test=x_test.reshape(-1,1)
y_pred=regression.predict(x_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,y_pred)
mean_squared_error(y_test,y_pred)
import math
math.sqrt(mean_squared_error(y_test,y_pred))
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regression.predict(x_train),color='blue')
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue') 

#Multiple Regression
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
df1=pd.read_excel(r'C:\Users\User\Desktop\Project\DataModel.xlsx')
x1=df1.iloc[:,1].values
x2=df1.iloc[:,2].values
y=df1.iloc[:,3].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=1/3,random_state=380)
from sklearn.linear_model import LinearRegression
x_train=x_train.reshape(-1,1)
regression=LinearRegression()
regression.fit(x_train,y_train)
regression.coef_
regression.intercept_
x_test=x_test.reshape(-1,1)
y_pred=regression.predict(x_test)
plt.title('Simple Linear Regression on Crop Production from 2010 to 2011')
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regression.predict(x_train),color='blue')
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue') 
plt.ylabel('Yield')
plt.xlabel('Production')
reg1=ols(formula='y~x1',data=df1)
fit1=reg1.fit()
print(fit1.summary())
from statsmodels.formula.api import ols
model=ols(formula='y~x1+x2',data=df1).fit()
print(model.summary())

