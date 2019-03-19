#!/usr/bin/env python
# coding: utf-8

# In[2]:


#guiding questions: how do I know if my dealer is offering a fair value for a traded in car? Did I put a fair value on my car?


# In[1]:


#setup, importing libraries and data
import pandas as pd
from pandas import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

dataset = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv"
df = pd.read_csv(dataset)


# In[2]:


#simple linear regression- understanding the relationship between the predictor (x) and the dependant variable (y)
#main equation: Yhat = a + bX

#loading modules for linear regression
from sklearn.linear_model import LinearRegression

#the linear regression object
lm = LinearRegression()

#highway-mpg predicting car price
X = df[["highway-mpg"]]
Y = df[["price"]]

#fitting the linear model using highway-mpg
lm.fit(X,Y)

#outputting the prediction
Yhat=lm.predict(X)
Yhat[0:5]

#value of intercept (a)
#take away # and execute: lm.intercept_

#value of Slope (b)
#Take away # and execute: lm.coef_

#so: Yhat = a +bX --> price = 38423.31 - 821.73 x highway-mpg


# In[3]:


#evaluating models using visualization

#importing visuaization package: seaborn
import seaborn as sns
from seaborn import *
get_ipython().run_line_magic('matplotlib', 'inline')

#for simple linear regression, it is best visualiazed by using regression plots

w = 10
h = 7

#graph for highway-mpg
plt.figure(figsize=(w,h))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# In[4]:


#visualizing the variance in the data with a residual plot; is highway mpg have a lot of data variance?

w = 12 
h = 10
plt.figure(figsize=(w,h))
sns.residplot(df["highway-mpg"], df["price"])
plt.show()

#residuals are not randomly spread out from the x axis; maybe a non linear model is appropriate for the data


# In[5]:


#multiple linear regression
#one continous response (dependant) variable and two or more predictor (independant) variables
#Yhat = a + b1X1 + b2X2 + b3X3 + b4X4 etc; a = intercept, x1 = predictor variable 1, b1 = coefficient of x1 (etc)

#using horsepower, curb weight, engine size, and highway-mpg
mlr = LinearRegression()

#condensing the predictor variables
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]

#fitting the linear model with the p-vars
mlr.fit(Z, df["price"])

#finding intercept a: -15806.62
#mlr.intercept_

#finding coeff of b1, b2, b3, b4 (53.49, 4.7, 81.5, 36.05)
#mlr.coef_

# price = -15806.62 + 53.49 * horsepower + 4.7 * curb-weight + 81.5 * engine-size + 36.05 * highway-mpg 


# In[6]:


#visualizing the multiple linear regression

Y_hat = mlr.predict(Z)
plt.figure(figsize=(w,h))
ax1 = sns.distplot(df["price"], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="fitted values", ax=ax1)

plt.title("actual vs fitted values for price")
plt.xlabel("price (in dollars)")
plt.ylabel("proportion of cars")

plt.show()
plt.close()

#fitted values are close to actual values; good


# In[7]:


#polynomial regression and pipelines

#quadratic, 2nd order: Yhat = a + b1X^2 + b2X^2
#Cubic, 3rd order: Yhat = a +b1X^2 + b2X^2 + b3X^3
#high order: Y = a + b1X^2 + b2X^2 + b3X^3....

#function to plot the data

def PlotPolly(model, indy_var, dep_var, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    
    plt.plot(indy_var, dep_var, ".", x_new, y_new, "-")
    plt.title("Polynomial fit with matplotlib for price ~ length")
    ax = plt.gca()
    ax.set_facecolor((.898, .898, .898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel("price of cars")
    
    plt.show()
    plt.close()
    
x = df["highway-mpg"]
y = df["price"]

#using 3rd order polynomial

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly (p, x, y, "highway-mpg")
np.polyfit(x, y, 3)

#this is clearly a etter model as it "hits" more data points


# In[8]:


#11 order polynomial mode
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p)
PlotPolly(p1, x, y, "Length")

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2)
pr

z_pr=pr.fit_transform(Z)

print("original data set contains (samples, features) ", Z.shape)

print("new data set contains (samples, features) ", z_pr.shape)


# In[9]:


#data pipelines simplify the steps in processing data

from sklearn.pipeline import *
from sklearn.preprocessing import *


#pipeline is created by a list of tuples including name of model and corresponding constructor
Input =[("scale", StandardScaler()), ("polynomial", PolynomialFeatures(include_bias=False)), ("mode",LinearRegression())]

#list is inputted as an argument to pipeline constructor
pipe=Pipeline(Input)

#normalizing the data
pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:4]


# In[15]:


#measures for in-sample evaluation
#R^2 / R-squared --> coeff of determination, a measure to indicate how close data is to the fitted regression line; value represents percentage of variation of the response variable (y) that is explained by a linear mode
#Mean Squared Error (MSE) --> measures average of the squares of errors, that is, the difference between actual value (y) and the estimated value (y but with a weird crown thing)

lm.fit(X,Y)

#finding the R^2
print("R square:", lm.score(X,Y))
#is .496565 ---> 49.65% of the variation of the price is explained by the simmple linear model horsepower_fit

#predicting output: Yhat
Yhat = lm.predict(X)
print("output of first four predicted value:", Yhat[0:4])

#finding MSE
from sklearn.metrics import mean_squared_error
mse= mean_squared_error(df["price"], Yhat)
print("the MSE of price and predicted value is:", mse)

#


# In[17]:


#MLR in sample evaluation

lm.fit(Z, df["price"])
print("r square", lm.score(Z, df["price"]))
#~80% of var in price is explained by the multi linear regression multi-fit
y_predict_multifit = lm.predict(Z)

print("MSE of price and predicted value using multifit is: ", mean_squared_error(df['price'], y_predict_multifit))

#MLR is best fit for the data as it has the highest r-squared and lowest MSE values
#checking r^2 and mse serves as a double check in the case that one variable is not useful or even acts as noise


# In[19]:


#polynomial in sample evaluation

from sklearn.metrics import *

r_squared = r2_score(y, p(x))
print ("the r square value is:", r_squared)

mean_squared_error(df["price"], p(x))


# In[20]:


#prediction and decision making

import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

#creating a new input
new_input = np.arange(1, 100, 1).reshape(-1,1)

#fit the model
lm.fit(X, Y)

#produce a prediction
yhat=lm.predict(new_input)
yhat[0:5]

#plot data
plt.plot(new_input, yhat)
plt.show()

#graph with preiction


# In[ ]:




