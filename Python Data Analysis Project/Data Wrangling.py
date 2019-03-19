#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
from pandas import * 
import numpy as np
from numpy import *

car_data = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = read_csv(car_data, names = headers)

#Identifying missing values, and handling them- we first have to do df.head(any int) to be able to see the head of the data 
#by doing df.head(ant int) one can see that there are numerous question marks throughout the dataset- however the missing data values have now been replaced by nan

df.replace ("?", nan, inplace = True)

#one can do df.head(any int) to check if the missing data values were indeed replaced by NaN


# In[49]:


#using .isnull() or .notnull() will tell us if data is missing via true or false, with true or false representing missing/not missing depending on which evaluator is used

data_eval = df.isnull()
data_eval.head(5)

#false means that there is a data value in place there. True represents no data value- dependant if using .isnull() or .notnull()


# In[50]:


for column in data_eval.columns.values.tolist():
    print(column)
    print(data_eval[column].value_counts())
    print("")
    
#using a for loop, and will print


# In[60]:


#dealing with the missing data
#drop the data- drop the row or column
#or replace data with the mean, freq, or replace based on other function
import numpy as np
from numpy import *

#the following will be replaced by mean: normalized-losses, stroke, bore, horsepower, and peak-rpm

avg_normal_loss = df["normalized-losses"].astype("float").mean()
print ("average normalized losses is", avg_normal_loss)
df["normalized-losses"].replace(np.nan, avg_normal_loss, inplace=True)

avg_stroke = df["stroke"].astype("float").mean()
print ("average stroke is", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

avg_bore = df["bore"].astype("float").mean()
print("average bore is", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)

avg_horsepower = df["horsepower"].astype("float").mean()
print("average horsepower is", avg_horsepower)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

avg_peak_rpm = df["peak-rpm"].astype("float").mean()
print("average peak rpm is", avg_peak_rpm)
df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace=True)


#replacing by freq; since most common number of doors is 4, it makes to do this replacement
door_number = df["num-of-doors"].value_counts().idxmax()
print ("most frequent door number is", door_number)
df["num-of-doors"].replace(np.nan, door_number, inplace=True)

#dropping rows without price- I plan on predicting price later on, thus I will drop rows with price and in future will attempt to predict car price
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)


# In[52]:


#using df.dtypes, we can see some columns have the wrong data type
#bore, stroke, normalized-losses, price, and peak-rpm need fixing

df["bore"] = df["bore"].astype("float")
df["stroke"] = df["stroke"].astype("float")
df["normalized-losses"] = df["normalized-losses"].astype("int")
df["price"] = df["price"].astype("float")
df["peak-rpm"] = df["peak-rpm"].astype("float")

#use df.dtypes to check if replacement is succesfull


# In[53]:


#standardizing the data
#converting mpg to liters per 100km

df["highway-mpg"] = 235/df["highway-mpg"]
df["city-mpg"] = 235/df["city-mpg"]

df.rename(columns={"highway-mpg" : "highway-L/100KM"}, inplace=True)
df.rename(columns={"city-mpg" : "city-L/100KM"}, inplace=True)

#if you do dfhead(), you can see the previous city-mpg and highway-mpg changed to L/100KM for each respective context


# In[54]:


#Data normalization-binning horsepower, city-mpg, and highway-mpg
#first the data must be visualized
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
from matplotlib import *


df["horsepower"]=df["horsepower"].astype("int", copy=True)
#histagram of horsepower to observe the distrubution



plt.pyplot.hist(df["horsepower"])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("frequency")
plt.pyplot.title("horsepower")


# In[55]:


#binning horsepower

hbin = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
hbin_groups = ["low", "medium", "high"]

df["horsepower-binned"] = pd.cut(df["horsepower"], hbin, labels=hbin_groups, include_lowest=True)

#do below to check on binning- to see if it was succesfull 
#df["horsepower-binned"].head(20)


# In[56]:


#horsepower binned graph

a = (0,1,2)

plt.pyplot.bar(hbin_groups, df["horsepower-binned"].value_counts())
plt.pyplot.title("horsepower binned")
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("frequency")


# In[57]:


#Now we are going to seperate fuel-type into gas and diesel; dummy variable- for regression modeling later, it requires integers

dvar1 = get_dummies(df["fuel-type"])
dvar1.head()

dvar1.rename(columns={"fuel-type-diesel" : "diesel", "fuel-type-diesel" : "gas"}, inplace=True)

df = concat([df, dvar1], axis=1)

df.drop(["fuel-type"], axis= 1, inplace=True)

df.head(5)

#a 1 means yes it is gas or diesel


#end

