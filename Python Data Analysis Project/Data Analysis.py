#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
from pandas import *
import numpy as np
from scipy import stats

data_set = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv"
df = pd.read_csv(data_set)


# In[112]:


#visualization- analyzing individual feature patterns

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#correlation between bore, stroke, compression ratio, and horsepower
bcsrgroup = df[["bore", "stroke", "compression-ratio", "horsepower"]].corr()
#type bcsrgroup to observe the correlation


# In[113]:


#scatterplot for continuous numerical variables (either float64 or int64; any value within some range)

#continuous numerical variable engine size predicting price?

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

df[["engine-size", "price"]].corr()

#engine size goes up, price goes up; positive correlation between these two variables. Regression line is almost perfectly diagnoal- meaning engine size is a good predictor for price


# In[114]:


#continuous numerical variable highway mpg as price predictor?

sns.regplot(x="highway-mpg", y="price", data=df)
df[["highway-mpg", "price"]].corr()

#as highway mpg gets better, price goes down; inverse relationship.
#scattered data points and near horizontal regression line would suggest a weak relationship


# In[115]:


#categorical variables- dtype object or int64. Categorical varibles are best visualized with boxplots. They describe a characteristic of a data unit

#categorical variable of body style as a predictor for price?
sns.boxplot(x="body-style", y="price", data=df)

#as one can see- there is much overlap and thus body style is not a good predictor


# In[116]:


#categorical variable engine location as price predictor?

sns.boxplot(x="engine-location", y="price", data=df)

#as one can see, rear engine cars are more costly in comparison to fwd


# In[117]:


#categorical variable drive wheels as price predictor?

sns.boxplot(x="drive-wheels", y="price", data=df)

#as one can see, rwd cars are more costly


# In[118]:


#descriptive statistical analysis

df.describe(include=["object"])


#value count as a variable- only works with panda series and not frame, thus it should be converted
drive_wheel_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheel_counts.rename(columns={"drive-wheels" : "value_counts"}, inplace=True)
drive_wheel_counts.index.name = "drive-wheel"

#engine location as a variable
engine_loc_counts = df["engine-location"].value_counts().to_frame()
engine_loc_counts.rename(columns={"engine-location" : "value_counts"}, inplace=True)
engine_loc_counts.index.name = "engine-location"


#drive_wheel_counts.head(10) --> drive wheels is much less skewed in comparison to engine location
#engine_loc_counts.head(10) --> we can see the data is skewed as there is only 3 rear engines versus the 198 front engine cars


# In[137]:


#grouping

#which type of drive wheel/body style combo is most valuable? grouping drive wheel/body style and averaging

df_group = df[["drive-wheels", "body-style", "price"]]
df_group1 = df_group.groupby(["drive-wheels", "body-style"], as_index=False).mean()
#df_group to check results



# In[138]:


#turning the df_group into a pivot table
group_pivot = df_group1.pivot(index="drive-wheels", columns="body-style")

#there is missing data; NaN thus it should be replaced with 0
group_pivot = group_pivot.fillna(0)

#group_pivot done again except


# In[139]:


#heat map of the relationship between body style vs price with the grouped results

#using the grouped results
plt.pcolor(group_pivot, cmap="RdBu")
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(group_pivot, cmap="RdBu")

#label names
rlabel = group_pivot.columns.levels[1]
clabel = group_pivot.index

#moving ticks and labels to center
ax.set_xticks(np.arange(group_pivot.shape[1]) + .5, minor=False)
ax.set_yticks(np.arange(group_pivot.shape[0]) + .5, minor=False)

#inserting labes
ax.set_xticklabels(rlabel, minor=False)
ax.set_yticklabels(clabel, minor=False)

#label too long --> rotate
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()


# In[122]:


#significance of correlation: pearson coefficient and p value

#wheel base v price, pearson coef and p value
pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print("pearson correlation coefficient is", pearson_coef, " with the p value of", p_value )

#P value <.0001 meaning the correlation is statistically significant, but with a mid-weak linear relationship


# In[123]:


#pearson coef and p value: horsepower vs price
pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
print("pearson coeff is ", pearson_coef, " P value is", p_value)

#strong linear relationship and p value <.0001. 


# In[124]:


#pearson coef and p value: length vs price
pearson_coef, p_value = stats.pearsonr(df["length"], df["price"])
print("pearson coeff is ", pearson_coef, " P-value is", p_value)
#moderately strong linear correlation and very statiscally significant


# In[125]:


#pearson coef and p value: width vs price
pearson_coef, p_value = stats.pearsonr(df["width"], df["price"])
print("pearson coeff is ", pearson_coef, " P-value is", p_value)
#strong linear relationship and very statistically significant


# In[126]:


#pearson coef and p value: curb-weight vs price
pearson_coef, p_value = stats.pearsonr(df["curb-weight"], df["price"])
print("pearson coeff is ", pearson_coef, " P-value is", p_value)
#strong linear relationship and very statiscally significant


# In[127]:


#pearson coef and p value: engine-size vs price
pearson_coef, p_value = stats.pearsonr(df["engine-size"], df["price"])
print("pearson coeff is ", pearson_coef, " P-value is", p_value)
#very strong linear relationship and very statisicaally significant 


# In[128]:


#pearson coef and p value: bore vs price
pearson_coef, p_value = stats.pearsonr(df["bore"], df["price"])
print("pearson coeff is ", pearson_coef, " P-value is", p_value)
#moderate/medium linear relationship but very statistically relevent 


# In[129]:


#pearson coef and p value: city-mpg vs price
pearson_coef, p_value = stats.pearsonr(df["city-mpg"], df["price"])
print("pearson coeff is ", pearson_coef, " P-value is", p_value)
#an inverse, moderately strong linear relationship but is very statistically relevent


# In[130]:


#pearson coef and p value: highway-mpg vs price
pearson_coef, p_value = stats.pearsonr(df["highway-mpg"], df["price"])
print("pearson coeff is ", pearson_coef, " P-value is", p_value)
#somewhat strong inverse linear relationship, but is very statistically relevent 


# In[141]:


#analysis of variance; testing differences between the means of wo or more groups
#returns F-test score (bigger score means larger difference between means)
#returns p-value
#ANOVA of drive-wheels and price

dwp_group = df_group[["drive-wheels", "price"]].groupby(["drive-wheels"])
#dwp_group.head(2) to check on results


# In[142]:


#ANOVA:all drive-wheel sets

f_val, p_val = stats.f_oneway(dwp_group.get_group("fwd")["price"], dwp_group.get_group("rwd")["price"], dwp_group.get_group("4wd")["price"])

print("ANOVA results: F score= ", f_val, "P-score= ", p_val)
#large F score means strong correlation, and P value of almost 0 means statistical significance 


# In[143]:


#ANOVA with fwd and rwd

f_val, p_val = stats.f_oneway(dwp_group.get_group("fwd")["price"], dwp_group.get_group("rwd")["price"])
print("ANOVA results: F score= ", f_val, "P-score= ", p_val)
#F score denotes a strong correlation and the P value of under 0 means strong statistical significance 


# In[145]:


#ANOVA with 4WD and RWD

f_val, p_val = stats.f_oneway(dwp_group.get_group("4wd")["price"], dwp_group.get_group("rwd")["price"])
print("ANOVA results: F score= ", f_val, "P-score= ", p_val)

#F score denotes a correlation, with P value denoting statistical significance


# In[146]:


#ANOVA with 4wd and fwd

f_val, p_val = stats.f_oneway(dwp_group.get_group("4wd")["price"], dwp_group.get_group("fwd")["price"])
print("ANOVA results: F score= ", f_val, "P-score= ", p_val)
#F score denotes very weak correlation and P score denotes very weak statistical significance 


# In[ ]:


#conclusion: I know know what the data looks like and what to take into account for when I predict car price
#continous numerical variables: length, width, curb-weight, engine-size, horsepower, city-mpg, highway-mpg, wheel-base, bore
#categorical variables: drive-wheels

