#!/usr/bin/env python
# coding: utf-8

# # For this Project, we will be analysing facebook dataset

# # 1.Data Set Up

# Import all the required vizualisation libraries for analysis and plotting

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_profiling
import warnings
warnings.filterwarnings('ignore')             # To suppress all the warnings in the notebook.
sns.set_style('whitegrid')


# # 2. Import Dataset from github and read as csv

# In[3]:


fb = pd.read_csv("https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Projects/facebook_data.csv")


# # 3.Understanding the Dataset

# In[9]:


fb.head()  


# In[10]:


fb.tail()


# In[11]:


fb.shape


# In[30]:


fb.columns


# There are 99003 rows and 15 columns in facebook dataset

# In[12]:


fb.describe()  # Descriptive statistics for the numerical variables


# In[13]:


fb.info()


# # 4.Pandas Profiling

# In[20]:


profile = pandas_profiling.ProfileReport(fb)
profile.to_file(output_file='fbdata.html')


# This is the overview of dataset info after Pandas Profiling

# # 5.Data Analysis

# In[33]:


fb.isnull().sum()                                          # finding null values


# There are 175 null values in gender and 2 null in tenure

# In[34]:


fb.gender.unique()


# In[36]:


print(any(fb['userid'].duplicated()))


# This returned false, which means there are only unique userid on our facebook dataset.

# # 6.Data Cleaning

# In[5]:


fb.drop(fb[fb['gender'].isnull()].index, inplace=True)                         # removing the null data on gender
fb.drop(fb[fb['tenure'].isnull()].index, inplace=True)                         # removing null date on tenure


# In[6]:


len(fb[(fb['likes_received']== 0) & (fb['likes']== 0) & (fb['friend_count']== 0) & (fb['friendships_initiated']== 0) & (fb['tenure']== 0)])


# There are 33 inactive users with no activity.

# In[7]:


fb.drop(fb[(fb['likes_received']== 0) & (fb['likes']== 0) & (fb['friend_count']== 0) & (fb['friendships_initiated']== 0) & (fb['tenure']== 0)].index, inplace=True)


# In[8]:


fb.shape


# # 7.Post Data Cleaning.

# In[62]:


fb.isnull().sum()


# Now there's no null values reflecting post data cleaning. We can proceed with data analysis & plotting

# In[9]:


print('minimum age user is :', fb['age'].min())
print('maximum age user is :', fb['age'].max())


# On this dataset, 113yrs is the oldest and 13yrs is the youngest user

# In[10]:


# Creating Age Group
labels = ['10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90=100','100-110','110-120']
fb['age_group']=pd.cut(fb.age,range(10,121,10),right=True, labels=labels)
fb.head()


# In[14]:


plt.figure(figsize=(10,3))
sns.countplot(x='age_group', data=fb, palette='viridis', hue='gender').set_title('User Age Group vs Gender')


# Maximum users belong to age group 20-30 after which there's decline in user trend. Male users are more than female. However users belonging to age group 50-70, female users are more than male slightly

# In[13]:


fb.age.plot.hist(figsize=(10,3), colormap='Dark2')
plt.title('Age Group Distribution')
plt.ylabel('count of users')
plt.xlabel('age')


# In[15]:


as_fig = sns.FacetGrid(fb,hue='gender',aspect=2)

as_fig.map(sns.kdeplot,'age',shade=True)

oldest = fb['age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()
plt.title('Age distribution using FacetGrid')


# In[79]:


fb['gender'].value_counts()


# Male users are more than female users

# In[24]:


plt.figure(figsize=(6,4))
sns.countplot(x='gender', data=fb, palette = 'viridis').set_title(' Countplot for Gender')


# 58552 users are male and 40241 users are female. Male users are more then female users. 

# In[84]:


fb.groupby('gender').friendships_initiated.describe()


# In[85]:


g = sns.FacetGrid(data=fb, col='gender',size=4, aspect=2)
g = g.map(plt.hist, 'friendships_initiated',bins=100)
plt.xlim(0, 400)


# Male users initiated friendship more then female users.

# In[87]:


fb.groupby('gender')['friend_count'].mean()


# In[27]:


fb.groupby('gender')['friend_count'].mean().plot.bar(color=['hotpink','blue'], grid = False, figsize=(7,4))
plt.title('Friends Count')


# Female has more friends than male. Seems like female are more popular

# In[118]:


fb.groupby('age_group').mean()['friend_count'].plot.bar()
plt.xlabel('age_group')
plt.ylabel('friend_count')
plt.title('Friends Count for Age Group')


# In[33]:


plt.figure(figsize=(14,6))
ax = sns.distplot(fb['tenure']/365, kde=False, bins=35,)
plt.xlabel('Number of Years',fontsize=15)
plt.xlim(0,7)
plt.title('Term in Years',fontsize=15)


# We can see people mostly uses facebook for a year and decline as years goes by. Need to promote the platform more

# In[120]:


fb.groupby('age_group').mean()['likes'].plot.bar(figsize=(14,6))
plt.xlabel('age_group')
plt.ylabel('Count of likes')
plt.title('Likes Given by Age Group')


# Age group 10-20 gave maximum likes among all users

# In[121]:


fb['Total_likes'] = fb['likes'] + fb['mobile_likes'] + fb['www_likes']


# In[123]:


fb['Total_likes_received'] = fb['likes_received'] + fb['mobile_likes_received'] + fb['www_likes_received']
fb.head()


# In[146]:


fb['mobile_likes'].describe()


# In[126]:


fb.groupby('age').count()['Total_likes'].plot()
plt.tight_layout()


# In[135]:


fb.groupby('gender').count()['Total_likes'].plot.kde()


# In[181]:


p = fb.pivot_table( index = 'age_group',columns = 'gender',values= 'Total_likes_received')


# In[191]:


p.plot(kind='bar', color=('salmon','seagreen'),fontsize=15)
plt.xlabel = ('age_group')
plt.ylabel = ('Total_likes_received')
plt.title ('Likes received by different age_group')


# Though female user count are lesser than male, they are give more likes than male. Which shows females are more active.

# In[187]:


p1 = fb.pivot_table( index = 'age_group',columns = 'gender',values= 'Total_likes')


# In[192]:


p1.plot(kind='bar', color=('salmon','seagreen'),fontsize=15)
plt.xlabel = ('age_group')
plt.ylabel = ('Total_likes')
plt.title ('Likes given by different age_group')


# In[194]:


fb.tenure.describe()


# In[201]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__)                          


# In[200]:


import cufflinks as cf
init_notebook_mode(connected=True)


# In[203]:


bd = sns.countplot(x='dob_day', data=fb)


# In[204]:


mn = sns.countplot(x='dob_month', data=fb)


# In[235]:


fb.groupby('dob_day').count()['dob_month'].plot()
plt.tight_layout()
plt.title('Date of birth Day')


# In[209]:


sns.factorplot('dob_month', hue='gender', kind='count', data=fb);
plt.title('Factor plot for male female and child')

