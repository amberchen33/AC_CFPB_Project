
# coding: utf-8

# # CFPB Customer Complait Issue

# ## Feature Engineering

# ### 1. Import Packages

# In[34]:

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Basic operations
import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

#
from datetime import *
import math


# ### 2. Load data

# In[2]:

data = pd.read_csv('./cleaned_1.csv')
data.shape


# In[8]:

data.head(3)


# ### 3. Feature Engineering

# In[4]:

def mean_rate_by_cat(data, cat):
    cat = str(cat)
    
    # Calculate mean response rate of each cat
    mean_rate_by_cat = data.groupby('%s'%cat).mean()
    
    # reset index
    mean_rate_by_cat['%s'%cat] = mean_rate_by_cat.index
    mean_rate_by_cat.columns = ['mean_response_rate_by_%s'%cat, '%s'%cat]
    mean_rate_by_cat = mean_rate_by_cat.reset_index(drop=True)
    
    # Calculate mean response rate of each cat
    mean_rate_by_cat['mean_response_rate_over_avg_by_%s'%cat] = mean_rate_by_cat['mean_response_rate_by_%s'%cat] / 0.972434 - 1

    return mean_rate_by_cat


# In[5]:

def freq_by_cat(data, cat):
    cat = str(cat)
    # Calculate the frequency of each cat
    freq = data.groupby('%s'%cat).count()/860606
    # reset index
    freq['%s'%cat] = freq.index
    freq.columns = ['freq_by_%s'%cat, '%s'%cat]
    freq = freq.reset_index(drop=True)
    
    return freq


# ** Time **

# In[9]:

# Time 
data.Date_received = pd.to_datetime(data.Date_received)
data.Date_sent_to_company = pd.to_datetime(data.Date_sent_to_company)

# Using np array in order to calcalate quicker
rece = np.array(data.Date_received)
sent = np.array(data.Date_sent_to_company)
days = (sent - rece).astype('timedelta64[D]') / np.timedelta64(1, 'D')

# New feature: days_spend
data['days_spend'] = days

# New features: year and month
data['year'] = data.Date_received.dt.year
data['month'] = data.Date_received.dt.month


# In[7]:

time = data[['year','month','Is_Timely_response']]

# Calculate mean response rate by month and year
mean_rate_by_time = time.groupby(['year','month']).mean()
mean_rate_by_time.columns = ['mean_rate_by_time']

# reset index
mean_rate_by_time = mean_rate_by_time.reset_index(level=['year', 'month'])
mean_rate_by_time['mean_rate_over_avg_by_time'] = mean_rate_by_time.mean_rate_by_time - 0.972434

# Two new features: mean_rate_over_avg_by_time, mean_rate_by_time
data = pd.merge(data, mean_rate_by_time, on=['year','month'])


# ** Company **

# In[10]:

# Extract Company 
company = data[['Company', 'Is_Timely_response']]
company = company.groupby('Company').count()
company = company.sort('Is_Timely_response', ascending=False)
company.columns = ['complaint_count']

# Scale Company based on number of complaints
scale1 = company[company.complaint_count<=10].index
scale10 = company[(company.complaint_count<=100) & (company.complaint_count>10)].index
scale100 = company[(company.complaint_count<=1000) & (company.complaint_count>100)].index
scale1000 = company[(company.complaint_count<=10000) & (company.complaint_count>1000)].index
scale10000 = company[company.complaint_count>10000].index

# Encode number of complaints
s1 = pd.DataFrame(index=scale1,data=np.ones(len(scale1)))
s10 = pd.DataFrame(index=scale10,data=2*np.ones(len(scale10)))
s100 = pd.DataFrame(index=scale100,data=3*np.ones(len(scale100)))
s1000 = pd.DataFrame(index=scale1000,data=4*np.ones(len(scale1000)))
s10000 = pd.DataFrame(index=scale10000,data=5*np.ones(len(scale10000)))

# Create complaint_scale data frame 
scale = pd.concat([s1,s10,s100,s1000,s10000])
scale.columns = ['complaint_scale']
scale['Company'] = scale.index

# New feature: complaint_scale
data = pd.merge(data, scale, on='Company')


# In[11]:

company = data[['Company', 'Is_Timely_response']]

mean_rate_by_company = mean_rate_by_cat(company, 'Company')
freq_by_company = freq_by_cat(company, 'Company')

# Two new features: mean_response_rate_by_company, mean_rate_over_avg_by_company
data = pd.merge(data, mean_rate_by_company, on='Company')
# One new feature: freq_by_company
data = pd.merge(data, freq_by_company, on='Company')


# In[ ]:




# ** Is_Consumer_disputed **

# In[12]:

data.Is_Consumer_disputed[data.Is_Consumer_disputed == 'Yes'] = 1
data.Is_Consumer_disputed[data.Is_Consumer_disputed == 'No'] = 0
data.Is_Consumer_disputed[data.Is_Consumer_disputed == 'NotKnow'] = -1


# In[13]:

Is_Consumer_disputed = data[['Is_Consumer_disputed', 'Is_Timely_response']]

mean_rate_by_disput = mean_rate_by_cat(Is_Consumer_disputed, 'Is_Consumer_disputed')
freq_by_disput = freq_by_cat(Is_Consumer_disputed, 'Is_Consumer_disputed')

# Two new features: mean_response_rate_by_issue, mean_rate_over_avg_by_issue
data = pd.merge(data, mean_rate_by_disput, on='Is_Consumer_disputed')
# One new feature: freq_by_issue
data = pd.merge(data, freq_by_disput, on='Is_Consumer_disputed')


# ** Issue **

# In[14]:

issue = data[['Issue', 'Is_Timely_response']]


# In[15]:

mean_rate_by_issue = mean_rate_by_cat(issue, 'Issue')
freq_by_issue = freq_by_cat(issue, 'Issue')

# Two new features: mean_response_rate_by_issue, mean_rate_over_avg_by_issue
data = pd.merge(data, mean_rate_by_issue, on='Issue')
# One new feature: freq_by_issue
data = pd.merge(data, freq_by_issue, on='Issue')


# In[16]:

'''
issue.str.contains(
    'Loan|loan|Debt|debt|credit|Credit|Lender|lender|broker'
).sum()

issue.str.contains(
    'Problem|problem|report|Report|info|Information|information|advertising|Advertising|Other'
).sum()

issue.str.contains(
    'Account|account|Lost|stolen|missing|statements|Billing|cash|withdrawals|card|Property|property|Deposite|Communication|Overdraft|overdraft|vehicle|Vehicle'
).sum()

'''


# In[ ]:




# ** Product **

# In[17]:

product = data[['Product', 'Is_Timely_response']]


# In[18]:

mean_rate_by_product = mean_rate_by_cat(product, 'Product')
freq_by_product = freq_by_cat(product, 'Product')

# Two new features: mean_response_rate_by_product, mean_rate_over_avg_by_product
data = pd.merge(data, mean_rate_by_product, on='Product')
# One new feature: freq_by_product
data = pd.merge(data, freq_by_product, on='Product')


# In[ ]:




# ** State **

# In[19]:

State = data[['State', 'Is_Timely_response']]


# In[20]:

mean_rate_by_State = mean_rate_by_cat(State, 'State')
freq_by_State = freq_by_cat(State, 'State')

# Two new features: mean_response_rate_by_product, mean_rate_over_avg_by_product
data = pd.merge(data, mean_rate_by_State, on='State')
# One new feature: freq_by_product
data = pd.merge(data, freq_by_State, on='State')


# In[ ]:




# ** ZIP_Code **

# In[21]:

ZIP_code = data[['ZIP_code', 'Is_Timely_response']]


# In[22]:

mean_rate_by_ZIP_code = mean_rate_by_cat(ZIP_code, 'ZIP_code')
freq_by_ZIP_code = freq_by_cat(ZIP_code, 'ZIP_code')

# Two new features: mean_response_rate_by_product, mean_rate_over_avg_by_product
data = pd.merge(data, mean_rate_by_ZIP_code, on='ZIP_code')
# One new feature: freq_by_product
data = pd.merge(data, freq_by_ZIP_code, on='ZIP_code')


# In[ ]:




# ** One-hot encoding **

# In[23]:

data1=data.copy()


# In[25]:

data1.head(2)


# In[26]:

dummies = pd.get_dummies(data1[[1,2,4,6]])


# In[27]:

data1 = data1.drop(
    ['Date_received', 'ZIP_code', 'Date_sent_to_company', 
     'Complaint_ID', 'Company', 'Product', 'Issue', 'State', 
     'Submitted_via', 'Company_response_to_consumer'], 1
)


# In[28]:

data1 = pd.concat([data1,dummies],1,join='inner')
data1.head(2)


# ** Outliers in days_spend **

# In[36]:

avg = data1.days_spend.mean()


# In[37]:

avg + 3 * math.sqrt(data1.days_spend.var())


# In[50]:

data1.Is_Timely_response[data1.days_spend > 53].mean()


# In[ ]:




# In[29]:

data1.to_csv('./data_for_tree.csv', index=False)
data1.shape


# In[ ]:




# ### 4. Feature Selection / Reduction

# In[59]:

import xgboost as xgb


# In[95]:

data1.shape


# In[96]:

data1.Is_Consumer_disputed = data1.Is_Consumer_disputed.astype('int')


# In[97]:

label = data1.Is_Timely_response
del data1['Is_Timely_response']


# In[98]:

dtrain = xgb.DMatrix(data1, label=label)


# In[100]:

param = {'max_depth': 20, 'eta': 100, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = ['auc', 'ams@0']


# In[69]:

import time


# In[101]:

start_time = time.time()

num_round = 300
bst = xgb.train(param, dtrain, num_round)

print("--- %s seconds ---" % (time.time() - start_time))


# In[102]:

xgb.plot_importance(bst)
plt.show()


# In[103]:

bst.get_fscore()


# In[ ]:



