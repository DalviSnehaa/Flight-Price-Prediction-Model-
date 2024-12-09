#!/usr/bin/env python
# coding: utf-8

# In[6]:


####################  Data Analysis & Calculation  #####################3
import numpy as np   
import pandas as pd  
import datetime  

####################  Visuvalization & plotting  #####################3
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#################### Machine Learning #####################3  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings  
warnings.filterwarnings("ignore")


# ## Import Data

# In[7]:


df = pd.read_excel(r"C:\datasets\Flight_Price_Train.xlsx")
df.head()


# ### Q.1 Perform Feature Engineering

# #### a. Perform basic exploration like checking for top 5 records, shape, statistical info, duplicates, Null values etc. 

# In[8]:


df.head(5)


# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df[df.duplicated()]


# ##### Here we found that 220 rows are duplicate rows.

# In[13]:


df = df.drop_duplicates()
df.shape


# #####  Duplicate rows are dropped in our dataset

# In[14]:


df.isnull().sum()


# ##### Missing value found in Total_stops  

# In[15]:


df = df.dropna()
df.shape


# #### b) Extract Date, Month, Year from Date of Journey column

# In[16]:


df.head()


# In[17]:


df['Journey_Date'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
df['Journey_Day'] = df['Journey_Date'].dt.day
df['Journey_Month'] = df['Journey_Date'].dt.month
df['Journey_Year'] = df['Journey_Date'].dt.year

df[['Date_of_Journey', 'Journey_Date', 'Journey_Day', 'Journey_Month', 'Journey_Year']].head()


# In[18]:


df = df.drop('Date_of_Journey', axis=1)


# ### Q.2 Perform Exploratory Data Analysis (EDA) tasks

# #### a. Which airline is most preferred airline

# In[19]:


df['Airline'].value_counts()


# In[20]:


sns.countplot(x = 'Airline', data=df)
plt.xticks(rotation=90);


# ##### The "Jet Airways" is the most preferred AirLine 

# ### c. Find the majority of the flights take off from which source

# In[21]:


df['Source'].value_counts()


# In[22]:


sns.countplot(x = 'Source', data=df)
plt.xticks(rotation=90);


# ### The major flights are taken off from "Delhi"

# #### d. Find maximum flights land in which destination

# In[23]:


df['Destination'].value_counts()


# In[24]:


sns.countplot(x = 'Destination', data=df)
plt.xticks(rotation=90);


# ### The maximum flights are landed in "Cochin"

# ### Q.3 Compare independent features with Target feature to check the impact on price

# #### a. Which airline has the highest price 

# In[25]:


average_price_by_airline = df.groupby('Airline')['Price'].mean()
average_price_by_airline.sort_values(ascending=False)


# In[26]:


airline_highest_price = average_price_by_airline.idxmax()
highest_average_price = average_price_by_airline.max()

print("The airline with the highest average price is:", airline_highest_price)
print("Average price for this airline:", highest_average_price)


# #### b. Check if the business class flights are high price or low and find only those flights which price is higher than 50k

# In[27]:


business_class_flights = df[df['Additional_Info'].str.contains('business', case=False)]


# In[28]:


business_class_flights


# In[29]:


print("Average price for business class flights:", business_class_flights['Price'].mean())


# In[30]:


print('Average price for all the flights:', df['Price'].mean())


# ### The Business class flights price is higher than other flights.

# In[31]:


business_class_flights[business_class_flights['Price'] > 50000]


# ### There are three busniness class flights whose price higher than 50k.

# In[32]:


df[df['Price'] > 50000]


# In[33]:


len(df[df['Price'] > 50000])


# ### There are eight flights with prices exceeding 50,000.

# ### Q.4 Perform encoding for the required features according to the data

# In[34]:


df.head()


# In[35]:


df['Dep_Time'] = pd.to_datetime(df['Dep_Time'])
df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])


# In[36]:


df['Dep_Hour'] = df['Dep_Time'].dt.hour
df['Dep_Minute'] = df['Dep_Time'].dt.minute
df['Arrival_Hour'] = df['Arrival_Time'].dt.hour
df['Arrival_Minute'] = df['Arrival_Time'].dt.minute


# In[37]:


df['Duration_minutes'] = df['Duration'].str.split().apply(lambda x: int(x[0][:-1])*60 + int(x[1][:-1]) if len(x) == 2 else int(x[0][:-1])*60)


# In[38]:


df.drop(['Dep_Time', 'Arrival_Time', 'Duration'], axis=1, inplace=True)


# In[39]:


df.head()


# In[40]:


# Convert total_stops in numeric
df['Total_Stops'] = df['Total_Stops'].str.split().apply(lambda x: int(x[0]) if len(x) == 2 else 0)


# In[41]:


df.head()


# In[42]:


# Label encoding for 'Route'
df['Route'] = df['Route'].astype('category').cat.codes


# In[43]:


for i in ['Airline', 'Source', 'Destination']:
    df[i] = df[i].astype('category').cat.codes


# In[44]:


# Label encoding for 'Additional_Info' (assuming it's a text column)
df['Additional_Info'] = df['Additional_Info'].astype('category').cat.codes


# In[45]:


df.drop('Journey_Date', axis=1, inplace=True)


# In[46]:


df


# ### Q.5 Build multiple model by using different algorithm such as Linear Regression, Decision Tree, and Random Forest etc. and check the performance of your model.

# In[47]:


sns.boxplot(df['Price'])


# In[48]:


Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

# Define the upper and lower bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers from the 'Price' column
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]


# In[49]:


df


# In[50]:


sns.boxplot(df['Price'])


# In[51]:


np.sqrt(df['Price']).hist()


# In[52]:


df['Price'] = np.sqrt(df['Price'])


# #### we applied transformation method to make price column normal.

# In[53]:


# Data partition

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[54]:


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['squared_error','friedman_mse'],
                'splitter': ['best','random']
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                
            }
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=101)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, error_score='raise')
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            #'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score'])

find_best_model_using_gridsearchcv(X_train,y_train)


# ##### Random forest gives us higher performance with 92% accuaracy on train dataset

# #### Q.6 Compare all of the models and justify your choice about the optimum model by using different evaluation technique and tune the models as per the requirement. 

# #### Upon evaluating all models, the random forest emerges as the most effective. Let's assess it using various evaluation methods.

# In[55]:


model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[56]:


# Evaluate models
def evaluate_model(actual, predictions):
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    return mse, mae, r2


# In[57]:


random_forest_mse, random_forest_mae, random_forest_r2 = evaluate_model(y_test, y_pred)


# In[58]:


print("Random Forest:")
print("Mean Squared Error:", random_forest_mse)
print("Root Mean Squared Error:", np.sqrt(random_forest_mse))
print("Mean Absolute Error:", random_forest_mae)
print("R-squared:", random_forest_r2)


# - Based on the assessment findings, it seems that the Random Forest model outperforms others in predictive precision, given its minimal Mean Squared Error (MSE), Mean Absolute Error (MAE), and maximal R-squared value in test data. This implies that, in comparison to alternative models, the Random Forest model is superior at encapsulating the intricate correlations between the attributes and flight costs.

# #### Q.7 Write a conclusion from the business point of view. Finally perform the same preprocessing technique for test data best practice using pipeline.

# ### Conclusion
# - From the standpoint of commerce, it's advantageous to have a precise prediction model for airfare for diverse participants in the travel sector, encompassing airlines, travel intermediaries, and consumers. 
# - Airlines can employ these models to fine-tune pricing tactics, efficiently oversee stock, and boost profits. 
# - Travel intermediaries can give customers more precise cost estimates and propose competitive rates. 
# - Consumers can make better-informed choices regarding their travel arrangements and locate the most attractive offers.

# In[59]:


df_test = pd.read_excel(r"C:\datasets\Flight_Price_Test.xlsx")


# In[60]:


df_test


# In[61]:


df_test['Journey_Date'] = pd.to_datetime(df_test['Date_of_Journey'], format='%d/%m/%Y')
df_test['Journey_Day'] = df_test['Journey_Date'].dt.day
df_test['Journey_Month'] = df_test['Journey_Date'].dt.month
df_test['Journey_Year'] = df_test['Journey_Date'].dt.year


# In[62]:


df_test.drop(['Date_of_Journey', 'Journey_Date'], axis=1, inplace=True)


# In[63]:


df_test['Dep_Time'] = pd.to_datetime(df_test['Dep_Time'])
df_test['Arrival_Time'] = pd.to_datetime(df_test['Arrival_Time'])


# In[64]:


df_test['Dep_Hour'] = df_test['Dep_Time'].dt.hour
df_test['Dep_Minute'] = df_test['Dep_Time'].dt.minute
df_test['Arrival_Hour'] = df_test['Arrival_Time'].dt.hour
df_test['Arrival_Minute'] = df_test['Arrival_Time'].dt.minute


# In[65]:


df_test['Duration_minutes'] = df_test['Duration'].str.split().apply(lambda x: int(x[0][:-1])*60 + int(x[1][:-1]) if len(x) == 2 else int(x[0][:-1])*60)


# In[66]:


df_test.drop(['Dep_Time', 'Arrival_Time', 'Duration'], axis=1, inplace=True)


# In[67]:


df_test['Total_Stops'] = df_test['Total_Stops'].str.split().apply(lambda x: int(x[0]) if len(x) == 2 else 0)


# In[68]:


df_test['Route'] = df_test['Route'].astype('category').cat.codes


# In[69]:


for i in ['Airline', 'Source', 'Destination']:
    df_test[i] = df_test[i].astype('category').cat.codes


# In[70]:


df_test['Additional_Info'] = df_test['Additional_Info'].astype('category').cat.codes


# In[71]:


df_test


# In[72]:


df_test['Predicted Price'] = np.round(np.square(model.predict(df_test)))


# In[73]:


df_test


# ###  we predicted the price of flights for our test dataset.

# #### Q.8 Calculate the
# - a) recency (R),
# - b) frequency (F)
# - c) monetary value (M)  for each customer based on the given dataset?
# 

# In[75]:


df2 = pd.read_csv(r"C:\datasets\RFM data.csv")


# In[76]:


df2.head()


# In[77]:


df2.shape


# In[78]:


df2.info()


# In[79]:


# Convert InvoiceDate from object to datetime format
df2['InvoiceDate'] = pd.to_datetime(df2['InvoiceDate'])  # yyyy-mm-dd
df2.head()


# In[80]:


len(list(df2['CustomerID']
         .unique()))


# In[81]:


print('{:,} rows; {:,} columns'
      .format(df2.shape[0], df2.shape[1]))

print('{:,} transactions don\'t have a customer id'
      .format(df2[df2.CustomerID.isnull()].shape[0]))

print('Transactions timeframe from {} to {}'.format(df2['InvoiceDate'].min(),
                                    df2['InvoiceDate'].max()))


# In[82]:


df2 = df2.dropna()


# In[83]:


from datetime import timedelta

snapshot_date = df2['InvoiceDate'].max() + timedelta(days=1)
print(snapshot_date)


# In[84]:


data_process = df2.groupby(['CustomerID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'})


# In[85]:


data_process.head()


# In[86]:


data_process.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalPrice': 'MonetaryValue'}, inplace=True)

data_process


# #### Q.9 
# ##### a. Calculate RFM scores. Each customer will get a note between 1 and 5 for each parameter for Recency(R), Frequency(F) and   Monetary value(M) Ex: Scale for Recency:

# In[87]:


#--Calculate R and F groups--
# Create labels for Recency  
r_labels = range(5, 0, -1)

# Create labels for Frequency

f_labels = range(1, 6)      

# Create labels for MonetaryValue
m_labels = range(1, 6)  


# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(data_process['Recency'], q=5, labels=r_labels)


# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(data_process['Frequency'], q=5, labels=f_labels)

# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(data_process['MonetaryValue'], q=5, labels=m_labels)


# Create new columns R and F 
data_process = data_process.assign(Recency_Rank = r_groups.values, Frequency_Rank = f_groups.values, Moentary_Rank = m_groups.values)


data_process.head()


# In[88]:


data_process['RFM_Segment_Concat'] = data_process.Recency_Rank.astype(str) + data_process.Frequency_Rank.astype(str) + data_process.Moentary_Rank.astype(str)
rfm = data_process
rfm.head()


# In[89]:


# Count num of unique segments
rfm_count_unique = rfm.groupby('RFM_Segment_Concat')['RFM_Segment_Concat'].nunique()
print("Combination of Ranks  :", rfm_count_unique.sum())


# In[90]:


rfm['RFM_Score'] = rfm[['Recency_Rank','Frequency_Rank','Moentary_Rank']].sum(axis=1)
rfm.head()


# #### b. Segment the customers based on their RFM scores using the dataset?

# In[91]:


def assign_segment(row):
    
    if row['Recency_Rank'] >= 4 and row['Frequency_Rank'] >= 4 and row['Moentary_Rank'] >= 4:
        return 'Champions'
    elif 3 <= row['Recency_Rank'] <= 5 and 3 <= row['Frequency_Rank'] <= 5 and 3 <= row['Moentary_Rank'] <= 5:
        return 'Loyal customers'
    elif row['Recency_Rank'] >= 4 and 2 <= row['Frequency_Rank'] <= 3 and 2 <= row['Moentary_Rank'] <= 3:
        return 'Potential loyalist'
    elif row['Recency_Rank'] >= 4 and row['Frequency_Rank'] <= 2 and row['Moentary_Rank'] <= 2:
        return 'Recent customers'
    elif row['Recency_Rank'] >= 4 and row['Frequency_Rank'] <= 2 and row['Moentary_Rank'] <= 2:
        return 'Promising'
    elif 3 <= row['Recency_Rank'] <= 5 and 3 <= row['Frequency_Rank'] <= 5 and 3 <= row['Moentary_Rank'] <= 5:
        return 'Needs attention'
    elif row['Recency_Rank'] <= 2 and row['Frequency_Rank'] <= 2 and row['Moentary_Rank'] <= 2:
        return 'About to sleep'
    elif 2 <= row['Recency_Rank'] <= 5 and 1 <= row['Frequency_Rank'] <= 3 and 1 <= row['Moentary_Rank'] <= 3:
        return 'At risk'
    elif 1 <= row['Recency_Rank'] <= 3 and row['Frequency_Rank'] >= 4 and row['Moentary_Rank'] >= 4:
        return "Can't lose them"
    elif row['Recency_Rank'] <= 2 and row['Frequency_Rank'] <= 2 and row['Moentary_Rank'] <= 2:
        return 'Hibernating'
    else:
        return 'Others'


# In[92]:


rfm['Customer_Segment'] = rfm.apply(assign_segment, axis=1)
rfm.head()


# In[93]:


Report = rfm[["Customer_Segment"]].reset_index()
Report.head()


# #### Q.10 
# ##### a. Visualize the RFM segments.

# In[94]:


import matplotlib.pyplot as plt

# Count occurrences of each segment
segment_counts = rfm['Customer_Segment'].value_counts()

# Plot pie chart
plt.figure(figsize=(8, 6))
plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Customer Segments Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# #### b. Conclude your findings of RFM analysis and suggest some strategies on it.

# * The RFM analysis has allowed us to categorize our customers into different segments based on their Recency, Frequency, and Monetary attributes. Here's a summary of the findings and recommended strategies from the RFM evaluation:
# 
# * Champions:
#   These customers are of the highest value, having made recent, regular, and high-cost purchases.
# ###### Strategies: Aim to keep these customers satisfied by providing unique rewards, customized deals, and top-tier treatment.   Inspire them to be ambassadors for our brand by delivering standout customer service and engaging them with loyalty   schemes.
# * Loyal Customers:
#   Though not as current as the Champions, Loyal Customers still show consistent high-value purchasing patterns.
# ###### Strategies: Sustain the fidelity of these customers by appreciating their repeated business, offering loyalty-based   discounts, and supplying exclusive benefits. Motivate them to boost their purchasing frequency through personalized cross-selling or upselling drives.
# * Potential Advocates:
#   These consumers have exhibited the possibility of becoming devoted, although their regularity and monetary contributions  might not be significant.
# ###### Suggested Course: Connect with these consumers through specific advertising strategies to heighten their buying frequency and average order cost. Provide benefits or rewards for repeated buying and introduce them to customer fidelity schemes to stimulate enduring devotion.
# * New and Ascending Clients:
#   These customers are fairly fresh or current, possessing potential for an expansion in both regularity and financial worth.
# ###### Suggested Course: Prioritize delivering a flawless introductory experience for these new clients to stimulate recurring business. Propose individualized suggestions based on their initial acquisitions and provide incentives to become more involved with your brand.
# * Requires Attention:
#   Despite being active, these clients might exhibit indications of waning engagement or depreciating value.
# ###### Suggested Course: Reconnect with these customers through targeted revival campaigns, exclusive deals, or bespoke communication. Detect the causes of their diminished activity and promptly address any worries or problems to avert customer loss.
# * On the Verge:
#   These clients are in danger of departing due to a decrease in frequency or financial contribution.
# ###### Tactics: Employ proactive retention tactics to reclaim these clients, such as custom win-back proposals, rewards for loyalty, or enhanced customer care. Initiate communication to comprehend their requirements and resolve any grievances to hinder their departure.
# * Must Keep:
#   Though not the most recurrent, these clients have demonstrated considerable financial worth and must be preserved.
# ###### Tactics: Concentrate on preserving the fidelity of these high-worth clients by offering exceptional assistance, tailored attention, and exclusive perks. Propose loyalty schemes or elite status to fortify their fidelity and stimulate ongoing expenditure.
# * Nearing Dormancy and Inactive:
#   These clients are dormant and might necessitate specific initiatives to reawaken their engagement.
# ###### Tactics: Implement specific reactivation drives, custom propositions, and incentives to reincorporate these clients. Provide unique deals or markdowns to respark their intrigue and incite them to conduct a transaction.
# 
# ##### Through the application of specific tactics designed for every RFM division, companies can efficiently enhance the value of a customer over their lifetime, boost client retention, and escalate the growth of revenue. Continuous observation and fine-tuning of these methods, in response to changing consumer habits, will be crucial for sustained prosperity.

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




