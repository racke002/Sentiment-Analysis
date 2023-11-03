#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import re, string, random
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#importing the data and data exploration
raw = pd.read_csv(r"C:\Users\Ryan\Desktop\Python Projects\Twitter_Data.csv")
raw.head()
raw.info()


# In[4]:


#cleaning the data
len(raw)
lines = raw.dropna()
len(lines)
#we had to drop 11 rows 


# In[5]:


#data has three sentiments: positive, negative and neutral
sentiments = lines.groupby('sentiment').describe()
sentiments


# In[6]:


#separating tweet from their sentiment labels
texts = lines.iloc[:,0].values
labels = lines.iloc[:,1].values


# In[7]:


#creating a for loop to remove unneccesary characters
cleaned_text = []

for sentence in range(0, len(texts)):
    remove = re.sub(r'\W', ' ', str(texts[sentence]))
    remove = re.sub(r'\s+[a-zA-Z]\s+', ' ', remove)
    remove = re.sub(r'\^[a-zA-Z]\s+', ' ', remove) 
    remove = re.sub(r'\s+', ' ', remove, flags=re.I)
    remove = re.sub(r'^b\s+', '', remove)
    remove = remove.lower()
    cleaned_text.append(remove)


# In[8]:


#Text embedding and vectorizing 
#TfidVectorizer sets tweet to a matrix of TF-IDF features. TF-IDF minimizes the weights of extremely common words
#max features set to 2500 words to avoid uncommon words
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
#here we turn the cleaned tweets into vector so that our models can understand them
features = vectorizer.fit_transform(cleaned_text).toarray()


# In[9]:


#splitting the data into testing and training data (75/25)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)


# In[10]:


len(x_train)
# Setting aside a validation set
x_val = x_train[0:12222]
partial_x_train = x_train[12222:122226]
y_val = y_train[0:12222]
partial_y_train = y_train[12222:122226]

len(y_val)/len(partial_y_train) #validation data roughly 10% of training


# In[16]:


#Random forest machine learning method
RF_model = RandomForestClassifier(n_estimators=20, random_state=0)
RF_model.fit(partial_x_train, partial_y_train)


# In[48]:


#predictions based on randomw forest model 
predictions = RF_model.predict(x_val)


# In[44]:


#creating confusion matrix from random forest predictions
get_ipython().run_line_magic('matplotlib', 'inline')
plot_confusion_matrix(RF_model, x_val, y_val, display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues', xticks_rotation='vertical')


# In[50]:


#determining precision, recall and accuracy of random forest method
print(classification_report(y_val,predictions))
print(accuracy_score(y_val, predictions))


# In[38]:


#creating a logistic regression model for machine learning
model = LogisticRegression(max_iter=1000, random_state=0)
model.fit(partial_x_train, partial_y_train)


# In[55]:


#predictions based on Logistic regression model
predictions2 = model.predict(x_val)


# In[57]:


#logistic regression method precision, recall and accuracy measures
print(classification_report(y_val,predictions2))
print(accuracy_score(y_val, predictions2))


# In[58]:


#Confusion matrix from logistic regression model
get_ipython().run_line_magic('matplotlib', 'inline')
plot_confusion_matrix(model, x_val, y_val, display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues', xticks_rotation='vertical')


# In[64]:


#final results with logistic regression model and testing data
predictions_final = model.predict(x_test)
print(classification_report(y_test,predictions_final))
print(accuracy_score(y_test, predictions_final))


# In[65]:


#Final logistic regression confuson matrix 
get_ipython().run_line_magic('matplotlib', 'inline')
plot_confusion_matrix(model, x_test, y_test, display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues', xticks_rotation='vertical')

