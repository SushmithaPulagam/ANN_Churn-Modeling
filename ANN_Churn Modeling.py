#!/usr/bin/env python
# coding: utf-8

# In[2]:



import keras
keras.__version__


# In[3]:


import pandas as pd

dataset = pd.read_csv("G:\\Article6 - Neural Networks\\ANN - Churn Modeling\\Churn_Modelling.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


X = dataset.iloc[:,3:13]
Y = dataset.iloc[:,13]


# In[7]:


X.head()


# In[8]:


Y.head()


# In[11]:


geography = pd.get_dummies(X['Geography'], drop_first = True)
gender = pd.get_dummies(X['Gender'], drop_first = True)


# In[13]:


X = pd.concat([X,geography,gender], axis=1)


# In[18]:


X = X.drop(['Geography','Gender'], axis = 1)


# In[19]:


X.head()


# In[21]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state  = 0)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[24]:


X_train


# In[25]:


X_test


# In[27]:


import keras
from keras.models import Sequential # mandatory module for all the NN's
from keras.layers import Dense  # to create hidden layers
from keras.layers import Dropout # to avoid overfitting


# In[29]:


# Initializing the ANN
classifier  = Sequential()


# In[33]:


classifier.add(Dense(units = 6,kernel_initializer = 'he_uniform',activation = 'relu',input_dim = 11))
classifier.add(Dense(units = 6,kernel_initializer = 'he_uniform',activation = 'relu'))
classifier.add(Dense(units = 1,kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))


# In[34]:


classifier.summary()


# In[36]:


classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


# In[37]:


model_history = classifier.fit(X_train,Y_train,validation_split = 0.33, batch_size = 10,nb_epoch = 100)


# In[41]:



import numpy as np
import matplotlib.pyplot as plt
print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[42]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[44]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


# In[46]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,Y_test)


# In[47]:


print(score)


# In[ ]:




