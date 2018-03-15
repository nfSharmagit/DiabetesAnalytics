
# coding: utf-8

# Project Objective:
# 
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
#The objective is to predict whether a patient has diabetes based on diagnostic measurements.
# 
# Dataset Reference:
# 
# https://www.kaggle.com/uciml/pima-indians-diabetes-database/data
# 
# Dataset Attributes: 
# 
# Pregnancies(NPG):   Number of times pregnant
# 
# Glucose(GPL):       Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# BloodPressure(DIA): Diastolic blood pressure (mm Hg) 80 - 90 is normal
# 
# SkinThickness(TSF): Triceps skin fold thickness (mm)
# 
# Insulin(INS):       2-Hour serum insulin (mu U/ml)
# 
# BMI:                Body mass index (weight in kg/(height in m)^2)
# 
# DiabetesPedigreeFunction(DPF): Diabetes pedigree function
# 
# Age: Age (years)

# In[59]:


import pandas as pd
import itertools
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, metrics
from sklearn.cross_validation import cross_val_score , cross_val_predict,train_test_split
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[60]:


#Function to measure algorithm accuracy 
def accuracy_score(y_actual,y_pred):
    return metrics.accuracy_score(y_actual,y_pred)*100


# In[61]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[62]:


#Import data
X=pd.read_csv("pima_X.csv")
X.head()


# In[63]:


X.info()


# In[64]:


#Drop the columns not required for analysis
X.drop(['Unnamed: 0'],axis=1)


# In[65]:


#import label data
y=pd.read_csv("pima_Y.csv")
y.head()


# In[66]:


#Splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4)


# # Random Forest

# In[67]:


#Inititae Random Forest Classifier 
clf= RandomForestClassifier()


# ### Cross-Validation 

# In[68]:


#Cross Validation into 10 folds 
clf_scores=cross_val_score(clf,X,np.ravel(y),cv=10)
print(clf_scores)


# In[69]:


#mean score and the 95% confidence interval of the score estimate
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)" % (clf_scores.mean(), clf_scores.std() * 2))


# In[70]:


# Prediction - Cross Validation
predicted= cross_val_predict(clf,X,np.ravel(y),cv=10)
print(predicted)


# In[71]:


#Algorithm accuarcy 
a_score_clf = accuracy_score(y,predicted)
print("Metric function accuracy for cross_val_predict data:%f" %a_score_clf)


# ### Train-Test Split

# In[72]:


#Fit the model on training data
clf.fit(X_train,np.ravel(y_train))


# In[73]:


#Predict on test data
y_clf_pred = clf.predict(X_test)
y_clf_pred


# In[74]:


#compare actual response value (y_test) with the predicted response value (y_clf_pred)
a_score_clf = accuracy_score(y_test,y_clf_pred)
print("Metric function accuracy for training data :%f" %a_score_clf)


# In[75]:


cnf_matrix = metrics.confusion_matrix(y_test,y_clf_pred)


# In[76]:


class_names = ['Diabetic','Non-Diabetic']


# In[77]:


#Plot Confusion Matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Random Forest-Confusion Matrix')
plt.show()


# In[78]:


print(metrics.classification_report(y_test,y_clf_pred,target_names = class_names))


# # K-NN

# In[79]:


#Initiate KnnClassifier
knn= KNeighborsClassifier(n_neighbors=7)


# In[80]:


#Cross Validation Score
knn_scores=cross_val_score(knn,X,np.ravel(y),cv=10)
print(knn_scores)


# In[81]:


#mean score and the 95% confidence interval of the score estimate
print("KNN Classifier Accuracy: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))


# In[82]:


#Cross Validation predict
Predict_knn = cross_val_predict(knn,X,np.ravel(y),cv=10)
print(Predict_knn)


# In[83]:


a_score_knn = accuracy_score(y,Predict_knn)
print("SVM - Metric function accuracy for cross_val_predict data:%f" %a_score_knn)


# # Train-Test Split

# In[84]:


#Fit the training data
knn.fit(X_train,np.ravel(y_train))


# In[85]:


#predict labels for test data
y_knn_predict=knn.predict(X_test)
print(y_knn_predict)


# In[86]:


#Accuracy on testing data
knn_test = accuracy_score(y_test,y_knn_predict)
print(knn_test)


# In[87]:


knn_matrix_knn = metrics.confusion_matrix(y_test,y_knn_predict)


# In[88]:


#Plot Confusion Matrix
plt.figure()
plot_confusion_matrix(knn_matrix_knn, classes=class_names,
                      title='Knn - Confusion matrix')
plt.show()


# In[89]:


print(metrics.classification_report(y_test,y_knn_predict,target_names = class_names))


# In[90]:


#Initiate Decision tree classifier
Tree = tree.DecisionTreeClassifier()


# In[91]:


#Cross Validation Score
Tree_scores=cross_val_score(Tree,X,np.ravel(y),cv=10)
print(Tree_scores)


# In[92]:


#mean score and the 95% confidence interval of the score estimate
print("KNN Classifier Accuracy: %0.2f (+/- %0.2f)" % (Tree_scores.mean(), Tree_scores.std() * 2))


# In[93]:


#Prediction-Cross Validation 
Predict_Tree = cross_val_predict(Tree,X,np.ravel(y),cv=10)
print(Predict_Tree)


# In[94]:


#Accuracy of Algorithm 
a_score_Tree = accuracy_score(y,Predict_Tree)
print("SVM - Metric function accuracy for cross_val_predict data:%f" %a_score_Tree)


# # Train-Test Split

# In[95]:


Tree.fit(X_train,np.ravel(y_train))


# In[96]:


#predict labels for test data
y_Tree_predict=Tree.predict(X_test)
print(y_Tree_predict)


# In[97]:


#Accuracy on testing data
Tree_test = accuracy_score(y_test,y_Tree_predict)
print(Tree_test)


# In[98]:


Tree_matrix = metrics.confusion_matrix(y_test,y_Tree_predict)


# In[99]:


#Plot Confusion Matrix 
plt.figure()
plot_confusion_matrix(Tree_matrix, classes=class_names,
                      title='Decision Tree - Confusion matrix')
plt.show()


# Results:
# Random Forest Accuracy : 76.04%
# KNN Accuracy           : 72.39%
# Decision Tree Accuracy : 72.91%
#     
# Created a predictive model to determine whether a individual has diabetes or no based on given parameters.
# Based on given data the random forest algorithm can be used for pediction given a information about particular patient.
#     
