
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import pickle 
from os import path

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from sklearn import preprocessing
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,average_precision_score,confusion_matrix
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import ClassificationReport
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")

from PIL import ImageTk,Image
import tkinter as tk
from tkinter import *
from PIL import Image
import sys
import subprocess





def Browse():

  data = pd.read_csv("C:\\Users\\prapti\\Desktop\\NIDS ML\\NIDS_ML\\UNSW_NB15_training-set.csv")
  data.head()

  data.shape

  data.info()

  data.drop('service',axis='columns',inplace=True)

  data.isnull().sum()

  data['attack_cat'].value_counts()

  data['state'].value_counts()

  data

  features = pd.read_csv("C:\\Users\\prapti\\Desktop\\NIDS ML\\NIDS_ML\\NUSW-NB15_features.csv",encoding='cp1252')

  features.head()

  features['Type '] = features['Type '].str.lower()

  # selecting column names of all data types
  nominal_names = features['Name'][features['Type ']=='nominal']
  integer_names = features['Name'][features['Type ']=='integer']
  binary_names = features['Name'][features['Type ']=='binary']
  float_names = features['Name'][features['Type ']=='float']

  cols = data.columns
  nominal_names = cols.intersection(nominal_names)
  integer_names = cols.intersection(integer_names)
  binary_names = cols.intersection(binary_names)
  float_names = cols.intersection(float_names)

  for c in integer_names:
    pd.to_numeric(data[c])

  for c in binary_names:
    pd.to_numeric(data[c])

  for c in float_names:
    pd.to_numeric(data[c])

  data.info()

  data

  plt.figure(figsize=(8,8))
  plt.pie(data.label.value_counts(),labels=['normal','abnormal'],autopct='%0.2f%%')
  plt.title("Pie chart distribution of normal and abnormal labels",fontsize=16)
  plt.legend()
  plt.show()

  plt.figure(figsize=(8,8))
  plt.pie(data.attack_cat.value_counts(),labels=data.attack_cat.unique(),autopct='%0.2f%%')
  plt.title('Pie chart distribution of multi-class labels')
  plt.legend(loc='best')
  plt.show()

  num_col = data.select_dtypes(include='number').columns

  # selecting categorical data attributes
  cat_col = data.columns.difference(num_col)
  cat_col = cat_col[1:]
  cat_col

  data_cat = data[cat_col].copy()
  data_cat.head()

  from sklearn import preprocessing
  le = preprocessing.LabelEncoder()
  data_cat['proto'] = le.fit_transform(data_cat['proto'])
  data_cat['state'] = le.fit_transform(data_cat['state'])

  data_cat

  data.drop(columns=cat_col,inplace=True)

  data = pd.concat([data, data_cat],axis=1)

  # selecting numeric attributes columns from data
  num_col = list(data.select_dtypes(include='number').columns)
  num_col.remove('id')
  num_col.remove('label')
  print(num_col)

  # using minmax scaler for normalizing data
  minmax_scale = MinMaxScaler(feature_range=(0, 1))
  def normalization(df,col):
    for i in col:
      arr = df[i]
      arr = np.array(arr)
      df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
    return df

  data.head()

  data = normalization(data.copy(),num_col)

  data.head()

  """Binary Labels"""

  bin_label = pd.DataFrame(data.label.map(lambda x:'normal' if x==0 else 'abnormal'))

  # creating a dataframe with binary labels (normal,abnormal)
  bin_data = data.copy()
  bin_data['label'] = bin_label

  # label encoding (0,1) binary labels
  le1 = preprocessing.LabelEncoder()
  enc_label = bin_label.apply(le1.fit_transform)
  bin_data['label'] = enc_label

  le1.classes_

  np.save("le1_classes.npy",le1.classes_,allow_pickle=True)

  """Multi-class Labels"""

  multi_data = data.copy()
  multi_label = pd.DataFrame(multi_data.attack_cat)
  multi_data = pd.get_dummies(multi_data,columns=['attack_cat'])
  le2 = preprocessing.LabelEncoder()
  enc_label = multi_label.apply(le2.fit_transform)
  multi_data['label'] = enc_label
  le2.classes_

  np.save("le2_classes.npy",le2.classes_,allow_pickle=True)

  num_col.append('label')

  # Correlation Matrix for Binary Labels
  plt.figure(figsize=(20,8))
  corr_bin = bin_data[num_col].corr()
  sns.heatmap(corr_bin,vmax=1.0,annot=False)
  plt.title('Correlation Matrix for Binary Labels',fontsize=16)
  plt.show()

  num_col = list(multi_data.select_dtypes(include='number').columns)
  # Correlation Matrix for Multi-class Labels
  plt.figure(figsize=(20,8))
  corr_multi = multi_data[num_col].corr()
  sns.heatmap(corr_multi,vmax=1.0,annot=False)
  plt.title('Correlation Matrix for Multi Labels',fontsize=16)
  plt.show()

  """Feature Selection

  Binary Labels
  """

  corr_ybin = abs(corr_bin['label'])
  highest_corr_bin = corr_ybin[corr_ybin >0.3]
  highest_corr_bin.sort_values(ascending=True)

  bin_cols = highest_corr_bin.index
  bin_cols

  bin_data = bin_data[bin_cols].copy()
  bin_data

  bin_data.to_csv('bin_data.csv')

  """Multi-class Labels"""

  # finding the attributes which have more than 0.3 correlation with encoded attack label attribute 
  corr_ymulti = abs(corr_multi['label'])
  highest_corr_multi = corr_ymulti[corr_ymulti >0.2]
  highest_corr_multi.sort_values(ascending=True)

  # selecting attributes found by using pearson correlation coefficient
  multi_cols = highest_corr_multi.index
  multi_cols

  multi_data = multi_data[multi_cols].copy()

  multi_data.to_csv('multi_data.csv')

  """BINARY CLASSIFICATION"""

  X = bin_data.drop(columns=['label'],axis=1)
  Y = bin_data['label']

  X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=50)

  #GaussianNB
  def GNB(X_train,y_train,X_test,y_test):
      gnb_clf = GaussianNB()
      pred = gnb_clf.fit(X_train, y_train).predict(X_test)
      pred= gnb_clf.predict(X_test)
      print ("GaussianNB:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.tight_layout()
      plt.title('Confusion matrix of Navie Bayes')
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=gnb_clf.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(gnb_clf, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  GNB(X_train,y_train,X_test,y_test)

  #KNeighbours
  def KNN1(X_train,y_train,X_test,y_test):
      knn = KNeighborsClassifier(n_neighbors=2)
      knn.fit(X_train, y_train)
      pred = knn.predict(X_test)
      print ("KNN:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of KNN', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=knn.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(knn, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  KNN1(X_train,y_train,X_test,y_test)

  #LogisticRegression
  from sklearn.linear_model import LogisticRegression
  def logisticreg(X_train,y_train,X_test,y_test):
      lr = LogisticRegression()
      lr.fit(X_train,y_train)
      pred = lr.predict(X_test)
      print ("LogisticRegression:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of LogisticRegression', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=lr.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(lr, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  logisticreg(X_train,y_train,X_test,y_test)

  #Randon Forest 
  def random_forest(X_train,y_train,X_test,y_test):
      rf = RandomForestClassifier(max_depth=2, min_samples_split=2)
      rf.fit(X_train,y_train)
      pred = rf.predict(X_test)
      print ("Random Forest:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of Random Forest', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=rf.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(rf, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  random_forest(X_train,y_train,X_test,y_test)

  #Stacking Classifier
  xg = xgb.XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0,min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
  rf = RandomForestClassifier(bootstrap=True,max_depth= 70,max_features= 'auto',min_samples_leaf= 4,min_samples_split= 10,n_estimators= 400)
  knn=KNeighborsClassifier()

  def stacking(X_train,y_train,X_test,y_test):
      classifiers=[rf,knn]
      sc = StackingClassifier(classifiers,meta_classifier=xg)  
      sc.fit(X_train,y_train)
      pred = sc.predict(X_test)
      print ("Stacking Classifier:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of Stacking Classifier', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=sc.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(sc, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  stacking(X_train,y_train,X_test,y_test)

  """MULTI-CLASS CLASSIFICATION"""

  X = multi_data.drop(columns=['label'],axis=1)
  Y = multi_data['label']
  X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30, random_state=100)

  #GaussianNB
  def GNB(X_train,y_train,X_test,y_test):
      gnb_clf = GaussianNB()
      pred = gnb_clf.fit(X_train, y_train).predict(X_test)
      pred= gnb_clf.predict(X_test)
      
      print ("GaussianNB:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.tight_layout()
      plt.title('Confusion matrix of GaussianNB', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=gnb_clf.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(gnb_clf, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  GNB(X_train,y_train,X_test,y_test)

  #KNeighbours
  def KNN1(X_train,y_train,X_test,y_test):
      knn = KNeighborsClassifier()
      knn.fit(X_train, y_train)
      pred = knn.predict(X_test)
      print ("KNN:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of KNN', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=knn.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(knn, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  KNN1(X_train,y_train,X_test,y_test)

  #LogisticRegression
  from sklearn.linear_model import LogisticRegression
  def logisticreg(X_train,y_train,X_test,y_test):
      lr = LogisticRegression()
      lr.fit(X_train,y_train)
      pred = lr.predict(X_test)
      print ("LogisticRegression:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of LogisticRegression', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=lr.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(lr, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  logisticreg(X_train,y_train,X_test,y_test)

  #Random Forest
  def random_forest(X_train,y_train,X_test,y_test):
      rf = RandomForestClassifier(max_depth=2, min_samples_split=2,)
      rf.fit(X_train,y_train)
      pred = rf.predict(X_test)
      print ("Random Forest:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of Random Forest', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=rf.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(rf, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
  random_forest(X_train,y_train,X_test,y_test)

  #Stacking Classifier
  xg = xgb.XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0,min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
  rf = RandomForestClassifier(bootstrap=True,max_depth= 70,max_features= 'auto',min_samples_leaf= 4,min_samples_split= 10,n_estimators= 400)
  knn=KNeighborsClassifier()

  def stacking(X_train,y_train,X_test,y_test):
      classifiers=[rf,knn]
      sc = StackingClassifier(classifiers,meta_classifier=xg)  
      sc.fit(X_train,y_train)
      pred = sc.predict(X_test)
      print(classification_report(y_test, pred))
      print(confusion_matrix(y_test, pred))
      print ("Stacking Classifier:Accuracy : ", accuracy_score(y_test,pred)*100)

      #confusion Matrix
      matrix =confusion_matrix(y_test, pred)
      sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
      plt.title('Confusion matrix of Stacking Classifier', y=1.1)
      plt.ylabel('Actual label')
      plt.xlabel('Predicted label')
      plt.show()

      #Classification Report
      prediction=sc.predict(X_test)
      print(classification_report(y_test, prediction))
      visualizer = ClassificationReport(sc, support=True)
      visualizer.fit(X_train, y_train)  
      visualizer.score(X_test, y_test)  
      g = visualizer.poof()
      print("Close the window..!!!!")
  stacking(X_train,y_train,X_test,y_test)





#------------tkinter-----------
HEIGHT = 520
WIDTH = 680

root = Tk()  # Main window 
f = Frame(root)

root.title("Network Intrusion Detection System")
root.geometry("680x520")
root.resizable(0,0)

canvas = Canvas(width=680, height=520, bg='white')
canvas.pack()
filename=("C:\\Users\\prapti\\Desktop\\NIDS ML\\NIDS_ML\\nids_image.jpg")
load = Image.open(filename)
load = load.resize((680, 250), Image.LANCZOS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
load = Image.open(filename)
img.place(x=1, y=1)


root.configure(background='#FCFCE5')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

frame = tk.Frame(root, bg='white', bd=10)
frame.place(relx=0.5, rely=0.50, relwidth=1, relheight=0.5, anchor='n')


submit = tk.Button(frame,font=40, text='Run',height=2,width="13",command=lambda: Browse(),fg="white", bg="blue")
submit.grid(row=5, column=10,padx=20,pady=20)
submit.place(relx=0.40, rely=0.25)


root.mainloop()

