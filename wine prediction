import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
df=pd.read_excel("//content//1788410-1767134-1729261-1613779-Red_wine__(1).xlsx")
df.head()
df.info()
df.describe()
df.isnull()
df.isull().sum()
for col in df.columns:
  if df[col].isnull().sum()>0:
    df[col]=df[col].fillna(df[col].mean())
df[col].isnull().sum().sum()
df.hist(bins=20,figsize=(10,10))
plt.show()
plt.bar(df["quality"],df["alcohol"])
plt.xlabel("quality")
plt.ylabel("alcohol")
plt.show()
plt.figure(figsize=(12,12))
sns.heatmap(df.corr()>0.7,annot=True,cbar=False)
plt.show()
df=df.drop("total sulfur dioxide",axis=1)
df["best quality"]=[1 if x>5 else 0 for x in df.quality]
df.replace({"white":1,"red":0}, inplace=True)
features=df.drop(["quality","best quality"],axis=1)
x_train.shape
X_test.shape
Y_train.shape
Y_test.shape
norm=MinMaxScaler()
x_train=norm.fit_transform(x_train)
x_test=norm.transform(x_test)
models=[LogisticRegression(),SVC(kernel='rbf')]
for i in range(2):
  models[i].fit(x_train,y_train)
  print(f'{models[i]}:')
  print('training accuracy:',metrics.roc_auc_score(y_train,models[i].predict(x_train)))
  print("validation accuracy:",metrics.roc_auc_score(y_test,models[i].predict(x_test)))
  print()
print(metrics.classification_report(y_test,models[1].predict(x_test)))
