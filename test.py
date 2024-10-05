import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


clos=["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df=pd.read_csv("magic04.data",names=clos)
df["class"]=(df["class"]=="g").astype(int)

# for label in clos[:-1]:
#     plt.hist(df[df["class"]==1][label],color='blue',label='gamma',alpha=0.7, density=True)
#     plt.hist(df[df["class"]==0][label],color='red',label='gamma',alpha=0.7, density=True)
#     plt.title(label)
#     plt.xlabel(label)
#     plt.ylabel("probability")
#     plt.legend()
#     plt.show()
    
train,valid,test=np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])# splitting the data to (train and test) data 

def scale_dataset(dataframe, oversample=False):
    x=dataframe[dataframe.columns[:-1]].values
    y=dataframe[dataframe.columns[-1]].values
    
    scaler=StandardScaler()
    x=scaler.fit_transform(x) 
    
    if oversample:
        ros=RandomOverSampler()
        x,y=ros.fit_resample(x,y)
    
    data=np.hstack((x,np.reshape(y,(-1,1))))
    
    return data,x,y

train , X_train, y_train =scale_dataset(train, oversample=True)
valid , X_valid, y_valid =scale_dataset(valid, oversample=False)
test , X_test, y_test =scale_dataset(test, oversample=False)

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred))    

# naive bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train,y_train)
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))    

#log regression

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model.fit(X_train,y_train)

y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))    

#SVM

from sklearn.svm import SVC

svm_model= SVC()
svm=svm_model.fit(X_train,y_train)

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))   
