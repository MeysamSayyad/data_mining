from preProccess import df_clean
from mlxtend.frequent_patterns import fpgrowth,apriori
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,log_loss
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df_clean=df_clean.drop(['Invoice ID','Date','Time',],axis=1)
train,temp=train_test_split(df_clean,test_size=0.2,shuffle=True,random_state=42)
val,test=train_test_split(temp,test_size=0.5,shuffle=True,random_state=42)





class_field='Customer type'
X_train = train.drop(class_field,axis=1)
X_train=pd.get_dummies(X_train)
Y_train=train[class_field]
X_val = val.drop(class_field,axis=1)
X_val=pd.get_dummies(X_val)
Y_val=val[class_field]
X_test = test.drop(class_field,axis=1)
X_test=pd.get_dummies(X_test)
Y_test=test[class_field]
model_pca = LogisticRegression(max_iter=2000)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_val=scaler.fit_transform(X_val)
X_test=scaler.fit_transform(X_test)
pca=PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train)
X_val_pca=pca.fit_transform(X_val)
X_test_pca=pca.fit_transform(X_test)
model_pca.fit(X_train_pca,Y_train)

param_grid={'C':[0.01,0.1,10,100],'solver':['lbfgs'],'max_iter':[2000]}
le=LabelEncoder()

Y_encoded=le.fit_transform(Y_train)
x_min, x_max = X_train_pca[:,0].min() - 1, X_train_pca[:,0].max() + 1
y_min, y_max = X_train_pca[:,1].min() - 1, X_train_pca[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))


# Predict on the grid
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
if Z.dtype.kind in {'U','O'}:
    Z = le.transform(Z)  # or convert to numeric codes

Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=Y_encoded, edgecolor='k', cmap='coolwarm')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Boundary in PCA 2D space')
plt.show()