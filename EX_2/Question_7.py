from preProccess import df_clean
from mlxtend.frequent_patterns import fpgrowth,apriori
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,log_loss,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
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

param_grid={'C':[0.001,0.01,0.1,10,100],'solver':['lbfgs'],'max_iter':[2000]}
le=LabelEncoder()

Y_encoded=le.fit_transform(Y_train)
Y_encoded_test=le.fit_transform(Y_test)
grid =GridSearchCV(estimator=model_pca,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=1)
grid.fit(X_val,Y_val)
print('Best C:',grid.best_params_['C'])
print('Best CV Accuracy:',grid.best_score_)
# best_model=grid.best_estimator_
# best_model.fit(X_train,Y_train)
print('-'*40)


best_model =LogisticRegression(C=100000000,max_iter=2000)
best_model.fit(X_train_pca,Y_train)
X_final_pca=np.vstack([X_train_pca,X_val_pca])
Y_final=np.hstack([Y_train,Y_val])

best_model.fit(X_final_pca,Y_final)
y_pred=best_model.predict(X_train_pca)
y_pred_test=best_model.predict(X_test_pca)



display=DecisionBoundaryDisplay.from_estimator(best_model,X_train_pca,response_method='predict',response=y_pred,cmap=plt.cm.RdYlBu,alpha=0.3)
display.plot()
display.ax_.set_xlabel('PC1')
display.ax_.set_ylabel('PC2')
display.ax_.set_title('Decision Boundary Scatter Traindata')
display.ax_.scatter(
    X_train_pca[:, 0],X_train_pca[:, 1], c=Y_encoded, edgecolor="black",label='Member'
)
display.ax_.legend()

plt.figure()

display=DecisionBoundaryDisplay.from_estimator(best_model,X_test_pca,response_method='predict',response=y_pred_test,cmap=plt.cm.RdYlBu,alpha=0.3)
display.plot()
display.ax_.set_xlabel('PC1')
display.ax_.set_ylabel('PC2')
display.ax_.set_title('Decision Boundary Scatter TestData')
print(Y_test)
print(Y_encoded_test)
display.ax_.scatter(
    X_test_pca[:, 0],X_test_pca[:, 1], c=Y_encoded_test, edgecolor="black",label='Normal'
)
display.ax_.legend()

plt.show()