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

model = LogisticRegression(max_iter=2000)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_val=scaler.fit_transform(X_val)
X_test=scaler.fit_transform(X_test)

model.fit(X_train,Y_train)

param_grid={'C':[0.001,0.01,0.1,10,100],'solver':['lbfgs'],'max_iter':[2000]}

grid =GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=1)
grid.fit(X_val,Y_val)
print('Best C:',grid.best_params_['C'])
print('Best CV Accuracy:',grid.best_score_)
# best_model=grid.best_estimator_
# best_model.fit(X_train,Y_train)
print('-'*40)

# Y_val_pred=best_model.predict(X_val)
# Y_val_pred_prob=best_model.predict_proba(X_val)
# print('Validation:\n')
# print('validation Accuracy:',
#       accuracy_score(Y_val,Y_val_pred))

# print(classification_report(Y_val,Y_val_pred))
# print('validation Confusion Matrix:\n',confusion_matrix(Y_val,Y_val_pred))
# print('loss Function:',log_loss(Y_val,Y_val_pred_prob))
print('-'*40)

best_model =LogisticRegression(C=grid.best_params_['C'],max_iter=2000)
best_model.fit(X_train,Y_train)
X_final=np.vstack([X_train,X_val])
Y_final=np.hstack([Y_train,Y_val])

best_model.fit(X_final,Y_final)
Y_test_pred=best_model.predict(X_test)
Y_test_pred_prob=best_model.predict_proba(X_test)
y_pred=best_model.predict(X_train)

print('Test:\n')
print('Test Accuracy:',
      accuracy_score(Y_test,Y_test_pred))

print('Member:1  Normal =0\n')
print(classification_report(Y_test,Y_test_pred))
print('Test Confusion Matrix:\n',confusion_matrix(Y_test,Y_test_pred))
print('loss Function:',log_loss(Y_test,Y_test_pred_prob))
print('\nIntercept (bias):',best_model.intercept_)
print('\nCoefficients (weights):',best_model.coef_)
print('Number of parameters :',best_model.coef_.size +1)

