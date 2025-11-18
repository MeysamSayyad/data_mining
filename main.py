import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from Question_2 import *
# arr = np.array([
#     [
#         [1,2,4,5]
#                  ,[1,99,25,98]
#                  ,[1,2,3,4]
#                  ],
#                  [
#                      [1,2,4,5]
#                   ,[1,99,25,98]
#                   ,[1,2,3,4]
#                   ]
#                   ])
# # print(arr[1,1,1])
# # print(arr.size == math.prod(arr.shape))
# # print(arr.dtype)


height =[155,158,160,162,163,165,168,170,172,175,177,178,180,181,183]


def computeZScore(arr,mean,var):
    retArr=[]
    for i in arr:
        retArr.append((i-mean)/var)
    return retArr






dataResults=pd.DataFrame([mean,std,min_value,max_value,Q1,Q2,Q3],index=['mean','std','min_value','max_value','Q1','Q2','Q3'],columns=df_numeric.columns)
print(dataResults)


Most_outliers=outliers_Count_Series[outliers_Count_Series.values == outliers_Count_Series.values[0]]
# Tax 5%,Total,gross income,cogs have most outliers
plt.figure() 
df_numeric.boxplot(list(Most_outliers.keys()))
plt.title('before Removing outliers')

df_numeric=Remove_outliers(df_numeric,Q1=Q1,Q3=Q3)
std_after=df_numeric.std()
std_df=pd.DataFrame({' before':std,' after':std_after})

print('\n\nstandard deviation(Before and After removing outliers):\n',std_df)


plt.figure()
df_numeric.boxplot(list(Most_outliers.keys()))
plt.title('after Removing outliers')



std_pastRemove=df_numeric.std()
print('before Removing Outliers:\n\n',std,'\n\nAfter Removing Outliers:\n\n',std_pastRemove)

def CosineSimilarity(df):
   norms = norm(df,axis=1,keepdims=True)
   divided=df/norms
   
   return np.dot(divided,divided.T)
def Euclidean(df):
   x= df.to_numpy()
#    sq_sum=np.sum(x**2,axis=1,keepdims=True)
   
#    result=sq_sum + sq_sum.T - 2*x.dot(x.T)
   
   result=np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)
   return np.sqrt(result)

df_cosine=pd.DataFrame(CosineSimilarity(df_numeric))
print(df_cosine)

# maxSim_first=df_cosine.max().sort_values(ascending=False).keys()[0]
# maxSim_second=df_cosine[maxSim_first].idxmax()
# plt.figure()
# print(df_cosine)
# df[[maxSim_first, maxSim_second]].plot.scatter('Total','Tax 5%')


# adding missing columns
def create_missing_columns(count,rowCount):
    global df
    
    for i in range(rowCount):
        sample=df.iloc[0,::].to_dict()
        headers=list(df)
        for i in range(count):
        
            index=np.random.randint(0,len(headers))
            
            
            sample.pop(headers[index],None)
            headers.pop(index)
        df =pd.concat([df,pd.DataFrame([sample])],ignore_index=True) 
create_missing_columns(8,100)
df_numeric=df.select_dtypes(['number'])
# get missing Data % 
print('\n missing data % : \n',(df.isna().sum()/len(df.index))*100)
#fill missing
df_fillMean=df_numeric.copy()
df_fillMedian=df_numeric.copy()
df_fillMode=df_numeric.copy()
# most frequent
df_fillMode=df_fillMode.apply(lambda x:x.fillna(x.mode()[0]))
print('Standard Deviation(most Frequent Fill)\n',df_fillMode.std())
# Mean 
df_fillMean=df_fillMean.apply(lambda x:x.fillna(x.mean()))
print('Standard Deviation(Mean Fill)\n',df_fillMean.std())
# median 
df_fillMedian=df_fillMedian.fillna(Q2)
print('Standard Deviation(Median Fill)\n',df_fillMedian.std())
print(df_fillMedian)
plt.show()

#changing nominal attributes Branch , City to Numeric
df=pd.get_dummies(df,columns=['Branch','City'],dtype='int')
df_changedTypes=df.loc[:,'Branch_A':'City_Yangon']
print('Cosine Similarity of two changed Attributes \n\n',CosineSimilarity(df_changedTypes))
print('\nEuclidean Similarity of two changed Attributes \n\n',Euclidean(df_changedTypes))
