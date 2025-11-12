import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
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


# height =[155,158,160,162,163,165,168,170,172,175,177,178,180,181,183]


# def computeZScore(arr,mean,var):
#     retArr=[]
#     for i in arr:
#         retArr.append((i-mean)/var)
#     return retArr




csv = 'supermarket_sales.csv'

# read csv file
df = pd.read_csv(csv)
df['dateTime']=df["Date"] + ' '+df['Time']
df['dateTime']=pd.to_datetime(df['dateTime'],format='%m/%d/%Y %H:%M')
df['dateTime']=df['dateTime']


# df['dateTime_Std']=(df['dateTime_Std']-df['dateTime_Std'].mean())/df['dateTime_Std'].std()
df_numeric=df.select_dtypes(include=['number'])


#calculate mean,variance,minvalue,max_value,Q1,Q2,Q3
mean=df_numeric.mean()
std=df_numeric.var()
min_value=df_numeric.min()
max_value=df_numeric.max()
Q1=df_numeric.quantile(0.25)
Q2=df_numeric.quantile(0.5)
Q3=df_numeric.quantile(0.75)

print('mean:\n',mean,'\n\nstandard deviation:\n',std,'\n\nmin_value\n',min_value,'\n\nmax_value\n',max_value,
      '\n\nQ1\n',Q1,'\n\nQ2\n',Q2,'\n\nQ3\n',Q3)


def CountOutliers(df,Q1,Q3):
    outlier_Count= {}
    for i in df:
        
        IQR=Q3[i]-Q1[i]
        lowerBound=Q1[i]-1.5*IQR
        upperBound=Q3[i]+ 1.5*IQR
        outlier_Count[i]=((df[i] < lowerBound) | (df[i]> upperBound)).sum()
       
        
    
    return outlier_Count
def Remove_outliers(df,Q1,Q3):
    df_result =df.copy()
    for i in df:
        
        IQR=Q3[i]-Q1[i]
        lowerBound=Q1[i]-1.5*IQR
        upperBound=Q3[i]+ 1.5*IQR
        
        df_result = df_result[(df_result[i] >= lowerBound) & (df_result[i]<= upperBound)]
        
        
    
    return df_result
outliers_Count_Series= pd.Series(CountOutliers(df_numeric,Q1=Q1,Q3=Q3))
outliers_Count_Series=outliers_Count_Series.sort_values(ascending=False)
Most_outliers=outliers_Count_Series[outliers_Count_Series.values == outliers_Count_Series.values[0]]
# Tax 5%,Total,gross income,cogs have most outliers
plt.figure() 
df_numeric.boxplot(list(Most_outliers.keys()))
plt.title('before')

df_numeric=Remove_outliers(df_numeric,Q1=Q1,Q3=Q3)




plt.figure()
df_numeric.boxplot(list(Most_outliers.keys()))
plt.title('after')



std_pastRemove=df_numeric.std()
print('before Removing Outliers:\n\n',std,'\n\nAfter Removing Outliers:\n\n',std_pastRemove)

def CosineSimilarity(df):
   norms=norm(df.T,axis=1)
   
   return (df.T @ df)/(norms[:,None] * norms[None,:])
df_cosine=pd.DataFrame(CosineSimilarity(df_numeric),index=df_numeric.columns)
for i in range(len(df_cosine.index)):
    df_cosine.iloc[i,i]=0
maxSim_first=df_cosine.max().sort_values(ascending=False).keys()[0]
maxSim_second=df_cosine[maxSim_first].idxmax()
plt.figure()

df[[maxSim_first, maxSim_second]].plot.scatter('Total','Tax 5%')


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
create_missing_columns(5,100)
# get missing Data % 
print('\n missing data % : \n',(df.isna().sum()/len(df.index))*100)
plt.show()

