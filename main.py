import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
df['dateTime_Std']=df["Date"] + ' '+df['Time']
df['dateTime_Std']=pd.to_datetime(df['dateTime_Std'],format='%m/%d/%Y %H:%M')
df['dateTime_Std']=df['dateTime_Std']


# df['dateTime_Std']=(df['dateTime_Std']-df['dateTime_Std'].mean())/df['dateTime_Std'].std()
df_numeric=df.select_dtypes(include=['number'])


#calculate mean,variance,minvalue,max_value,Q1,Q2,Q3
mean=df_numeric.mean()
variance=df_numeric.var()
min_value=df_numeric.min()
max_value=df_numeric.max()
Q1=df_numeric.quantile(0.25)
Q2=df_numeric.quantile(0.5)
Q3=df_numeric.quantile(0.75)

# print('mean:\n',mean,'\n\nvariance:\n',variance,'\n\nmin_value\n',min_value,'\n\nmax_value\n',max_value,
#       '\n\nQ1\n',Q1,'\n\nQ2\n',Q2,'\n\nQ3\n',Q3)

outlier_Count= {}
def Remove_outliers(df,Q1,Q3):
    for i in df:
        
        IQR=Q3[i]-Q1[i]
        lowerBound=Q1[i]-1.5*IQR
        upperBound=Q3[i]+ 1.5*IQR
        outlier_Count[i]=((df[i] < lowerBound) | (df[i]> upperBound)).sum()
        df = df[(df[i] >= lowerBound) & (df[i]<= upperBound)]
        
      
    return df
plt.figure() 
df_numeric.boxplot('Tax 5%')
plt.title('before')

df_numeric=Remove_outliers(df_numeric,Q1=Q1,Q3=Q3)
outliers_Count_Series= pd.Series(outlier_Count)
outliers_Count_Series=outliers_Count_Series.sort_values(ascending=False)
# Tax 5% has most outliers
plt.figure()
df_numeric.boxplot('Tax 5%')
plt.title('after')

plt.show()

variance_pastRemove=df_numeric.var()
print('before Removing Outliers:\n\n',variance,'\n\nAfter Removing Outliers:\n\n',variance_pastRemove)