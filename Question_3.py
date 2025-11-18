import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from Question_2 import std,df_numeric,Q1,Q3

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
df_numeric=Remove_outliers(df_numeric,Q1=Q1,Q3=Q3)
std_after=df_numeric.std()
std_df=pd.DataFrame({' before':std,' after':std_after,'diffrence':std-std_after})

print('\n\nstandard deviation(Before and After removing outliers):\n',std_df)
