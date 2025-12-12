import pandas as pd
import numpy as np

csv = 'supermarket_sales.csv'


# read csv file
df = pd.read_csv(csv)
# df['dateTime']=df["Date"] + ' '+df['Time']
# df['dateTime']=pd.to_datetime(df['dateTime'],format='%m/%d/%Y %H:%M')
# # convert to timestamp
# df['dateTime']=df['dateTime'].astype('int64')//10**9
df_numeric=df.select_dtypes(include=['number'])


#calculate mean,variance,minvalue,max_value,Q1,Q2,Q3
mean=df_numeric.mean()
std=df_numeric.std()
min_value=df_numeric.min()
max_value=df_numeric.max()
Q1=df_numeric.quantile(0.25)
Q2=df_numeric.quantile(0.5)
Q3=df_numeric.quantile(0.75)

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
df_Proccessed=Remove_outliers(df_numeric,Q1,Q3)

def deletePerfectCorrelations(df):
    corr_Mat=df.corr().abs()
    upper=corr_Mat.where(np.triu(np.ones(corr_Mat.shape),k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(to_drop,axis=1, inplace=True)
deletePerfectCorrelations(df_Proccessed)


df_Proccessed=df_Proccessed.reset_index(drop=True)
df_clean=df.loc[df_Proccessed.index].copy()


