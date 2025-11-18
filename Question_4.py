from Question_2 import *
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(csv)

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
fig,axes=plt.subplots(1,2)
df_numeric=df.select_dtypes(include=['number'])
outliers_Count_Series= pd.Series(CountOutliers(df_numeric,Q1=Q1,Q3=Q3))
outliers_Count_Series=outliers_Count_Series.sort_values(ascending=False)
Most_outliers=outliers_Count_Series[outliers_Count_Series.values == outliers_Count_Series.values[0]]
# Tax 5%,Total,gross income,cogs have most outliers

sns.boxplot(df_numeric[Most_outliers.keys()],ax=axes[0])
axes[0].set_title('before Removing outliers')
axes[0]

df_numeric=Remove_outliers(df_numeric,Q1=Q1,Q3=Q3)
std_after=df_numeric.std()
std_df=pd.DataFrame({' before':std,' after':std_after})

print('\n\nstandard deviation(Before and After removing outliers):\n',std_df)


print(df_numeric[Most_outliers.keys()])
sns.boxplot(df_numeric[Most_outliers.keys()],ax=axes[1])
axes[1].set_title('after Removing outliers ')
plt.suptitle('Boxplot for Columns with most Outliers Before & After removing outliers ')
plt.tight_layout()
plt.show()