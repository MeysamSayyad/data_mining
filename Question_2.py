import pandas as pd
csv = 'supermarket_sales.csv'

# read csv file
df = pd.read_csv(csv)
# df['dateTime']=df["Date"] + ' '+df['Time']
# df['dateTime']=pd.to_datetime(df['dateTime'],format='%m/%d/%Y %H:%M')
# df['dateTime']=df['dateTime']


# df['dateTime_Std']=(df['dateTime_Std']-df['dateTime_Std'].mean())/df['dateTime_Std'].std()
df_numeric=df.select_dtypes(include=['number'])


#calculate mean,variance,minvalue,max_value,Q1,Q2,Q3
mean=df_numeric.mean()
std=df_numeric.std()
min_value=df_numeric.min()
max_value=df_numeric.max()
Q1=df_numeric.quantile(0.25)
Q2=df_numeric.quantile(0.5)
Q3=df_numeric.quantile(0.75)

dataResults=pd.DataFrame([mean,std,min_value,max_value,Q1,Q2,Q3],index=['mean','std','min_value','max_value','Q1','Q2','Q3'],columns=df_numeric.columns)
print(dataResults)