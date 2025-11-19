from Question_2 import Q2
import numpy as np
import pandas as pd
csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)
df['dateTime']=df["Date"] + ' '+df['Time']
df['dateTime']=pd.to_datetime(df['dateTime'],format='%m/%d/%Y %H:%M')
# convert to timestamp
df['dateTime']=df['dateTime'].astype('int64')//10**9
df_numeric=df.select_dtypes(include=['number']).copy()
# adding missing columns
def create_missing_columns(count,rowCount):
    global df_numeric
    
    for i in range(rowCount):
        sample=df_numeric.iloc[0,::].to_dict()
        headers=list(df_numeric)
        for i in range(count):
        
            index=np.random.randint(0,len(headers))
            
            
            sample.pop(headers[index],None)
            headers.pop(index)
        df_numeric =pd.concat([df_numeric,pd.DataFrame([sample])],ignore_index=True) 
create_missing_columns(3,100)

# get missing Data % 
print('\n missing data % : \n',(df_numeric.isna().sum()/len(df_numeric.index))*100)
#fill missing
df_fillMean=df_numeric.copy()
df_fillMedian=df_numeric.copy()
df_fillMode=df_numeric.copy()
# most frequent
df_fillMode=df_fillMode.apply(lambda x:x.fillna(x.mode()[0]))
print('Standard Deviation(most Frequent Fill)\n',df_fillMode.std())
# Mean 
df_fillMean=df_fillMean.apply(lambda x:x.fillna(x.mean()))
print('Standard Deviation(Mean )\n',df_fillMean.std())
# median 
df_fillMedian=df_fillMedian.fillna(Q2)
print('Standard Deviation(Median )\n',df_fillMedian.std())
print(df_fillMedian)

df_fillStd=pd.DataFrame({'Most Frequent std':df_fillMode.std(),
              'Mean Fill std':df_fillMean.std(),'Median Fill std':df_fillMedian.std()})


df_stdResult=pd.concat([df_numeric.std(),df_fillStd],axis=1)
df_stdResult=pd.DataFrame(df_stdResult)
df_stdResult.rename(columns={0:'original std'},inplace=True)
print('original & New Standard Deviation based on each fill Type \n',df_stdResult)
print('lowest standard deviation for each attribute\n',df_stdResult.idxmin(axis=1))