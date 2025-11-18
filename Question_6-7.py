from Question_2 import Q2
import numpy as np
import pandas as pd
csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)
df_numeric=df.select_dtypes(include=['number']).copy()
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
print('Standard Deviation(Mean )\n',df_fillMean.std())
# median 
df_fillMedian=df_fillMedian.fillna(Q2)
print('Standard Deviation(Median )\n',df_fillMedian.std())
print(df_fillMedian)

df_fillStd=pd.DataFrame({'Most Frequent std':df_fillMode.std(),
              'Mean Fill std':df_fillMean.std(),'Median Fill std':df_fillMedian.std()})
print('New Standard Deviation based on each fill Type \n',df_fillStd)
df_stdResult=pd.concat([df_numeric.std(),df_fillStd],axis=1)
print(df_stdResult)
print(df_stdResult.idxmin(axis=1))