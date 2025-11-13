import numpy as np

import pandas as pd

df=pd.read_csv('bloodTypes_numeric.csv')
df_h_b=df[['height','A','B','AB','O']]
arr_euc= np.array([[0]*15]*15,dtype='float')
arr_cosine= np.array([[0]*15]*15,dtype='float')
print(df_h_b)
def euclidision(x,y):
    sum=0
    for i in range(len(x)):
        
        sum+=(x.iloc[i]-y.iloc[i])**2
    
    return np.sqrt(sum)
def cosineSim(x,y):

    dot=np.dot(x,y)  
     
    
    return dot/(np.linalg.norm(x) * np.linalg.norm(y))

for i in range(15):
    for j in range(15):
       arr_euc[i,j]= euclidision(df_h_b.iloc[i,:],df_h_b.iloc[j,:])
       arr_cosine[i,j]=cosineSim(df_h_b.iloc[i,:],df_h_b.iloc[j,:])
    
df_euc=pd.DataFrame(arr_euc,columns=list(range(1,16)),index=list(range(1,16)))

df_euc.to_clipboard()
df_cosine=pd.DataFrame(arr_cosine,columns=list(range(1,16)),index=list(range(1,16)))
# df_cosine.to_clipboard()
