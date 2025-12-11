from preProccess import df_clean
from mlxtend.frequent_patterns import fpgrowth,apriori
import pandas as pd
import time


sup=0.4
conf=0.75

for col in df_clean.select_dtypes(include=['number']).columns:
    df_clean[col]=pd.qcut(df_clean[col],q=3,duplicates='drop')

df_encoded=pd.get_dummies(df_clean)
start_fpGrowth=time.time()
df_frequent=fpgrowth(df_encoded,min_support=sup,use_colnames=True)
end_fpGrowth=time.time()

start_Apriori=time.time()
df_frequent=apriori(df_encoded,min_support=sup,use_colnames=True)
end_Apriori=time.time()
fp_Growth_time=end_fpGrowth-start_fpGrowth
Apriori_time=end_Apriori-start_Apriori
print('\ntime spent for FP-Growth:',fp_Growth_time)
print('\ntime spent for Apriori:',Apriori_time)
print('\ntime Diffrence between two algorithms(Apriori-FP-Growth):',Apriori_time-fp_Growth_time)


