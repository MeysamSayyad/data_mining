from preProccess import df_clean
from mlxtend.frequent_patterns import fpgrowth,association_rules
import pandas as pd


sup=0.4
conf=0.75

for col in df_clean.select_dtypes(include=['number']).columns:
    df_clean[col]=pd.qcut(df_clean[col],q=3,duplicates='drop')

df_encoded=pd.get_dummies(df_clean)

df_frequent=fpgrowth(df_encoded,min_support=sup,use_colnames=True)
print('frequent Patterns - الگو های مکرر\n',df_frequent)

