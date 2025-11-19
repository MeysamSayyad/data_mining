import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from seaborn import heatmap
from matplotlib import pyplot as plt

csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)
df['dateTime']=df["Date"] + ' '+df['Time']
df['dateTime']=pd.to_datetime(df['dateTime'],format='%m/%d/%Y %H:%M')
# convert to timestamp
df['dateTime']=df['dateTime'].astype('int64')//10**9

df_numeric_cols=df.select_dtypes('number').columns.tolist()
feature=df_numeric_cols[2]
df_numeric_oneColoumn=df[[feature]]
print(df_numeric_oneColoumn)

sim_matrix=cosine_similarity(df_numeric_oneColoumn)
print(sim_matrix)
heatmap(sim_matrix)
plt.title(f'Cosine Similarity Heatmap for {feature}')
plt.savefig('cosine_similarity_heatmap_one_attribute.png')

df_numeric=df.select_dtypes('number')
plt.figure()
df_corr=df_numeric.corr()
heatmap(df_corr)
plt.title('Correlation Heatmap for Numeric Attributes')

plt.savefig('correlation_heatmap.png')


plt.show()