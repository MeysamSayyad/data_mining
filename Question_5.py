import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)
df_numeric=df.select_dtypes(include=['number']).copy()
df_numeric.fillna(df_numeric.mean())
scaler=StandardScaler()
df_numeric_scaled=scaler.fit_transform(df_numeric)
df_cosine=cosine_similarity(df_numeric_scaled.T)
np.fill_diagonal(df_cosine,0)
df_cosine=pd.DataFrame(df_cosine,index=df_numeric.columns,columns=df_numeric.columns)
max_sim=df_cosine.stack().max()
max_sim_attrs=df_cosine.stack().idxmax()
print('attributes with most Similarity:',max_sim_attrs)
print('similarity Value:',np.clip(max_sim,-1,1))

df[[max_sim_attrs[0], max_sim_attrs[1]]].plot.scatter(max_sim_attrs[0],max_sim_attrs[1])
plt.title('Scatter plot between most similar attributes')
plt.show()
