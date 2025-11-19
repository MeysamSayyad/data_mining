import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.preprocessing import StandardScaler
csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)
df_numeric=df[['Unit price','Quantity']]
scaler=StandardScaler()
df_standard=scaler.fit_transform(df_numeric)

#changing nominal attributes Branch , City to Numeric


# cosine_sim=cosine_similarity(df_standard)
# euclidean_sim=euclidean_distances(df_standard)
# np.fill_diagonal(cosine_sim,0)
# np.fill_diagonal(euclidean_sim,0)
# print('Cosine Similarity of two changed Attributes (Rows) \n\n',pd.DataFrame(cosine_sim))
# print('\nEuclidean Similarity of two changed Attributes (Rows) \n\n',pd.DataFrame(euclidean_sim))

cosine_sim_T=cosine_similarity(df_standard.T)
euclidean_sim_T=euclidean_distances(df_standard.T)
np.fill_diagonal(cosine_sim_T,0)
np.fill_diagonal(euclidean_sim_T,0)
df_cosine=pd.DataFrame(cosine_sim_T,index=df_numeric.columns,columns=df_numeric.columns)
df_euclidean=pd.DataFrame(euclidean_sim_T,index=df_numeric.columns,columns=df_numeric.columns)
print('Cosine Similarity of two numeric Attributes  \n\n',df_cosine)
print('\nEuclidean Similarity of two numeric Attributes  \n\n',df_euclidean)

df_cosine.to_clipboard()