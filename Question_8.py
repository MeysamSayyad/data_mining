import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)
df_select=df[['Branch','Product line']]

#changing nominal attributes Branch , City to Numeric
df_changedTypes=pd.get_dummies(df_select,dtype='int')

cosine_sim=cosine_similarity(df_changedTypes)
euclidean_sim=euclidean_distances(df_changedTypes)
np.fill_diagonal(cosine_sim,0)
np.fill_diagonal(euclidean_sim,0)
print('Cosine Similarity of two changed Attributes \n\n',pd.DataFrame(cosine_sim))
print('\nEuclidean Similarity of two changed Attributes \n\n',pd.DataFrame(euclidean_sim))

cosine_sim_T=cosine_similarity(df_changedTypes.T)
euclidean_sim_T=euclidean_distances(df_changedTypes.T)
np.fill_diagonal(cosine_sim_T,0)
np.fill_diagonal(euclidean_sim_T,0)
df_cosine=pd.DataFrame(cosine_sim_T,index=df_changedTypes.columns,columns=df_changedTypes.columns)
df_euclidean=pd.DataFrame(euclidean_sim_T,index=df_changedTypes.columns,columns=df_changedTypes.columns)
print('Cosine Similarity of two changed Attributes \n\n',df_cosine)
print('\nEuclidean Similarity of two changed Attributes \n\n',df_euclidean)

df_cosine.to_clipboard()