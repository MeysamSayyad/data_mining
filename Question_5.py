import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)
df['dateTime']=df["Date"] + ' '+df['Time']
df['dateTime']=pd.to_datetime(df['dateTime'],format='%m/%d/%Y %H:%M')
# convert to timestamp
df['dateTime']=df['dateTime'].astype('int64')//10**9
df_numeric=df.select_dtypes(include=['number']).copy()
df_numeric.fillna(df_numeric.mean())
scaler=StandardScaler()
df_numeric_scaled=scaler.fit_transform(df_numeric)
df_cosine=cosine_similarity(df_numeric_scaled.T)
np.fill_diagonal(df_cosine,0)
df_cosine=pd.DataFrame(df_cosine,index=df_numeric.columns,columns=df_numeric.columns)
max_sim=df_cosine.stack().max()
max_sim_attrs=df_cosine.stack().idxmax()
print('Cosine similarity:\n',df_cosine)
print('attributes with most Similarity:',max_sim_attrs)
print('similarity Value:',np.clip(max_sim,-1,1))

df[[max_sim_attrs[0], max_sim_attrs[1]]].plot.scatter(max_sim_attrs[0],max_sim_attrs[1])
# db = DBSCAN(eps=25, min_samples=6)   # tweak these two values if needed
# df['cluster'] = db.fit_predict(df[['Total', 'Cost']])

# print("Number of clusters found (excluding noise):", df['cluster'].nunique() - 1 if -1 in df['cluster'].values else df['cluster'].nunique())
# print("Number of noise points (label = -1):     ", (df['cluster'] == -1).sum())

# # -------------------------------------------------
# # 3. Plot with beautiful colors
# # -------------------------------------------------
# plt.figure(figsize=(11, 8))

# # Unique cluster labels (including -1 for noise)
# labels = df['cluster'].unique()
# colors = sns.color_palette("tab10", len(labels))

# for i, label in enumerate(labels):
#     mask = df['cluster'] == label
#     if label == -1:
#         # Noise points → black X
#         plt.scatter(df.loc[mask, 'Total'], df.loc[mask, 'Cost'],
#                     c='black', s=80, marker='x', linewidth=2, label='Noise')
#     else:
#         plt.scatter(df.loc[mask, 'Total'], df.loc[mask, 'Cost'],
#                     s=90, label=f'Cluster {label}', alpha=0.85)

# plt.title('DBSCAN Clustering on Your Scatter Plot\n'
#           '(Dense lower part → one big cluster | High values → mostly noise)',
#           fontsize=16, pad=20)
# plt.xlabel('Total', fontsize=14)
# plt.ylabel('Cost', fontsize=14)
# plt.legend(title='DBSCAN Result', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
# plt.grid(True)
plt.title('Scatter plot between most similar attributes')
plt.show()
