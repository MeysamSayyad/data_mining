import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from seaborn import heatmap
from matplotlib import pyplot as plt

csv = 'supermarket_sales.csv'
df = pd.read_csv(csv)

df_numeric_cols=df.select_dtypes('number').columns.tolist()
feature=df_numeric_cols[1]
df_numeric_oneColoumn=df[[feature]]


sim_matrix=cosine_similarity(df_numeric_oneColoumn)
print(sim_matrix)
heatmap(sim_matrix)
plt.title('Cosine Similarity Heatmap for one Numeric Attribute')
plt.savefig('cosine_similarity_heatmap_one_attribute.png')

df_numeric=df.select_dtypes('number')
plt.figure()
df_corr=df_numeric.corr()
heatmap(df_corr)
plt.title('Correlation Heatmap for Numeric Attributes')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')


plt.show()