from mlxtend.frequent_patterns import apriori,fpgrowth,association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import time 

data= [['A', 'B','C' ,'E'],
['B', 'D'],
['B', 'C'],
['A' ,'B' ,'D'],
['A', 'B', 'C', 'E'],
['A', 'B', 'C', 'E'],
['A' ,'B', 'C', 'E'],
['A' ,'B', 'D'],
['B', 'C', 'E'],
]

te =TransactionEncoder()
te_ary=te.fit(data).transform(data)


df = pd.DataFrame(te_ary,columns=te.columns_)


frequent_patterns_FPGrowth= fpgrowth(df,min_support=0.33,use_colnames=True)
accociationRules= association_rules(frequent_patterns_FPGrowth)
print(accociationRules)

df = pd.DataFrame(te_ary,columns=te.columns_)
start = time.time()
frequent_patterns= apriori(df,min_support=0.12,use_colnames=True)
end=time.time()
start_FPGrowth=time.time()
frequent_patterns_FPGrowth= fpgrowth(df,min_support=0.12,use_colnames=True)

end_FPGrowth=time.time()
print('min_sup = 1\n')

print(frequent_patterns.T ,f'\ntime used for Apriori :{end-start}')
print(frequent_patterns_FPGrowth.T ,f'\ntime used FP growth :{end_FPGrowth-start_FPGrowth}')
print('Apriori - FPGrowth',(end-start)-(end_FPGrowth-start_FPGrowth))

df = pd.DataFrame(te_ary,columns=te.columns_)
start = time.time()
frequent_patterns= apriori(df,min_support=0.67,use_colnames=True)
end=time.time()
start_FPGrowth=time.time()
frequent_patterns_FPGrowth= fpgrowth(df,min_support=0.67,use_colnames=True)
end_FPGrowth=time.time()
print('\nmin_sup = 6\n')
print(frequent_patterns.T ,f'\ntime used for Apriori :{end-start}')
print(frequent_patterns_FPGrowth.T ,f'\ntime used FP growth :{end_FPGrowth-start_FPGrowth}')
print('Apriori - FPGrowth',(end-start)-(end_FPGrowth-start_FPGrowth))

df_Grades_Org=pd.read_csv('EX_2/Grades_Org.csv')
print(df_Grades_Org.mean())
print(df_Grades_Org.std())
normalized_df=(df_Grades_Org-df_Grades_Org.mean())/df_Grades_Org.std()
print(normalized_df)
df_Grades=pd.read_csv('EX_2/Grades.csv')
print(df_Grades)
print(df_Grades['Final score (out of 20)(y)'].sum())
print((df_Grades['Study hours per week(x1)-Norm ']*-df_Grades['Final score (out of 20)(y)']))
print((df_Grades['Study hours per week(x1)-Norm ']*-df_Grades['Final score (out of 20)(y)']).sum())
print((df_Grades['Bedtime(x2)-Norm']*-df_Grades['Final score (out of 20)(y)']))
print((df_Grades['Bedtime(x2)-Norm']*-df_Grades['Final score (out of 20)(y)']).sum())