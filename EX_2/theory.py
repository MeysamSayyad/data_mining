from mlxtend.frequent_patterns import apriori,fpgrowth,association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import time 

FrequentData= [['A', 'B','C' ,'E'],
['B', 'D'],
['B', 'C'],
['A' ,'B' ,'D'],
['A', 'B', 'C', 'E'],
['A', 'B', 'C', 'E'],
['A' ,'B', 'C', 'E'],
['A' ,'B', 'D'],
['B', 'C', 'E'],
]
# Transforming FrequentData  for using as input for Fp-growth and Apriori
te =TransactionEncoder()
te_ary=te.fit(FrequentData).transform(FrequentData)


df = pd.DataFrame(te_ary,columns=te.columns_)


frequent_patterns_FPGrowth= fpgrowth(df,min_support=0.33,use_colnames=True)
accociationRules= association_rules(frequent_patterns_FPGrowth)
# Getting Valid accociation Rules
print(accociationRules)

# Measuring Calculation Time with Support 1 for Apriori & FP-growth
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


# Measuring Calculation Time with Support 6 for Apriori & FP-growth
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
secondTry=1.465+ (df_Grades['Study hours per week(x1)-Norm ']*0.37693)+(df_Grades['Bedtime(x2)-Norm']*-0.32034)
SecondTry_diffrence=secondTry - df_Grades['Final score (out of 20)(y)']
secondtry_multiply_x1=df_Grades['Study hours per week(x1)-Norm ']*SecondTry_diffrence
secondtry_multiply_x2=df_Grades['Bedtime(x2)-Norm']*SecondTry_diffrence
secondTry_result_1=(SecondTry_diffrence.sum()*0.1)/20
print(secondTry)
print(SecondTry_diffrence)
print(secondTry_result_1)
print((secondtry_multiply_x1.sum()*0.1)/20)
print((secondtry_multiply_x2.sum()*0.1)/20)
thetha_0=1.465-secondTry_result_1
thetha_1=0.37693 -((secondtry_multiply_x1.sum()*0.1)/20)
thetha_2=-0.32034-((secondtry_multiply_x2.sum()*0.1)/20)
print('x0 second try result',thetha_0)
print('x1 second try result',thetha_1)
print('x2 second try result',thetha_2)



thirdTry=thetha_0+ (df_Grades['Study hours per week(x1)-Norm ']* thetha_1)+(df_Grades['Bedtime(x2)-Norm']*thetha_2)
thirdTry_diffrence=thirdTry - df_Grades['Final score (out of 20)(y)']
thirdTry_multiply_x1=df_Grades['Study hours per week(x1)-Norm ']*thirdTry_diffrence
thirdtry_multiply_x2=df_Grades['Bedtime(x2)-Norm']*thirdTry_diffrence
thirdTry_result_1=(thirdTry_diffrence.sum()*0.1)/20
print(thirdTry)
print(thirdTry_diffrence)
print(thirdTry_result_1)
print((thirdTry_multiply_x1.sum()*0.1)/20)
print((thirdtry_multiply_x2.sum()*0.1)/20)
thetha_0=thetha_0-thirdTry_result_1
thetha_1=thetha_1 -((thirdTry_multiply_x1.sum()*0.1)/20)
thetha_2=thetha_2-((thirdtry_multiply_x2.sum()*0.1)/20)
print('x0 third try result',thetha_0)
print('x1 third try result',thetha_1)
print('x2 third try result',thetha_2)

final_h=thetha_0+ (df_Grades['Study hours per week(x1)-Norm ']* thetha_1)+(df_Grades['Bedtime(x2)-Norm']*thetha_2)
print(final_h)
final_diff_to2=(final_h-df_Grades['Final score (out of 20)(y)'])**2
print(final_diff_to2)
LossFunc=(final_diff_to2.sum())/40
print('new Loss function',LossFunc)