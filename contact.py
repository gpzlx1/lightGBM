import pandas as pd

df1 = pd.read_csv('temp1.csv')
df2 = pd.read_csv('temp2.csv')

frames = [df1,df2]
result = pd.concat(frames)

result.to_csv('temp3.csv', index=False, header=True,encoding='utf-8' )