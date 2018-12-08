import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print('Loading data...')
data = pd.read_csv('temp1.csv')   # 读取数据
test_data = data.loc[data['current_service'].isin(['89950166'])]
user_id_test_data = test_data.pop('user_id')
test_data.pop('current_service')
reversedict = {0: 89950166, 1: 99999830}
print('begin predict...')
bst = lgb.Booster(model_file='binary_model.txt')
y_pred = bst.predict(test_data)
temp_result = [int(y+0.5) for y in y_pred]
final_result = [reversedict[x] for x in temp_result]
DataSet = list(zip(user_id_test_data, final_result))
df = pd.DataFrame(data=DataSet, columns=['user_id', 'current_service'])
df.to_csv('temp3.csv', index=False, header=True, encoding='utf-8')

df1 = pd.read_csv('temp3.csv')
df2 = pd.read_csv('temp2.csv')
frames = [df1, df2]
result = pd.concat(frames)
result.to_csv('result.csv', index=False, header=True, encoding='utf-8' )