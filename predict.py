import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print('Loading data...')
data = pd.read_csv('test.csv')   # 读取数据
user_id = data.pop('user_id')



print('begin predict...')
bst = lgb.Booster(model_file='model.txt')
y_pred = bst.predict(data)