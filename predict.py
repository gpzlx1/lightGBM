import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print('Loading data...')
data = pd.read_csv('test.csv')   # 读取数据
user_id = data.pop('user_id')

antidit = {0: 89950166, 1: 89950167, 2: 89950168, 3: 90063345, 4: 90109916, 5: 90155946, 6: 99999825, 7: 99999826, 8: 99999827, 9: 99999828, 10: 99999830}

print('begin predict...')
bst = lgb.Booster(model_file='model.txt')
y_pred = bst.predict(data)
temp_result = [y.argmax() for y in y_pred]
final_result = [antidit[x] for x in temp_result]

