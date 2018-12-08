import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print('Loading data...')
test_data = pd.read_csv('test_org.csv')   # 读取数据
header = ['service_type', 'is_mix_service', 'online_time', 'many_over_bill',
          'contract_type', 'contract_time', 'is_promise_low_consume',
          'net_service', 'pay_times', 'gender', 'age', 'complaint_level', 'former_complaint_num']
test_data[header] = test_data[header].astype(int)

user_id = test_data.pop('user_id')
reversedict = {0: 89950166, 1: 89950167, 2: 89950168, 3: 90063345, 4: 90109916, 5: 90155946,
           6: 99999825, 7: 99999826, 8: 99999827, 9: 99999828, 10: 99999830}

print('begin predict...')
bst = lgb.Booster(model_file='model.txt')
y_pred = bst.predict(test_data)
temp_result = [y.argmax() for y in y_pred]
final_result = [reversedict[x] for x in temp_result]

test_data.insert(0, 'user_id', user_id)
test_data.insert(1, 'current_service', final_result)
test_data.to_csv('temp1.csv', index=False, header=True, encoding='utf-8')

DataSet = list(zip(user_id, final_result))
df = pd.DataFrame(data=DataSet, columns=['user_id','current_service'])
df = df[df['current_service'] != 89950166]
df.to_csv('temp2.csv', index=False, header=True, encoding='utf-8')
