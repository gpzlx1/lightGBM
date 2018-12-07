import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print('Loading data...')
test_data = pd.read_csv('test_org.csv')   # 读取数据
test_data['service_type']=test_data['service_type'].astype(int)
test_data['is_mix_service']=test_data['is_mix_service'].astype(int)
test_data['online_time']=test_data['online_time'].astype(int)
test_data['many_over_bill']=test_data['many_over_bill'].astype(int)
test_data['contract_type']=test_data['contract_type'].astype(int)
test_data['contract_time']=test_data['contract_time'].astype(int)
test_data['is_promise_low_consume']=test_data['is_promise_low_consume'].astype(int)
test_data['net_service']=test_data['net_service'].astype(int)
test_data['pay_times']=test_data['pay_times'].astype(int)
test_data['gender']=test_data['gender'].astype(int)
test_data['age']=test_data['age'].astype(int)
test_data['complaint_level']=test_data['complaint_level'].astype(int)
test_data['former_complaint_num']=test_data['former_complaint_num'].astype(int)
user_id = test_data.pop('user_id')
antidit = {0: 89950166, 1: 89950167, 2: 89950168, 3: 90063345, 4: 90109916, 5: 90155946, 6: 99999825, 7: 99999826, 8: 99999827, 9: 99999828, 10: 99999830}

print('begin predict...')
bst = lgb.Booster(model_file='modewait.txt')
y_pred = bst.predict(test_data)
temp_result = [y.argmax() for y in y_pred]
final_result = [antidit[x] for x in temp_result]
test_data.insert(0,'user_id',user_id)
test_data.insert(1,'current_service',final_result)
test_data.to_csv('predictwait.csv', index=False, header=True,encoding='utf-8' )

DataSet = list(zip(user_id,final_result))
df = pd.DataFrame(data = DataSet ,columns=['user_id','current_service'])
df = df.loc[df['current_service'] != 89950166]
df.to_csv('temp1.csv', index=False, header=True,encoding='utf-8' )