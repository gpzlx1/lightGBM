# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print('Loading data...')
train_data = pd.read_csv('train_all.csv')   # 读取数据
train_data['service_type']=train_data['service_type'].astype(int)
train_data['is_mix_service']=train_data['is_mix_service'].astype(int)
train_data['online_time']=train_data['online_time'].astype(int)
train_data['many_over_bill']=train_data['many_over_bill'].astype(int)
train_data['contract_type']=train_data['contract_type'].astype(int)
train_data['contract_time']=train_data['contract_time'].astype(int)
train_data['is_promise_low_consume']=train_data['is_promise_low_consume'].astype(int)
train_data['net_service']=train_data['net_service'].astype(int)
train_data['pay_times']=train_data['pay_times'].astype(int)
train_data['gender']=train_data['gender'].astype(int)
train_data['age']=train_data['age'].astype(int)
train_data['complaint_level']=train_data['complaint_level'].astype(int)
train_data['former_complaint_num']=train_data['former_complaint_num'].astype(int)
train_data['current_service']=train_data['current_service'].astype(int)

dit = {89950166:0, 89950167:1, 89950168:2, 90063345:3, 90109916:4, 90155946:5, 99999825:6, 99999826:7, 99999827:8, 99999828:9, 99999830:10}
antidit = {0: 89950166, 1: 89950167, 2: 89950168, 3: 90063345, 4: 90109916, 5: 90155946, 6: 99999825, 7: 99999826, 8: 99999827, 9: 99999828, 10: 99999830}
train_data.pop('user_id')
ty = train_data.pop('current_service')
ty = ty.astype(int)
y = [dit[x] for x in ty]

col = train_data.columns
x = train_data[col]
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)


train = lgb.Dataset(train_x,label=train_y)
valid = lgb.Dataset(valid_x, label=valid_y, reference=train)





# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 12,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                train,
                num_boost_round=20,
                valid_sets=valid,
                early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
temp_result = [y.argmax() for y in y_pred]

cnt1 = 0
cnt2 = 0
for i in range(len(valid_y)):
    if temp_result[i] == valid_y[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))


final_result = [antidit[x] for x in temp_result]
