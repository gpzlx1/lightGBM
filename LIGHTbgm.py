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



train_data.pop('user_id')
y = train_data.pop('current_service')
y = y.astype(int)
col = train_data.columns
x = train_data[col]
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)


train = lgb.Dataset(train_x,train_y)
valid = lgb.Dataset(valid_x, valid_y, reference=train)





# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
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
# eval
print('The rmse of prediction is:', mean_squared_error(valid_y, y_pred) ** 0.5)