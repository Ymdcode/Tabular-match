import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from deepforest import CascadeForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pwd = os.getcwd()

data=pd.read_csv(pwd+"/data/train.csv")
test = pd.read_csv(pwd+'/data/test.csv')

value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
data = data.replace({'target':value_map})
data = data.drop(columns=['id'])

x_train = data.iloc[:, :-1]
y_train = data['target']
x_test = test.iloc[:, 1:]

#deepforest
model = CascadeForestClassifier(n_jobs=2, n_estimators=4, n_trees=100)
model.fit(x_train.values, y_train.values)
proba = model.predict_proba(x_test.values)
output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
output.to_csv('deepforest.csv', index=False)

#randomforest
model = RandomForestClassifier(n_jobs=2, n_estimators=400)
model.fit(x_train, y_train)
proba = model.predict_proba(x_test)
output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
output.to_csv('randomforest.csv', index=False)

#
model = HistGradientBoostingClassifier(max_iter=200,
                                       validation_fraction=None,
                                       learning_rate=0.01,
                                       max_depth=10,
                                       min_samples_leaf=24,
                                       max_leaf_nodes=60,
                                       random_state=111,
                                       verbose=1)
model.fit(x_train, y_train)
proba = model.predict_proba(x_test)
output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
output.to_csv('HGBDT.csv', index=False)

#xgboost
parms = {'learning_rate': 0.03817329673009776, 'gamma': 0.3993428240049768, 'reg_alpha': 3,
         'reg_lambda': 1, 'n_estimators': 334, 'colsample_bynode': 0.2695766080178446,
         'colsample_bylevel': 0.6832712495239914, 'subsample': 0.6999062848890633,
         'min_child_weight': 100, 'colsample_bytree': 0.34663755614898173}

model = XGBClassifier(objective='multi:softprob',
                      eval_metric = "mlogloss",
                      num_class = 9,
                      tree_method = 'gpu_hist',
                      max_depth = 14,
                      use_label_encoder=False, **parms)
model.fit(x_train, y_train)
proba = model.predict_proba(x_test)
output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
output.to_csv('xgboost.csv', index=False)

#LGBM
model = LGBMClassifier(objective = 'multiclass',
                       reg_lambda = 10,
                       learning_rate = 0.1,
                       max_depth = 4,
                       seed = 14000605,
                       colsample_bytree = 0.5,
                       subsample = 0.9,
                       is_unbalance = True)
model.fit(x_train, y_train)
proba = model.predict_proba(x_test)
output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
output.to_csv('lgbm.csv', index=False)