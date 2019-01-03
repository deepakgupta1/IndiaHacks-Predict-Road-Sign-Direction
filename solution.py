# india hacks ml 2017: predict the road sign - here

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from datetime import datetime



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')



test['SignFacing (Target)'] = 'Rear'


train_test = pd.concat([train, test])


train_test['DetectedCamera'].loc[train_test['DetectedCamera'] == 'Rear'] = 0
train_test['DetectedCamera'].loc[train_test['DetectedCamera'] == 'Front'] = 1
train_test['DetectedCamera'].loc[train_test['DetectedCamera'] == 'Left'] = 2
train_test['DetectedCamera'].loc[train_test['DetectedCamera'] == 'Right'] = 3
train_test['SignFacing (Target)'].loc[train_test['SignFacing (Target)'] == 'Rear'] = 0
train_test['SignFacing (Target)'].loc[train_test['SignFacing (Target)'] == 'Front'] = 1
train_test['SignFacing (Target)'].loc[train_test['SignFacing (Target)'] == 'Left'] = 2
train_test['SignFacing (Target)'].loc[train_test['SignFacing (Target)'] == 'Right'] = 3


train_test['front'] = 0
train_test['rear'] = 0
train_test['left'] = 0
train_test['right'] = 0
train_test['front'].loc[train_test['DetectedCamera'] == 1] = 1
train_test['rear'].loc[train_test['DetectedCamera'] == 0] = 1
train_test['left'].loc[train_test['DetectedCamera'] == 2] = 1
train_test['right'].loc[train_test['DetectedCamera'] == 3] = 1
#train_test.drop(['DetectedCamera'], axis=1, inplace=True)


train_test['quadrant'] = 0
train_test['quadrant'].loc[train_test['AngleOfSign'] < 90] = 1
train_test['quadrant'].loc[(train_test['AngleOfSign'] > 90) & (train_test['AngleOfSign'] <= 180)] = 2
train_test['quadrant'].loc[(train_test['AngleOfSign'] > 180) & (train_test['AngleOfSign'] <= 270)] = 3
train_test['quadrant'].loc[(train_test['AngleOfSign'] > 270) & (train_test['AngleOfSign'] <= 360)] = 4


train_test['axis_angle'] = 0
train_test['axis_angle'].loc[(train_test['AngleOfSign'] < 45)] = train_test['AngleOfSign'].loc[(train_test['AngleOfSign'] < 45)]
train_test['axis_angle'].loc[(train_test['AngleOfSign'] >= 45) & (train_test['AngleOfSign'] < 135)] = np.abs(90-train_test['AngleOfSign'].loc[(train_test['AngleOfSign'] >= 45) & (train_test['AngleOfSign'] < 135)])
train_test['axis_angle'].loc[(train_test['AngleOfSign'] >= 135) & (train_test['AngleOfSign'] < 225)] = np.abs(180-train_test['AngleOfSign'].loc[(train_test['AngleOfSign'] >= 135) & (train_test['AngleOfSign'] < 225)])
train_test['axis_angle'].loc[(train_test['AngleOfSign'] >= 225) & (train_test['AngleOfSign'] < 315)] = np.abs(270-train_test['AngleOfSign'].loc[(train_test['AngleOfSign'] >= 225) & (train_test['AngleOfSign'] < 315)])
train_test['axis_angle'].loc[(train_test['AngleOfSign'] >= 315)] = np.abs(360-train_test['AngleOfSign'].loc[(train_test['AngleOfSign'] >= 315)])


train = train_test[:train.shape[0]]
test = train_test[train.shape[0]:]
predictors = ['AngleOfSign', 'SignAspectRatio', 'SignWidth', 'SignHeight', 'front', 'rear', 'left', 'right', 'quadrant', 'axis_angle']


# making the model now
def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=30):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['SignFacing (Target)'].values)
        xgb_param['num_class'] = 4
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='mlogloss', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
        
    alg.fit(dtrain[predictors], dtrain['SignFacing (Target)'], eval_metric='mlogloss')
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,:]
    
    print '\nModel Report:'
    print 'logloss (Train): ', metrics.log_loss(dtrain['SignFacing (Target)'].astype(int), np.array(dtrain_predprob))
    
    feat_imp = pd.Series(alg.booster().get_fscore())
    return alg


xgb1 = XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 1000,
    max_depth = 4,
    gamma = 0,
    objective = 'multi:softmax',
    #colsample_bytree = 0.7,
    #subsample = 0.8,
    #min_child_weight = 6,
    #reg_alpha = 0, 
    seed = 27
)
model1 = modelfit(xgb1, train, predictors)


submit = pd.DataFrame()
submit['Id'] = test['Id']
preds = model1.predict_proba(test[predictors])
submit['Front'] = preds[:, 1]
submit['Left'] = preds[:, 2]
submit['Rear'] = preds[:, 0]
submit['Right'] = preds[:, 3]
submit.to_csv('submit.csv', index=False)
