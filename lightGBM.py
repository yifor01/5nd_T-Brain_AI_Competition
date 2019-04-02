# lightGBM

from sklearn.grid_search import GridSearchCV  # Perforing grid search
import warnings
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

print('Load Data...............')
df = pd.read_csv("x_train_cat.csv")
test = pd.read_csv("x_test_cat.csv")


# FX model 
print('FX model..................')
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y2[df.CUST_START_DT<9998]
x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=.7,random_state=1234)

print('CV........................')
params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'learning_rate':0.04,
          'num_leaves':30, 
          'max_depth': 5
    }
    
data_train = lgb.Dataset(x_train,y_train)
cv_results = lgb.cv(params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, 
                    metrics='auc',early_stopping_rounds=200,seed=0)
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())


# step1
params_test1={'max_depth': [13], 
              'num_leaves':[50],
              'scale_pos_weight': [7,8,9],
              'cat_smooth':[4,5,6],
              'min_child_weight':[1,2,3]}

gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                                       objective='binary',
                                                       learning_rate=0.04, 
                                                       n_estimators= len(cv_results['auc-mean']), 
                                                       max_depth=5, 
                                                       bagging_fraction = 0.8,
                                                       feature_fraction = 0.8,n_jobs=6), 
                       param_grid = params_test1, scoring='f1',cv=5,verbose=3)
gsearch1.fit(x_train,y_train)
print(gsearch1.best_params_, gsearch1.best_score_)

#############################
# step 2
params_test2={'max_bin': list(range(5,256,10)), 
              'min_data_in_leaf':list(range(1,102,10))}
              
gsearch2 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                                       objective='binary',
                                                       learning_rate=0.04, 
                                                       min_child_weight = gsearch1.best_params_['min_child_weight'],
                                                       max_depth = gsearch1.best_params_['max_depth'], 
                                                       scale_pos_weight = gsearch1.best_params_['scale_pos_weight'], 
                                                       num_leaves = gsearch1.best_params_['num_leaves'], 
                                                       cat_smooth = gsearch1.best_params_['cat_smooth'],                  
                                                       n_estimators= len(cv_results['auc-mean']), 
                                                       bagging_fraction = 0.8,
                                                       feature_fraction = 0.8), 
                       param_grid = params_test2, scoring='f1',cv=5,verbose=3)
gsearch2.fit(x_train,y_train)
gsearch2.best_params_, gsearch2.best_score_


###############################
# step 3
params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_freq': list(range(0,81,10))
}
              
gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                                       objective='binary',
                                                       learning_rate=0.04, 
                                                       min_child_weight = gsearch1.best_params_['min_child_weight'],
                                                       max_depth = gsearch1.best_params_['max_depth'], 
                                                       scale_pos_weight = gsearch1.best_params_['scale_pos_weight'], 
                                                       num_leaves = gsearch1.best_params_['num_leaves'], 
                                                       cat_smooth = gsearch1.best_params_['cat_smooth'],                  
                                                       n_estimators= len(cv_results['auc-mean']), 
                                                       max_bin = gsearch2.best_params_['max_bin'],  
                                                       min_data_in_leaf = gsearch2.best_params_['min_data_in_leaf'],  
                                                       bagging_fraction = 0.8,
                                                       feature_fraction = 0.8), 
                       param_grid = params_test3, scoring='f1',cv=5,verbose=3)

gsearch3.fit(x_train,y_train)
gsearch3.best_params_, gsearch3.best_score_
#######################################
# step 4
params_test4={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
              'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
}
              
gsearch4 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                                       objective='binary',
                                                       learning_rate=0.04, 
                                                       min_child_weight = gsearch1.best_params_['min_child_weight'],
                                                       max_depth = gsearch1.best_params_['max_depth'], 
                                                       scale_pos_weight = gsearch1.best_params_['scale_pos_weight'], 
                                                       num_leaves = gsearch1.best_params_['num_leaves'], 
                                                       cat_smooth = gsearch1.best_params_['cat_smooth'],                  
                                                       n_estimators= len(cv_results['auc-mean']), 
                                                       max_bin = gsearch2.best_params_['max_bin'],  
                                                       min_data_in_leaf = gsearch2.best_params_['min_data_in_leaf'],  
                                                       bagging_fraction = gsearch3.best_params_['bagging_fraction'],
                                                       feature_fraction = gsearch3.best_params_['feature_fraction'],
                                                       bagging_freq = gsearch3.best_params_['bagging_freq'] ), 
                       param_grid = params_test4, scoring='f1',cv=5,verbose=3)
gsearch4.fit(x_train,y_train)
gsearch4.best_params_, gsearch4.best_score_
#######################################
# step 5
params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
              
gsearch5 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                                       objective='binary',
                                                       learning_rate=0.04, 
                                                       min_child_weight = gsearch1.best_params_['min_child_weight'],
                                                       max_depth = gsearch1.best_params_['max_depth'], 
                                                       scale_pos_weight = gsearch1.best_params_['scale_pos_weight'], 
                                                       num_leaves = gsearch1.best_params_['num_leaves'], 
                                                       cat_smooth = gsearch1.best_params_['cat_smooth'],                  
                                                       n_estimators= len(cv_results['auc-mean']), 
                                                       max_bin = gsearch2.best_params_['max_bin'],  
                                                       min_data_in_leaf = gsearch2.best_params_['min_data_in_leaf'],  
                                                       bagging_fraction = gsearch3.best_params_['bagging_fraction'],
                                                       feature_fraction = gsearch3.best_params_['feature_fraction'],
                                                       bagging_freq = gsearch3.best_params_['bagging_freq'],
                                                       lambda_l1 =  gsearch4.best_params_['lambda_l1'],
                                                       lambda_l2 = gsearch4.best_params_['lambda_l2']  ), 
                       param_grid = params_test5, scoring='f1',cv=5,verbose=3)
gsearch5.fit(x_train,y_train)
gsearch5.best_params_, gsearch5.best_score_
#######################################
# step 6  (lr + nround)
from sklearn.metrics import f1_score
import numpy as np

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

lgbtrain = lgb.Dataset(x_train,label=y_train)
lgbval = lgb.Dataset(x_valid,label= y_valid)
for eta in [0.005,0.01,0.02,0.03,0.04,0.05]:
    evals_result = {}
    num_round = 4000
    params = {'boosting_type': 'gbdt',
              'objective': 'binary',          
              'learning_rate':eta ,
              'num_boost_round':num_round,
              'min_child_weight' : gsearch1.best_params_['min_child_weight'],
              'max_depth' : gsearch1.best_params_['max_depth'], 
              'scale_pos_weight' : gsearch1.best_params_['scale_pos_weight'], 
              'num_leaves' : gsearch1.best_params_['num_leaves'], 
              'cat_smooth' : gsearch1.best_params_['cat_smooth'],                  
              'max_bin' : gsearch2.best_params_['max_bin'],  
              'min_data_in_leaf' : gsearch2.best_params_['min_data_in_leaf'],  
              'bagging_fraction' : gsearch3.best_params_['bagging_fraction'],
              'feature_fraction' : gsearch3.best_params_['feature_fraction'],
              'bagging_freq' : gsearch3.best_params_['bagging_freq'],
              'lambda_l1' :  gsearch4.best_params_['lambda_l1'],
              'lambda_l2' : gsearch4.best_params_['lambda_l2'],
              'min_split_gain' : gsearch5.best_params_['min_split_gain']
        }
    clf = lgb.train(params, lgbtrain, valid_sets=[lgbval, lgbtrain],
                    valid_names=['val', 'train'], feval=lgb_f1_score,
                    verbose_eval=False,
                    evals_result=evals_result)
    print('eta : ',eta,
          ',nround : ',np.where(np.array(evals_result['val']['f1'])==max(evals_result['val']['f1']))[0][0],
          ',f1 : ',max(evals_result['val']['f1']))
