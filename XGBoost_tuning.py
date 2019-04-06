#Import libraries:
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


print('Load Data...............')
df = pd.read_csv("x_train_cat.csv")
test = pd.read_csv("x_test_cat.csv")


##########################################################################################################
# FX model 
print('FX model..................')
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y2[df.CUST_START_DT<9998]
x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=.7,random_state=1234)


# step 0: Unblance data
param_test0 = {'scale_pos_weight':[ 0.1, 1,10,50,100,150,200,sum(y==0)/sum(y==1),
                                   sum(y_train==0)/sum(y_train==1),
                                   sum(y==1)/sum(y==0)]}

gsearch0 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=200, 
                                                  max_depth=5,
                                                  min_child_weight=1, gamma=0, 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, 
                                                  scale_pos_weight=1, 
                                                  seed=27,n_jobs=4), 
                        param_grid = param_test0, scoring='f1',iid=False, cv=5,verbose=3)
gsearch0.fit(x_train,y_train)
gsearch0.grid_scores_, gsearch0.best_params_, gsearch0.best_score_



# detail 
c1 = gsearch0.best_params_['scale_pos_weight']
param_test0 = {'scale_pos_weight':list(range(max(c1-20,1),c1+20,5))}
gsearch0 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=200, 
                                                  max_depth=5,
                                                  min_child_weight=1, gamma=0, 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=1, 
                                                  seed=27), 
                        param_grid = param_test0, scoring='f1',iid=False, cv=5,verbose=3)
gsearch0.fit(x_train,y_train)
gsearch0.grid_scores_, gsearch0.best_params_, gsearch0.best_score_



# Step 1: Fix learning rate and number of estimators for tuning tree-based parameters
params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary:logistic',
          'metric': 'auc',
          'learning_rate':0.05,
          'num_leaves':30, 
          'max_depth': 5,
          'n_estimators':1000,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'scale_pos_weight':gsearch0.best_params_['scale_pos_weight']
    }


data_train = xgb.DMatrix(x_train,y_train)
cv_results = xgb.cv(params, data_train, num_boost_round=3000, nfold=5, stratified=False, shuffle=True, 
                    metrics='auc',early_stopping_rounds=100,seed=0,verbose_eval=True)
print('best n_estimators:', len(cv_results['test-auc-mean']))
print('best cv score:', pd.Series(cv_results['test-auc-mean']).max())


# Step 2: Tune max_depth and min_child_weight
param_test1 = {'max_depth':list(range(3,12,2)),
               'min_child_weight':list(range(1,6,2))}

gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=len(cv_results['test-auc-mean']), 
                                                  max_depth=5,
                                                  min_child_weight=1, gamma=0, 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=gsearch0.best_params_['scale_pos_weight'], 
                                                  seed=27), 
                        param_grid = param_test1, scoring='f1',iid=False, cv=5,verbose=3)
gsearch1.fit(x_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# detail
c4 = gsearch1.best_params_['max_depth']
c5 = gsearch1.best_params_['min_child_weight']
param_test2 = {'max_depth':list(range(max(1,c4-2),c4+2,1)),
               'min_child_weight':list(range(max(0,c5-2),c5+2,1))}

gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=len(cv_results['test-auc-mean']), 
                                                  max_depth=5,
                                                  min_child_weight=1, 
                                                  gamma=0, 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=gsearch0.best_params_['scale_pos_weight'], 
                                                  seed=27), 
                        param_grid = param_test2, scoring='f1',iid=False, cv=5,verbose=3)
gsearch2.fit(x_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
print(gsearch2.best_params_)



# Step 3: Tune gamma
param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=len(cv_results['test-auc-mean']), 
                                                  max_depth=gsearch2.best_params_['max_depth'],
                                                  min_child_weight=gsearch2.best_params_['min_child_weight'], 
                                                  gamma=0, 
                                                  subsample=0.8, 
                                                  colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=gsearch0.best_params_['scale_pos_weight'], 
                                                  seed=27), 
                        param_grid = param_test3, scoring='f1',iid=False, cv=5,verbose=3)
gsearch3.fit(x_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
print(gsearch3.best_params_)


# Step 4: Tune subsample and colsample_bytree
param_test4 = {'subsample':[i/10.0 for i in range(6,10)],
               'colsample_bytree':[i/10.0 for i in range(6,10)]}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=len(cv_results['test-auc-mean']), 
                                                  max_depth=gsearch2.best_params_['max_depth'],
                                                  min_child_weight=gsearch2.best_params_['min_child_weight'], 
                                                  gamma=gsearch3.best_params_['gamma'], 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=gsearch0.best_params_['scale_pos_weight'], 
                                                  seed=27), 
                        param_grid = param_test4, scoring='f1',iid=False, cv=5,verbose=3)
gsearch4.fit(x_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# detail 
a1 = gsearch4.best_params_['subsample']*100
a2 = gsearch4.best_params_['colsample_bytree']*100

param_test5 = {'subsample':[i/100.0 for i in range(int(a1-10),int(a1+15),5)],
               'colsample_bytree':[i/100.0 for i in range(int(a2-10),int(a2+15),5)]}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=len(cv_results['test-auc-mean']), 
                                                  max_depth=gsearch2.best_params_['max_depth'],
                                                  min_child_weight=gsearch2.best_params_['min_child_weight'], 
                                                  gamma=gsearch3.best_params_['gamma'], 
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=gsearch0.best_params_['scale_pos_weight'], 
                                                  seed=27), 
                        param_grid = param_test5, scoring='f1',iid=False, cv=5,verbose=3)
gsearch5.fit(x_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, 
print(gsearch5.best_params_,gsearch5.best_score_)

#cStep 5: Tuning Regularization Parameters
param_test6 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=len(cv_results['test-auc-mean']), 
                                                  max_depth=gsearch2.best_params_['max_depth'],
                                                  min_child_weight=gsearch2.best_params_['min_child_weight'], 
                                                  gamma=gsearch3.best_params_['gamma'], 
                                                  subsample=gsearch5.best_params_['subsample'], 
                                                  colsample_bytree=gsearch5.best_params_['colsample_bytree'],
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=gsearch0.best_params_['scale_pos_weight'], 
                                                  seed=27), 
                        param_grid = param_test6, scoring='f1',iid=False, cv=5,verbose=3)
gsearch6.fit(x_train,y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

##################################################################################
# detail
a3 = gsearch6.best_params_['reg_alpha']

param_test7 = {'reg_alpha':[0, a3/10,a3/5,a3,2*a3,5*a3,10*a3]}
gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.05 , 
                                                  n_estimators=len(cv_results['test-auc-mean']), 
                                                  max_depth=gsearch2.best_params_['max_depth'],
                                                  min_child_weight=gsearch2.best_params_['min_child_weight'], 
                                                  gamma=gsearch3.best_params_['gamma'], 
                                                  subsample=gsearch5.best_params_['subsample'], 
                                                  colsample_bytree=gsearch5.best_params_['colsample_bytree'],
                                                  objective= 'binary:logistic', nthread=5, 
                                                  scale_pos_weight=gsearch0.best_params_['scale_pos_weight'], 
                                                  seed=27), 
                        param_grid = param_test7, scoring='f1',iid=False, cv=5,verbose=3)
gsearch6.fit(x_train,y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
print(gsearch6.best_params_)

# Step 6: Reducing Learning Rate (0.05)
params1 = {    
          'boosting_type': 'gbdt',
          'objective': 'binary:logistic',
          'metric': 'auc',
          'learning_rate':0.05,
          'num_leaves':30, 
          'reg_alpha':gsearch6.best_params_['reg_alpha'],
          'max_depth': gsearch2.best_params_['max_depth'],
          'subsample':gsearch5.best_params_['subsample'],
          'colsample_bytree':gsearch5.best_params_['colsample_bytree'],
          'gamma':gsearch3.best_params_['gamma'], 
          'scale_pos_weight':gsearch0.best_params_['scale_pos_weight'],
          'min_child_weight':gsearch2.best_params_['min_child_weight'],
}
print(params1)



params2 = {    
          'boosting_type': 'gbdt',
          'objective': 'binary:logistic',
          'metric': 'auc',
          'learning_rate':0.05,
          'num_leaves':30, 
          'reg_alpha':1,
          'max_depth': 5,
          'subsample':.6,
          'colsample_bytree':.55,
          'gamma':0, 
          'scale_pos_weight':45,
          'min_child_weight':6,
}




from sklearn.metrics import f1_score

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err


best_fx = [0,0,0]
for eta in [0.01,0.02,0.03,0.04,0.05]:
    xgb1 = XGBClassifier(learning_rate = eta , 
                         reg_alpha = 1,
                         n_estimators=2000, 
                         max_depth=5,
                         min_child_weight=6, 
                         gamma=0, 
                         subsample=.6, 
                         colsample_bytree=.55,
                         objective= 'binary:logistic', nthread=5, 
                         scale_pos_weight=45, 
                         seed=27,n_jobs=5)
    eval_set  = [(x_train,y_train), (x_valid,y_valid)]
    xgb1.fit(x_train,y_train,eval_set=eval_set, eval_metric=f1_eval)
    pre_xgb = xgb1.predict(x_valid)

    if best_fx[2] < f1_score(y_valid,pre_xgb):
        best_fx[0] = eta
        best_fx[1] = np.where(np.array(xgb1.evals_result()['validation_1']['f1_err'])==min(xgb1.evals_result()['validation_1']['f1_err']))[0][0]
        best_fx[2] = f1_score(y_valid,pre_xgb)
        print(best_fx)
        print('--'*40)
print(best_fx)



################################

#Import libraries:
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


print('Load Data...............')
df = pd.read_csv("x_train_cat.csv")
test = pd.read_csv("x_test_cat.csv")

# CC
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y1[df.CUST_START_DT<9998]
xgb_cc = XGBClassifier(learning_rate = 0.005 , 
                     reg_alpha = 0,
                     n_estimators=458, 
                     max_depth=6,
                     min_child_weight=2, 
                     gamma=0.1, 
                     subsample=.9, 
                     colsample_bytree=.85,
                     objective= 'binary:logistic', nthread=5, 
                     scale_pos_weight=11, 
                     seed=27,n_jobs=5)
xgb_cc.fit(X,y)
cc_class =  xgb_cc.predict(test.iloc[:,1:487]).astype('int')
print(sum(cc_class)==1)
### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.CC_IND = cc_class
submit.to_csv("xxxxx_cc_691.csv",index=False)

################################
# FX
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y2[df.CUST_START_DT<9998]
xgb_fx = XGBClassifier(learning_rate = 0.02 , 
                     reg_alpha = 0,
                     n_estimators=199, 
                     max_depth=10,
                     min_child_weight=6, 
                     gamma=0, 
                     subsample=.7, 
                     colsample_bytree=.6,
                     objective= 'binary:logistic', nthread=5, 
                     scale_pos_weight=6, 
                     seed=27,n_jobs=5)
xgb_fx.fit(X,y)
fx_class =  xgb_fx.predict(test.iloc[:,1:487]).astype('int')
print(sum(fx_class)==1)
### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.FX_IND = fx_class
submit.to_csv("xxxxx_fx_4222.csv",index=False)

################################
# ln 
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y3[df.CUST_START_DT<9998]

xgb1 = XGBClassifier(learning_rate = 0.005 , 
                     reg_alpha = 1,
                     n_estimators=269, 
                     max_depth=5,
                     min_child_weight=6, 
                     gamma=0, 
                     subsample=.6, 
                     colsample_bytree=.55,
                     objective= 'binary:logistic', nthread=6, 
                     scale_pos_weight=45, 
                     seed=27)
xgb1.fit(X,y)

ln_class =  xgb1.predict(test.iloc[:,1:487]).astype('int')
print(sum(ln_class==1))
### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.LN_IND = ln_class
submit.to_csv("xxxxx_ln_1397.csv",index=False)

################################
# wm
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y4[df.CUST_START_DT<9998]

xgb_wm = XGBClassifier(learning_rate = 0.01 , 
                     reg_alpha = 0.0005,
                     n_estimators=71, 
                     max_depth=5,
                     min_child_weight=1, 
                     gamma=0.1, 
                     subsample=.6, 
                     colsample_bytree=.65,
                     objective= 'binary:logistic', nthread=6, 
                     scale_pos_weight=5, 
                     seed=27)
xgb_wm.fit(X,y)
wm_class =  xgb_wm.predict(test.iloc[:,1:487]).astype('int')
print(sum(wm_class==1))
### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.WM_IND = wm_class
submit.to_csv("xxxxx_wm_806.csv",index=False)
################################



