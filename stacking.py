
import warnings
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression



print('Load Data...............')
df = pd.read_csv("x_train_cat.csv")
test = pd.read_csv("x_test_cat.csv")
cat_index = [4,5,7,8,9,10]
#################################################

# CC model 
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y1[df.CUST_START_DT<9998]

lgb_cc = lgb.LGBMClassifier(boosting_type='gbdt',
                            objective='binary',
                            cat_smooth = 3,                  
                            max_depth = 13, 
                            min_child_weight = 3,
                            num_leaves =60, 
                            scale_pos_weight = 13, 
                            max_bin = 125,  
                            min_data_in_leaf = 51,  
                            bagging_fraction = 0.6,
                            bagging_freq = 0,
                            feature_fraction = 0.8,
                            lambda_l1 =  0.001,
                            lambda_l2 = 0.00001,
                            min_split_gain = 0.2,
                            learning_rate=0.01, 
                            n_estimators=365  )
xgb_cc = XGBClassifier(learning_rate = 0.005 , 
                     reg_alpha = 0,
                     n_estimators=458, 
                     max_depth=6,
                     min_child_weight=2, 
                     gamma=0.1, 
                     subsample=.9, 
                     colsample_bytree=.85,
                     objective= 'binary:logistic', nthread=6, 
                     scale_pos_weight=11, 
                     seed=27,n_jobs=5)
cat_cc = CatBoostClassifier(eval_metric='F1',random_seed=42,
                            learning_rate=0.01,
                            depth=9,
                            l2_leaf_reg = 30,
                            iterations=3000,
                            bootstrap_type='Bernoulli',
                            subsample= 0.9,thread_count=6,
                            scale_pos_weight = 15)
lgb_cc.fit(X,y)
xgb_cc.fit(X,y)
cat_cc.fit(X,y,cat_features=cat_index)

cc_1 = lgb_cc.predict_proba(X)[:,1]
cc_2 = xgb_cc.predict_proba(X)[:,1]
cc_3 = cat_cc.predict_proba(X)[:,1]
a1 = lgb_cc.predict_proba(test.iloc[:,1:487])[:,1]
a2 = xgb_cc.predict_proba(test.iloc[:,1:487])[:,1]
a3 = cat_cc.predict_proba(test.iloc[:,1:487])[:,1]

cc_train = pd.DataFrame({'model1':cc_1,'model2':cc_2,'model3':cc_3})
cc_test = pd.DataFrame({'model1':a1,'model2':a2,'model3':a3})

logistic_regression = LogisticRegression()
logistic_regression.fit(cc_train,y)
cc_pred = logistic_regression.predict_proba(cc_test)[:,1]
cc_class = logistic_regression.predict(cc_test)

### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.CC_IND = cc_class
submit.to_csv("cc_fin.csv",index=False)

#################################################################################
# FX model 
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y2[df.CUST_START_DT<9998]

lgb_fx = lgb.LGBMClassifier(boosting_type='gbdt',
                            objective='binary',
                            cat_smooth = 4,                  
                            max_depth = 13, 
                            min_child_weight = 2,
                            num_leaves =50, 
                            scale_pos_weight = 7, 
                            max_bin = 95,  
                            min_data_in_leaf = 11,  
                            bagging_fraction = 0.8,
                            bagging_freq = 30,
                            feature_fraction = 0.8,
                            lambda_l1 =  0,
                            lambda_l2 = 0,
                            min_split_gain = 0,
                            learning_rate=0.01, 
                            n_estimators=128  )
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

cat_fx = CatBoostClassifier(eval_metric='F1',random_seed=42,
                            learning_rate=0.02,
                            depth=12,
                            l2_leaf_reg = 30,
                            iterations=3000,
                            bootstrap_type='Bernoulli',
                            subsample= 0.6,thread_count=6,
                            scale_pos_weight = 4)

lgb_fx.fit(X,y)
xgb_fx.fit(X,y)
cat_fx.fit(X,y,cat_features=cat_index)


fx_train = pd.DataFrame({'model1':lgb_fx.predict_proba(X)[:,1],
                         'model2':xgb_fx.predict_proba(X)[:,1],
                         'model3':cat_fx.predict_proba(X)[:,1]})
fx_test = pd.DataFrame({'model1':lgb_fx.predict_proba(test.iloc[:,1:487])[:,1],
                        'model2':xgb_fx.predict_proba(test.iloc[:,1:487])[:,1],
                        'model3':cat_fx.predict_proba(test.iloc[:,1:487])[:,1]})

logistic_regression = LogisticRegression()
logistic_regression.fit(fx_train,y)
fx_pred = logistic_regression.predict_proba(fx_test)[:,1]
fx_class = logistic_regression.predict(fx_test)
print(sum(fx_class==1))


### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.FX_IND = fx_class
submit.to_csv("fx_fin.csv",index=False)

#################################################################################
# LN model 
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y3[df.CUST_START_DT<9998]

lgb_ln = lgb.LGBMClassifier(boosting_type='gbdt',
                            objective='binary',
                            cat_smooth = 3,                  
                            max_depth = 5, 
                            min_child_weight = 1,
                            num_leaves =10, 
                            scale_pos_weight = 10, 
                            max_bin = 205,  
                            min_data_in_leaf = 101,  
                            bagging_fraction = 0.9,
                            bagging_freq = 30,
                            feature_fraction = 0.7,
                            lambda_l1 =  0.001,
                            lambda_l2 = 0.00001,
                            min_split_gain = 0,
                            learning_rate=0.05, 
                            n_estimators=207  )



xgb_ln = XGBClassifier(learning_rate = 0.005 , 
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

cat_ln = CatBoostClassifier(eval_metric='F1',random_seed=42,
                            learning_rate=0.04,
                            depth=6,
                            l2_leaf_reg = 40,
                            iterations=3000,
                            bootstrap_type='Bernoulli',
                            subsample= 1,thread_count=6,
                            scale_pos_weight = 20)
lgb_ln.fit(X,y)
xgb_ln.fit(X,y)
cat_ln.fit(X,y,cat_features=cat_index)

ln_1 = lgb_ln.predict_proba(X)[:,1]
ln_2 = xgb_ln.predict_proba(X)[:,1]
ln_3 = cat_ln.predict_proba(X)[:,1]
b1 = lgb_ln.predict_proba(test.iloc[:,1:487])[:,1]
b2 = xgb_ln.predict_proba(test.iloc[:,1:487])[:,1]
b3 = cat_ln.predict_proba(test.iloc[:,1:487])[:,1]

ln_train = pd.DataFrame({'model1':ln_1,'model2':ln_2,'model3':ln_3})
ln_test = pd.DataFrame({'model1':b1,'model2':b2,'model3':b3})

logistic_regression = LogisticRegression()
logistic_regression.fit(ln_train,y)
ln_pred = logistic_regression.predict_proba(ln_test)[:,1]
ln_class = logistic_regression.predict(ln_test)
print(sum(ln_class==1))


### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.LN_IND = ln_class
submit.to_csv("ln_fin.csv",index=False)

#################################################################################

# WM model 
X = df.iloc[:,1:487][df.CUST_START_DT<9998]
y = df.y4[df.CUST_START_DT<9998]

lgb_wm = lgb.LGBMClassifier(boosting_type='gbdt',
                            objective='binary',
                            cat_smooth = 3,                  
                            max_depth = 5, 
                            min_child_weight = 2,
                            num_leaves =20, 
                            scale_pos_weight = 5, 
                            max_bin = 85,  
                            min_data_in_leaf = 1,  
                            bagging_fraction = 0.9,
                            bagging_freq = 30,
                            feature_fraction = 0.9,
                            lambda_l1 =  0.9,
                            lambda_l2 = 0.9,
                            min_split_gain = 0,
                            learning_rate=0.03, 
                            n_estimators=358  )


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

cat_wm = CatBoostClassifier(eval_metric='F1',random_seed=42,
                            learning_rate=0.06,
                            depth=6,
                            l2_leaf_reg = 20,
                            iterations=300,
                            bootstrap_type='Bernoulli',
                            subsample= 0.9,thread_count=6,
                            scale_pos_weight = 5)

lgb_wm.fit(X,y)
xgb_wm.fit(X,y)
cat_wm.fit(X,y,cat_features=cat_index)

wm_train = pd.DataFrame({'model1':lgb_wm.predict_proba(X)[:,1],
                         'model2':xgb_wm.predict_proba(X)[:,1],
                         'model3':cat_wm.predict_proba(X)[:,1]})
wm_test = pd.DataFrame({'model1':lgb_wm.predict_proba(test.iloc[:,1:487])[:,1],
                        'model2':xgb_wm.predict_proba(test.iloc[:,1:487])[:,1],
                        'model3':cat_wm.predict_proba(test.iloc[:,1:487])[:,1]})

logistic_regression = LogisticRegression()
logistic_regression.fit(wm_train,y)
wm_pred = logistic_regression.predict_proba(wm_test)[:,1]
wm_class = logistic_regression.predict(wm_test)
print(sum(wm_class==1))


### SUBMOIT      
submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.WM_IND = wm_pred
submit.to_csv("wm_prob_mix.csv",index=False)

