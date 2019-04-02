
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("x_train_cat.csv")
test = pd.read_csv("x_test_cat.csv")

##########################################################################################################
# model wm(7.06)**
X = df.iloc[:,1:487]
y = df.y4
cat_index = [4,5,7,8,9,10]
x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=.7,random_state=1234)


model4 = CatBoostClassifier(eval_metric='F1',random_seed=42,
                           learning_rate=0.03,
                           depth=5,
                           l2_leaf_reg = 40,
                           iterations=443,
                           bootstrap_type='Bernoulli',
                           subsample= 0.9,
                           scale_pos_weight = sum(y==0)/sum(y==1) )

model4.fit(x_train,y_train,cat_features=cat_index,eval_set=(x_valid,y_valid))

# eta--d-l2-sub-----best------------LB (n)
# 0.02,5,40,0.9,0.8948167 (605)
# 0.04,5,40,0.9,0.8952618 (338)
# 0.03,5,40,0.9,0.8960203 (443) -->7.04 (533)
# 0.03,5,40,0.9,0.8960203 (443) -->7.06 (550)

model4.fit(X,y,cat_features=cat_index)

wm_class =  model4.predict(test.iloc[:,1:487]).astype('int')
sum(wm_class==1)

wm_prob = model4.predict_proba(test.iloc[:,1:487])[:,1]
wm_rank = pd.Series(1-wm_prob).rank()
wm_class = (wm_rank<550).astype('int')
sum(wm_class==1)
##########################################################################################################
##########################################################################################################
# model ln(1.94)
X = df.iloc[:,1:487]
y = df.y3
cat_index = [4,5,7,8,9,10]

x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=.7,random_state=1234)

model3 = CatBoostClassifier(eval_metric='F1',random_seed=42,#use_best_model=True,
                           learning_rate=0.02,
                           depth=5,
                           l2_leaf_reg = 40,
                           iterations=811,
                           bootstrap_type='Bernoulli',
                           subsample= 0.8,
                           scale_pos_weight = sum(y==0)/sum(y==1) )

#model3.fit(x_train,y_train,cat_features=cat_index,eval_set=(x_valid,y_valid))

model3.fit(X,y,cat_features=cat_index)
# eta--d-sub-----best
# 0.03,5,0.8, 0.8093625 (445)
# 0.02,5,0.8, 0.8147297 (811) --> 1.94



ln_class =  model3.predict(test.iloc[:,1:487]).astype('int')
sum(ln_class==1)

ln_prob = model3.predict_proba(test.iloc[:,1:487])[:,1]
ln_rank = pd.Series(1-ln_prob).rank()
ln_class = (ln_rank<190).astype('int')
sum(ln_class==1)




##########################################################################################################
##########################################################################################################
# model cc(1.094225)

df1 = pd.read_csv("x_train_cat_old.csv")
test1 = pd.read_csv("x_test_cat_old.csv")
X = df1.iloc[:,1:487][df1.CUST_START_DT<9998]
y = df1.y1[df1.CUST_START_DT<9998]


x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=.7,random_state=1234)

model1 = CatBoostClassifier(eval_metric='F1',random_seed=42,
                            learning_rate=0.02,
                            depth=5,
                            iterations=584,
                            l2_leaf_reg = 40,
                            bootstrap_type='Bernoulli',
                            subsample= 0.9,
                            scale_pos_weight = sum(y==0)/sum(y==1) )

model1.fit(x_train,y_train,cat_features=cat_index,eval_set=(x_valid,y_valid))
# eta--d-sub--best(old)-------LB(n)
# 0.03,5,0.9,0.6670104 (415)
# 0.02,4,0.9,0.6730795 (696)
# 0.02,5,0.9,0.6740732 (583)-->1.084337(500)


# eta--d-sub--best
# 0.04,6,0.9,0.7088480 (123)
# 0.04,5,0.9,0.7111617 (183) 
# 0.04,4,0.9,0.7115882 (177)
# 0.01,4,0.9,0.7119679 (766)
# 0.02,4,0.9,0.7120393 (493)
model1.fit(X,y,cat_features=cat_index)

sum(test1.AGE<0)


cc_class =  model1.predict(test1.iloc[:,1:487]).astype('int')
sum(cc_class==1)

cc_prob = model1.predict_proba(test1.iloc[:,1:487])[:,1]
cc_rank = pd.Series(1-cc_prob).rank()
cc_class = (cc_rank<500).astype('int')
sum(cc_class[test1.AGE<0]==1)

##########################################################################################################
##########################################################################################################
# model fx(0.485692)
X = df.iloc[:,1:487]
y = df.y2

x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=.7,random_state=1234)

model2 = CatBoostClassifier(eval_metric='F1',random_seed=42,
                           learning_rate=0.03,
                           depth=5,
                           l2_leaf_reg = 40,
                           iterations=769,
                           bootstrap_type='Bernoulli',
                           subsample= 0.9,
                           scale_pos_weight = sum(y==0)/sum(y==1) )

#model2.fit(x_train,y_train,cat_features=cat_index,eval_set=(x_valid,y_valid))
model2.fit(X,y,cat_features=cat_index)

# eta--d-sub--best
# 0.02,5,0.9, 0.8434389 (434)
# 0.05,5,0.9, 0.8484024 (440)
# 0.03,5,0.9, 0.8487022 (769)


fx_class =  model2.predict(test.iloc[:,1:487]).astype('int')
sum(fx_class==1)

fx_prob = model2.predict_proba(test.iloc[:,1:487])[:,1]
fx_rank = pd.Series(1-fx_prob).rank()
fx_class = (fx_rank<2200).astype('int')
sum(fx_class==1)








##########################################################################################################
##########################################################################################################



submit = pd.read_csv("TBN_Y_ZERO.csv")
submit.CC_IND = cc_class
submit.FX_IND = fx_class
submit.LN_IND = ln_class
submit.WM_IND = wm_class





submit.to_csv("cc_500_old.csv",index=False)




##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

print(sum(y_train==0)/sum(y_train==1))
print(sum(y_valid==0)/sum(y_valid==1))




QW = pd.DataFrame({"old":df1.CUST_START_DT,"new":df.CUST_START_DT,"dif":df.CUST_START_DT-df1.CUST_START_DT})



