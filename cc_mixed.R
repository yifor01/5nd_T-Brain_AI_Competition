library(tidyverse);library(lubridate);library(xgboost);library(dummies)

# 資料讀取
dat1 = read.csv("TBN_CC_APPLY.csv");dat2 = read.csv("TBN_CIF.csv")
dat3 = read.csv("TBN_CUST_BEHAVIOR.csv");dat4 = read.csv("TBN_FX_TXN.csv")
dat5 = read.csv("TBN_LN_APPLY.csv");dat6 = read.csv("TBN_RECENT_DT.csv")
dat7 = read.csv("TBN_WM_TXN.csv");df = read.csv("TBN_Y_ZERO.csv")
colnames(dat1)[2] = "CC_DT";colnames(dat4)[2] = "FX_DT"
colnames(dat5)[2] = "LN_DT";colnames(dat7)[2] = "WM_DT"

# 資料處理
dat1$CUST_NO = as.character(dat1$CUST_NO);dat2$CUST_NO = as.character(dat2$CUST_NO)
dat3$CUST_NO = as.character(dat3$CUST_NO);dat4$CUST_NO = as.character(dat4$CUST_NO)
dat5$CUST_NO = as.character(dat5$CUST_NO);dat6$CUST_NO = as.character(dat6$CUST_NO)
dat7$CUST_NO = as.character(dat7$CUST_NO);df$CUST_NO = as.character(df$CUST_NO)
dat3$PAGE = as.character(dat3$PAGE)

### 網站處理(dat3) ###
dat3$PAGE = sub("http://www.esunbank.com.tw/" ,replacement="",dat3$PAGE) 
dat3$PAGE = sub("https://www.esunbank.com.tw/" ,replacement="",dat3$PAGE) 
a1 = unique(dat3$PAGE)

# 建立網址轉換矩陣
web_t = data.frame("PAGE" = a1 ,"index" = seq(1,length(a1) ))
web_t$PAGE = as.character(web_t$PAGE)
web_t_ = web_t
web_t_ = web_t_ %>% mutate("LN1"=as.numeric(grepl("qodr",PAGE)),
                           "CC1"=as.numeric(grepl("cugfkt",PAGE)),
                           "CC2"=as.numeric(grepl("krtuo",PAGE)),
                           "WM1"=as.numeric(grepl("deoxt",PAGE)),
                           "WM2"=as.numeric(grepl("eorf",PAGE)),
                           "WM3"=as.numeric(grepl("wgdqth",PAGE)),
                           "WM4"=as.numeric(grepl("gpdpgu",PAGE)),
                           "WM5"=as.numeric(grepl("tudfg",PAGE)),
                           "FX1"=as.numeric(grepl("udtg",PAGE)),
                           "FX2"=as.numeric(grepl("fgposkt",PAGE)),
                           "FX3"=as.numeric(grepl("iougkjr",PAGE))) %>% 
  mutate("CC"=as.numeric(CC1+CC2>0),"LN"= as.numeric(LN1>0), "FX"=as.numeric(FX1+FX2+FX3>0),
         "WM"= as.numeric(WM1+WM2+WM3+WM4+WM5>0)) %>% select(PAGE,CC,FX,LN,WM) %>% as.tibble()

# cc

dat3 # 網頁
dat2 # 基本
dat1 # 1-120 cc
dat6 # 1以前 cc

cc_cust = union(unique(dat1$CUST_NO) ,
                unique(filter(dat6[,c(1,2)],!is.na(CC_RECENT_DT))$CUST_NO) )


add1 = filter(dat6[,c(1,2)], CUST_NO %in% cc_cust & !is.na(CC_RECENT_DT))
colnames(add1)[2] = "CC_DT"


left_join(dat1,add1) %>% as.tibble() %>% arrange(CUST_NO,CC_DT) %>% group_by(CUST_NO) %>% 
  summarise(n=n()) %>% arrange(desc(n)) %>% select(n) %>% table

##################################################################################

df_cc = left_join(dat1,add1) %>% as.tibble() %>% filter(CC_DT<=9447+120)

df_cc = left_join(df_cc,dat2) %>% arrange(CUST_NO,CC_DT) %>% select(-c(CUST_START_DT)) 


df_cc = df_cc %>%  group_by(CUST_NO) %>% arrange(CC_DT) %>% 
  mutate(Diff = CC_DT - lag(CC_DT)) %>% ungroup() %>% 
  arrange(CUST_NO,Diff) %>% filter(!is.na(Diff))


df_cc = df_cc %>% mutate("ex_DT"=CC_DT-Diff) %>% select(-Diff)

df_cc = df_cc[,c(1,9,2:8)]

web = left_join(dat3,web_t_) %>% as.tibble() %>% select(-PAGE)

# web_n1
q1 = left_join(df_cc[,1:3],web) %>% 
  filter(VISITDATE>=ex_DT & VISITDATE<=CC_DT) %>%
  group_by(CUST_NO,ex_DT,CC_DT) %>% summarise("web_n1"=n())
# web_n2
q2 = left_join(df_cc[,1:3],distinct(web)) %>% 
  filter(VISITDATE>=ex_DT & VISITDATE<=CC_DT) %>%
  group_by(CUST_NO,ex_DT,CC_DT) %>% summarise("web_n2"=n())

# wm_web_n1
q3 = left_join(df_cc[,1:3],web) %>% 
  group_by(CUST_NO,ex_DT,CC_DT) %>% 
  summarise("CC_web1"=sum(CC),
            "FX_web1"=sum(FX),
            "LN_web1"=sum(LN),
            "WM_web1"=sum(WM)) 
# wm_web_n2
q4 = left_join(df_cc[,1:3],distinct(web)) %>% 
  group_by(CUST_NO,ex_DT,CC_DT) %>% 
  summarise("CC_web2"=sum(CC),
            "FX_web2"=sum(FX),
            "LN_web2"=sum(LN),
            "WM_web2"=sum(WM)) 


df_cc = left_join(df_cc,q1)
df_cc = left_join(df_cc,q2)
df_cc = left_join(df_cc,q3)
df_cc = left_join(df_cc,q4)



# NA
colSums(is.na(df_cc))[colSums(is.na(df_cc))>0]
df_cc$GENDER_CODE = as.numeric(df_cc$GENDER_CODE)
df_cc$AGE[is.na(df_cc$AGE)] = -1
df_cc$CHILDREN_CNT[is.na(df_cc$CHILDREN_CNT)] = -1
df_cc$EDU_CODE[is.na(df_cc$EDU_CODE)] = -1
df_cc$GENDER_CODE[is.na(df_cc$GENDER_CODE)] = 1
df_cc$INCOME_RANGE_CODE[is.na(df_cc$INCOME_RANGE_CODE)] = -1
df_cc$WORK_MTHS[is.na(df_cc$WORK_MTHS)] = -1
df_cc[is.na(df_cc)] = 0


xgbcv = xgb.cv(data = as.matrix( df_cc[,-c(1,2,3)] ),
               label = as.numeric(df_cc$CC_DT-df_cc$ex_DT),
               eta=0.01,
               max_depth=5,
               subsample=0.8,
               colsample_bytree=0.8,
               print_every_n = 200,
               nrounds = 500,nfold=5)
print(min(xgbcv$evaluation_log$test_rmse_mean))
print(which.min(xgbcv$evaluation_log$test_rmse_mean))
# 0.01.5.0.8.0.8,481,13.8371





xgb1 = xgboost(data = as.matrix( df_cc[,-c(1,2,3)] ),
               label = as.numeric(df_cc$CC_DT-df_cc$ex_DT),
               eta=0.01,
               max_depth=5,subsample=0.8,colsample_bytree=0.8,
               nrounds =  481)
xgb.importance(model=xgb1)



predict(xgb1,as.matrix( df_cc[,-c(1,2,3)] ))


AAA2 = data.frame("CUST_NO"=df_cc[,1],
             "next1"=df_cc$CC_DT+predict(xgb1,as.matrix(df_cc[,-c(1,2,3)]))) %>% 
  group_by(CUST_NO) %>% summarise("Q1"=min(next1)-9447-90,"Q2"=max(next1)-9447-90) 


AAA4 = left_join(AAA2,dat1 %>% filter(CC_DT>9447+90) %>% group_by(CUST_NO) %>%
                   summarise(n=1)) %>% 
  mutate("ans"=as.numeric( (Q2<60 & Q2>0) ) )
AAA4[is.na(AAA4)] = 0
table("true"=AAA4$n,"pred"=AAA4$ans)




##################################
cc = read.csv("lgb_cc_prob.csv")

cc$CC_IND[cc$CC_IND>0.7] = 0

cc$CC_IND[which(cc$CUST_NO %in% AAA4$CUST_NO[AAA4$ans==1])] = 
  cc$CC_IND[which(cc$CUST_NO %in% AAA4$CUST_NO[AAA4$ans==1])]+0.3


table(cc$CC_IND>0.9)


cc$CC_IND = as.numeric(cc$CC_IND>0.9)



write.csv(cc,file = "cc_ssqw.csv",row.names = F)





