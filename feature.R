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
### 客戶基本資料(無瀏覽資料) ###
cust_data = left_join(dat6,dat2) 

# 顧客累計瀏覽網頁數
cust_data = left_join(cust_data,dat3 %>% group_by(CUST_NO) %>% summarise("web_n1" = n()))

# 顧客累計瀏覽獨立網頁數
cust_data = left_join(cust_data,dat3 %>% group_by(CUST_NO) %>% unique() %>% summarise("web_n2" = n()))

# 利用交易資料補 CUST_START_DT
mis_data = full_join(dat1,dat4[,1:2])
mis_data = full_join(mis_data,dat5[,1:2])
mis_data = full_join(mis_data,dat7[,1:2])
mis_data = full_join(mis_data,dat6) 
mis_data = full_join(mis_data,dat2[,c(1,4)]) %>% as.tibble()
mis_data = distinct(mis_data)
mis_data[is.na(mis_data)] = 999999
mis_data = mis_data %>% group_by(CUST_NO) %>% 
  summarise("mis_time" = min(CC_DT,FX_DT,LN_DT,WM_DT,
                             CC_RECENT_DT,FX_RECENT_DT,LN_RECENT_DT,
                             WM_RECENT_DT,CUST_START_DT))
mis_data = left_join(cust_data[,c(1,8)],mis_data)

cust_data$CUST_START_DT[is.na(cust_data$CUST_START_DT)] = mis_data$mis_time[is.na(cust_data$CUST_START_DT)]
rm(mis_data)

cust_data$CUST_START_DT[cust_data$CUST_START_DT>99999] = NA

# 顧客每天造訪網頁頻率(銀行往來日) &  顧客每天造訪獨立網頁頻率(銀行往來日)
cust_data = cust_data %>% mutate("web_f1" = web_n1/(9597-CUST_START_DT),"web_f2" = web_n2/(9597-CUST_START_DT)  )  

cust_data = left_join(cust_data,dat3 %>% group_by(CUST_NO) %>% summarise("first_web" = min(VISITDATE)))
# 顧客每天造訪網頁頻率(瀏覽初始日) & 顧客每天造訪獨立網頁頻率(瀏覽初始日)
cust_data = cust_data %>% mutate("web_f3" = web_n1/(9597-first_web),"web_f4" = web_n2/(9597-first_web)  )  

# 增加dat6 index排序
cust_data$s_index = seq(1,nrow(cust_data))
cust_data$GENDER_CODE = as.numeric(cust_data$GENDER_CODE)

# 補遺失值
colSums(is.na(cust_data))[colSums(is.na(cust_data))>0]
cust_data$CC_RECENT_DT[is.na(cust_data$CC_RECENT_DT)] = 9999
cust_data$FX_RECENT_DT[is.na(cust_data$FX_RECENT_DT)] = 9999
cust_data$LN_RECENT_DT[is.na(cust_data$LN_RECENT_DT)] = 9999
cust_data$WM_RECENT_DT[is.na(cust_data$WM_RECENT_DT)] = 9999
cust_data$CUST_START_DT[is.na(cust_data$CUST_START_DT)] = 9999
cust_data$web_f1[is.na(cust_data$web_f1)] = 9999 
cust_data$web_f2[is.na(cust_data$web_f2)] = 9999
cust_data[is.na(cust_data)] = -999   # 6 個


# 紀錄有基本資料的人
cust_data = left_join(cust_data,dat2 %>% group_by(CUST_NO) %>% summarise("have2"=n()) ,by="CUST_NO")
cust_data[is.na(cust_data)] = 0


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


webdata = function(num){
  # 60-90
  X_web = left_join(dat3,web_t) %>% filter(VISITDATE>(9447+num-30) & VISITDATE<=(9447+num) ) %>% 
    select(CUST_NO,index) %>% group_by(CUST_NO,index) %>% 
    mutate("aa"=1) %>% summarise_all(funs(sum(aa))) %>% spread(index,3) %>% ungroup()
  
  X_web = left_join(dat6[,1:2],X_web) %>% select(-CC_RECENT_DT) %>% as.tibble()
  colnames(X_web)[-1] = paste0("ww_",as.numeric(colnames(X_web)[-1]))
  
  # 計算60-90瀏覽頻率
  web_fr1 = left_join(dat3,web_t) %>% filter(VISITDATE>(9447+num-30) & VISITDATE<=(9447+num) ) %>%
    as.tibble() %>% unique() %>% group_by(CUST_NO) %>% summarise("WWW_f1" = n()/30)
  web_fr2 = left_join(dat3,web_t) %>% filter(VISITDATE>(9447+num-30) & VISITDATE<=(9447+num) ) %>%
    as.tibble()  %>% group_by(CUST_NO) %>% summarise("WWW_f2" = n()/30 )
  
  web_df = left_join(dat3,web_t_) %>% filter(VISITDATE>(9447+num-30) & VISITDATE<=(9447+num) ) %>%
    group_by(CUST_NO) %>% 
    summarise("ccword" = sum(CC)/30,"fxword" = sum(FX)/30,"lnword" = sum(LN)/30,"wmword" = sum(WM)/30)
  
  web_df_old = left_join(dat3,web_t_) %>% filter(VISITDATE<=(9447+num))  %>% group_by(CUST_NO) %>% 
    summarise("ccword_old" = sum(CC)/(9447+num-min(VISITDATE)) ,"fxword_old" = sum(FX)/(9447+num-min(VISITDATE)),
              "lnword_old" = sum(LN)/(9447+num-min(VISITDATE)),"wmword_old" = sum(WM)/(9447+num-min(VISITDATE)))
  web_df[is.na(web_df)] = 0 ; web_df_old[is.na(web_df_old)] = 0
  web_df_fin = left_join(web_df_old,web_df) %>% mutate("CC_ing" = as.numeric(ccword>ccword_old),
                                                       "FX_ing" = as.numeric(fxword>fxword_old),
                                                       "LN_ing" = as.numeric(lnword>lnword_old),
                                                       "WM_ing" = as.numeric(wmword>wmword_old)) %>% 
    select(-c(ccword_old,fxword_old,lnword_old,wmword_old))
  X_web = left_join(X_web,web_fr1);  X_web = left_join(X_web,web_fr2);  X_web = left_join(X_web,web_df_fin)
  X_web[is.na(X_web)] = 0
  return(X_web)
}

# X_train
X_web = webdata(90)


# X_test
X_web120 = webdata(120)

webname = intersect(colnames(X_web),colnames(X_web120))
#################################################################################################
### X ###

# num 填30,60,90,120
# train(1-90);test(120)
CC = function(num){
  num1 = 9447 + num 
  cc = dat1 %>% filter(CC_DT<=num1)
  # 一天重複辦卡數
  cc1 = cc %>% group_by(CUST_NO) %>% summarise("cc_n1" = n()) 
  # 重複辦卡天數
  cc2 = cc %>% unique() %>% group_by(CUST_NO)  %>% summarise("cc_n2" = n()) 
  # 最後一個交易時間間隔
  cc3 = cc %>% group_by(CUST_NO) %>% arrange(CC_DT) %>% 
    summarise("cc_time"= last(diff(c(CC_DT[1],CC_DT) )) ) %>% filter(cc_time>0)
  # 第一次交易時間 + 入行第一次交易時長
  cc4 = left_join(cc %>% group_by(CUST_NO) %>% summarise("first_cc" = first(CC_DT) ),cust_data[,c(1,8)]) %>% 
    mutate("cc_ctime" = first_cc-CUST_START_DT ) %>% select(CUST_NO,cc_ctime)
  cc4[,2][is.na(cc4[,2])] = 0
  # 150天距離最後一筆交易天數 & CC購買頻率
  cc6 = cc %>% group_by(CUST_NO) %>% summarise("last_cc"=num1 - max(CC_DT),"cc_freq" = (max(CC_DT)-min(CC_DT)) / num  )  
  
  # 前2期的y & 前1期的y
  qq1 = left_join(cust_data[,1:2],filter(dat1,CC_DT<=(num1-60) & CC_DT>(num1-90)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq1$a1[is.na(qq1$a1)] = 0
  qq2 = left_join(cust_data[,1:2],filter(dat1,CC_DT<=(num1-30) & CC_DT>(num1-60)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq2$a1[is.na(qq2$a1)] = 0
  
  ## x ##
  x_cc = left_join(cust_data[,1:2],cc1);  x_cc = left_join(x_cc,cc2);  x_cc = left_join(x_cc,cc3)
  x_cc = left_join(x_cc,cc4);x_cc = left_join(x_cc,cc6);  x_cc[is.na(x_cc)] = 0
  x_cc$c_m1 = qq1$a1;x_cc$c_m2 = qq2$a1
  
  ## y ##
  q1 = left_join(cust_data[,1:2],filter(dat1,CC_DT>num1) %>% group_by(CUST_NO) %>% 
                   summarise(a1 = n())) %>% select(CUST_NO,a1) 
  q1$a1[is.na(q1$a1)] = 0;q1$a1[q1$a1>0] = 1
  return(list(x=x_cc ,y=q1$a1,a=sum(q1$a1==0)/sum(q1$a1==1)) )
};cc_90 = CC(90);cc_120 = CC(120)
FX = function(num){
  num1 = 9447 + num 
  fx = dat4 %>% filter(FX_DT<=num1)
  # 重複外匯天數
  fx1 = fx %>% group_by(CUST_NO) %>% summarise("fx_n1" = n())
  # 最後一個交易時間間隔
  fx2 = fx %>% group_by(CUST_NO) %>% arrange(FX_DT) %>% 
    summarise("fx_time"=last( diff(c(FX_DT[1],FX_DT) ) ) ) %>% filter(fx_time>0)
  # 第一次交易時間 + 入行第一次交易時長
  fx3 = left_join(fx %>% group_by(CUST_NO) %>% summarise("first_fx" = first(FX_DT) ),cust_data[,c(1,8)]) %>% 
    mutate("fx_ctime" = first_fx-CUST_START_DT ) %>% select(CUST_NO,fx_ctime)
  fx3[,2][is.na(fx3[,2])] = 0
  # 交易金額累計
  fx4 = fx %>% group_by(CUST_NO) %>% summarise("fx_money" = sum(FX_TXN_AMT)) 
  
  # 150天距離最後一筆交易天數 & fx購買頻率
  fx6 = fx %>% group_by(CUST_NO) %>% summarise("last_fx"=num1 -max(FX_DT),"fx_freq" = (max(FX_DT)-min(FX_DT))/num )
  
  # 前2期的y & 前1期的y
  qq1 = left_join(cust_data[,1:2],filter(dat4,FX_DT<=(num1-60) & FX_DT>(num1-90)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq1$a1[is.na(qq1$a1)] = 0
  qq2 = left_join(cust_data[,1:2],filter(dat4,FX_DT<=(num1-30) & FX_DT>(num1-60)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq2$a1[is.na(qq2$a1)] = 0
  
  ## x ##
  x_fx = left_join(cust_data,fx1);  x_fx = left_join(x_fx,fx2);  x_fx = left_join(x_fx,fx3)
  x_fx = left_join(x_fx,fx4); x_fx = left_join(x_fx,fx6)   ;x_fx[is.na(x_fx)] = 0
  x_fx$f_m1 = qq1$a1;x_fx$f_m2 = qq2$a1
  
  ## y ##
  q2 = left_join(cust_data[,1:2],filter(dat4,FX_DT>num1) %>% group_by(CUST_NO) %>% 
                   summarise(a1 = n())) %>% select(CUST_NO,a1) 
  q2$a1[is.na(q2$a1)] = 0
  q2$a1[q2$a1>0] = 1
  
  return(list(x=x_fx ,y=q2$a1,a=sum(q2$a1==0)/sum(q2$a1==1)) )
};fx_90 = FX(90);fx_120 = FX(120)
LN = function(num){
  num1 = 9447 + num 
  ln = dat5 %>% filter(LN_DT<=num1)
  # 重複信貸天數
  ln1 = ln %>% group_by(CUST_NO) %>% summarise("ln_n1" = n()) 
  # 最後一個交易時間間隔
  ln2 = ln  %>% group_by(CUST_NO) %>% arrange(LN_DT) %>% 
    summarise("ln_time"=last( diff(c(LN_DT[1],LN_DT) ) ) ) %>% filter(ln_time>0)
  # 第一次交易時間 + 入行第一次交易時長
  ln3 = left_join(ln %>% group_by(CUST_NO) %>% summarise("first_ln" = first(LN_DT) ),cust_data[,c(1,8)]) %>% 
    mutate("ln_ctime" = first_ln-CUST_START_DT ) %>% select(CUST_NO,ln_ctime)
  ln3[,2][is.na(ln3[,2])] = 0
  # 交易金額累計
  ln4 = ln %>% group_by(CUST_NO) %>% summarise("ln_money" = sum(LN_AMT)) 
  
  # 申貸用途類別統計
  ln5 = spread(ln,LN_USE,LN_AMT);  ln5[is.na(ln5)] = 0
  colnames(ln5)[3:23] = paste0("LN_",seq(1,21))
  ln5 = ln5 %>% group_by(CUST_NO) %>% 
    summarise(sum(LN_1),sum(LN_2),sum(LN_3),sum(LN_4),sum(LN_5),sum(LN_6),sum(LN_7),sum(LN_8),
              sum(LN_9),sum(LN_10),sum(LN_11),sum(LN_12),sum(LN_13),sum(LN_14),sum(LN_15),
              sum(LN_16), sum(LN_17),sum(LN_18),sum(LN_19),sum(LN_20),sum(LN_21))
  colnames(ln5)[2:22] = paste0("LN_mon_",seq(1,21))
  
  # num前 最後交易紀錄
  ln6 = ln %>% group_by(CUST_NO) %>% summarise("ln_his" = last(LN_DT))
  # 150天距離最後一筆交易天數 & ln購買頻率
  ln7 = ln %>% group_by(CUST_NO) %>% summarise("last_ln"=num1-max(LN_DT),"ln_freq"= (max(LN_DT)-min(LN_DT))/num )
  
  # 前2期的y & 前1期的y
  qq1 = left_join(cust_data[,1:2],filter(dat5,LN_DT<=(num1-60) & LN_DT>(num1-90)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq1$a1[is.na(qq1$a1)] = 0
  qq2 = left_join(cust_data[,1:2],filter(dat5,LN_DT<=(num1-30) & LN_DT>(num1-60)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq2$a1[is.na(qq2$a1)] = 0
  
  ## x ##
  x_ln = left_join(cust_data,ln1);  x_ln = left_join(x_ln,ln2);  x_ln = left_join(x_ln,ln3)
  x_ln = left_join(x_ln,ln4);x_ln = left_join(x_ln,ln5);x_ln = left_join(x_ln,ln6);x_ln = left_join(x_ln,ln7)
  x_ln[is.na(x_ln)] = 0 ;   x_ln$l_m1 = qq1$a1;x_ln$l_m2 = qq2$a1
  ## y ##
  q3 = left_join(cust_data[,1:2],filter(dat5,LN_DT>num1) %>% group_by(CUST_NO) %>% 
                   summarise(a1 = n())) %>% select(CUST_NO,a1) 
  q3$a1[is.na(q3$a1)] = 0;  q3$a1[q3$a1>0] = 1
  return(list(x=x_ln ,y=q3$a1,a=sum(q3$a1==0)/sum(q3$a1==1)) )
};ln_90 = LN(90);ln_120 = LN(120)
WM = function(num){
  num1 = 9447 + num ;  wm = dat7 %>% filter(WM_DT<=num1)
  # 一天重複信託數
  wm1 = wm %>% group_by(CUST_NO) %>% summarise("wm_n1" = n()) 
  # 重複信託天數
  wm2 = wm %>% unique() %>% group_by(CUST_NO)  %>% summarise("wm_n2" = n())
  # 最後一個交易時間間隔
  wm3 = wm %>% group_by(CUST_NO) %>% arrange(WM_DT) %>% 
    summarise("wm_time"=last( diff( c(WM_DT[1],WM_DT) ) ) ) %>% filter(wm_time>0)
  # 第一次交易時間 + 入行第一次交易時長
  wm4 = left_join(wm %>% group_by(CUST_NO) %>% summarise("first_wm" = first(WM_DT) ),cust_data[,c(1,8)])
  wm4 = wm4 %>% mutate("wm_ctime" = first_wm-CUST_START_DT ) 
  wm4 = wm4[,c(1,4)]
  wm4[,2][is.na(wm4[,2])] = 0
  
  # 交易金額累計
  wm5 = wm %>% group_by(CUST_NO) %>% summarise("wm_money" = sum(WM_TXN_AMT)) 
  wm_ = wm
  wm_$index = seq(1:nrow(wm))
  # num前 最後交易紀錄
  wm6 = wm %>% group_by(CUST_NO) %>% summarise("wm_his" = last(WM_DT) )
  # 風險類別統計
  wm7 = spread(wm_,CUST_RISK_CODE,WM_TXN_AMT)
  colnames(wm7)[5:9] = paste0("RISK_",seq(1,5))
  
  wm7 = wm7 %>% group_by(CUST_NO) %>% 
    summarise("R1"=sum(RISK_1),"R2"=sum(RISK_2),"R3"=sum(RISK_3),"R4"=sum(RISK_4),"R5"=sum(RISK_5))
  wm7[is.na(wm7)] = 0
  
  # 信託類別統計
  wm8 = spread(wm_,INVEST_TYPE_CODE,WM_TXN_AMT)
  colnames(wm8)[5:6] = paste0("INVEST_",seq(1,2))
  wm8 = wm8 %>% group_by(CUST_NO) %>% 
    summarise("I1"=sum(INVEST_1),"I2"=sum(INVEST_2))
  wm8[is.na(wm8)] = 0
  # 150天距離最後一筆交易天數 & 購買WM頻率
  wm9 = wm %>% group_by(CUST_NO) %>% summarise("last_wm"= num1 - max(WM_DT),"wm_freq"=(max(WM_DT)-min(WM_DT))/num )
  
  # 前2期的y & 前1期的y
  qq1 = left_join(cust_data[,1:2],filter(dat7,WM_DT<=(num1-60) & WM_DT>(num1-90)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq1$a1[is.na(qq1$a1)] = 0
  qq2 = left_join(cust_data[,1:2],filter(dat7,WM_DT<=(num1-30) & WM_DT>(num1-60)) %>% group_by(CUST_NO) %>% 
                    summarise(a1 = n())) %>% select(CUST_NO,a1) 
  qq2$a1[is.na(qq2$a1)] = 0
  ## x ##
  x_wm = left_join(cust_data,wm1);  x_wm = left_join(x_wm,wm2);x_wm = left_join(x_wm,wm3) 
  x_wm = left_join(x_wm,wm4);  x_wm = left_join(x_wm,wm5);  x_wm = left_join(x_wm,wm6)
  x_wm = left_join(x_wm,wm7);  x_wm = left_join(x_wm,wm8);  x_wm = left_join(x_wm,wm9);  x_wm[is.na(x_wm)] = 0
  x_wm$w_m1 = qq1$a1;x_wm$w_m2 = qq2$a1
  ## y ##
  q4 = left_join(cust_data[,1:2],filter(dat7,WM_DT>num1) %>% group_by(CUST_NO) %>% 
                   summarise(a1 = n())) %>% select(CUST_NO,a1) 
  q4$a1[is.na(q4$a1)] = 0;  q4$a1[q4$a1>0] = 1
  return(list(x=x_wm ,y=q4$a1,a=sum(q4$a1==0)/sum(q4$a1==1)) )
};wm_90 = WM(90);wm_120 = WM(120)
##########################################################################################
# model

# X_train
XX_train = left_join(cc_90$x,fx_90$x )
XX_train = left_join(XX_train,ln_90$x)
XX_train = left_join(XX_train,wm_90$x)
#XX_train = left_join(XX_train,X_web[,c(1,2,5,6,8,10,11,15,32,80,93,111,497:504)] )
XX_train = left_join(XX_train,X_web[,webname] )
XX_train = left_join(cust_data,XX_train)


# X_test 
XX_test = left_join(cc_120$x,fx_120$x)
XX_test = left_join(XX_test ,ln_120$x)
XX_test = left_join(XX_test ,wm_120$x)
#XX_test = left_join(XX_test,X_web120[,c(1,2,5,6,8,10,11,15,31,76,89,107,478:485)])
XX_test = left_join(XX_test,X_web120[,webname])
XX_test = left_join(cust_data,XX_test)

#rm(cc_120,fx_120,ln_120,wm_120)

QQ = which(cust_data$CUST_START_DT<9998 | cust_data$CUST_NO %in% df$CUST_NO )

XX_test = left_join(df[,1:2],XX_test) %>% select(-CC_IND) 


XX_train$CUST_START_DT[XX_train$CUST_START_DT>(9447+90)] = 9999



colSums(is.na(XX_train))[colSums(is.na(XX_train))>0]
colSums(is.na(XX_test))[colSums(is.na(XX_test))>0]
######################################################################################################
write.csv(data.frame(XX_train,y1=cc_90$y,y2=fx_90$y,y3=ln_90$y,y4=wm_90$y),file = "x_train_cat.csv",row.names = F)
write.csv(XX_test,file = "x_test_cat.csv",row.names = F)
