'''
files:
trans.pickle - raw transaction file
members - raw
members_trimmed.pickle - memebrs without those who joined after cutoff and whose latest exp date is in range of 30 days before cutoff
trans_trimmed.pickle - not including past cutoffdate data. only members who are in the members_trim table

Featuers-
X.msno - unique user identifier
X.times_churn - number of timese a user has churned
X.join_date  - registration date from members table
X.days - Total Days on plan
x.city - nominal  - dummy city_A,B,C
X.bd - Age of user
X.gender - Gender
X.registrated_via - how the user registered
x.num_cancel - number of cancellation in trans table
X.cnt_trans - number of transactions
X.tot_price - total list price
X.tot_paid - Total actual paid
X.ct_pmt - count of actual payments
X.ct_plan30 - number of plans 0-30 days
X.ct_pln_ovr30 - number of plans over 30
X.cnt_autorenew  -num of autorenew
X.last_trans_weekday - the weekday of the last transaction
X.can_ratio - Cancelations out of total transactions
X.plandays_max/min/mean/std - descriptive statistics of # days of plans
X.pmt_method_num - number of unique payment methods
X. pln_uniq - unique plan days
X.plan_list_price_max/min/mean/std - plan list price statistics
X.actual_amount_paid_max/min/mea/std - Actual amnt paid stats
X.actual_amount_paid_cnt - unique amount paid
X.auto_renew_cnt - num of auto renewls
X.last_pmt_mtd - last used payment method
X.last_pln_days - num of days in last plan
X.last_plan_price  - price of last plan
X.last_paid - last payment
X.last_auto_renew - was last transaction autorenew
X.last_trans_date - date of last trans
X.last_exp_date - Expiration date of last trans
X.last_is_cancel - is last transaction cancellation
X.days_first_last_trans - days from first to last transaction
X.days_since_reg - days since registration
X.days_since_last_trans - days since last transaction
X.days_before_churn - Number of days left before flagged churn
X.days_logged - 
X.last_logged
X.days_since_last_log
X.sum_25/sum_50/sum_75/sum_985/sum_100 - sum songs played till %
X.sum_unq - sum of unique songs
X.sum_seconds - total seconds played
X.mean_25/mean_50/mean_75/mean_985/mean_100 - means songs per day played till %
X.mean_unq - mean unique songs per day
X.mean_seconds - means seconds played per day
X.max_25/max_50/max_75/max_985/max_100 - max songs per day till %
X.max_unq - max of unique songs in a session
X.max_seconds - max seconds played per day
X.min_25/min_50/min_75/min_985/min_100
X.min_unq - min of unique songs
X.min_seconds - min seconds played per day
X.std_25/std_50/std_75/std_985/std_100
X.std_unq - std of unique songs
X.std_seconds - stdseconds played per day
X.days_last_30/days_last_10 - days logged in last 30/10
X.secs_last_30/secs_last_10 - seconds played last 30 /10 days
'''

import pandas as pd
import numpy as np
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import sklearn as sk
import time

pd.set_option('display.max_columns', 100)

cutoff_date = pd.Timestamp('20170128') 
cutoff_date_start  =  cutoff_date - datetime.timedelta(days=31)
cutoff_end_date =  cutoff_date + datetime.timedelta(days=31)



###############Functions###############
def remove_outliers(sr):    #removing outliers of a list using 3 std limit
    return abs(sr- sr.mean() < 3 * sr.std())

def categorizing(df,sr,name): #df, sr - sereis (df.sr), name = name of the new variable
    just_dummy = pd.get_dummies(sr)
    just_dummy= just_dummy.add_prefix(name +'_')
    df = df.join(just_dummy,how='left')
    df = df.drop(name,axis=1)
    return df



                ###############Uploading tables###############
members = pd.read_pickle('members.pickle')
#Originally used: members = pd.read_csv('members_v3.csv')
members.sort_values('registration_init_time', inplace=True)
# Convert date & time format
members.registration_init_time= pd.to_datetime(members.registration_init_time, format='%Y%m%d')
#Remove all members that have reg_init_time<28/1/2017 because this is the cutoffpoint
members = members[members.registration_init_time<cutoff_date] #6434594

sns.countplot(members.registration_init_time.apply(lambda x: x.year) )

#Filter out data past 27/1/207
trans_trimmed = trans[trans.transaction_date<cutoff_date] #remove out of sample data
#Create list of members whose latest exp_date is between 28/12/2016-28/1/2017
trans_trimmed = trans_trimmed.sort_values(by=['msno','transaction_date']) #Sort by msno and transaction_date
last_log = trans_trimmed.groupby(['msno']).last()
panel = pd.DataFrame((last_log[(last_log.membership_expire_date<cutoff_date) & ( last_log.membership_expire_date>=cutoff_date_start)]).index)
trans_trimmed = trans_trimmed[trans_trimmed.msno.isin(panel.msno)]
trans_trimmed.to_pickle('trans_trimmed.pickle')

#Uploading transactions
trans = pd.read_pickle('trans.pickle')
#Originally trans = pd.read_csv('transactions.csv')  #21,547,746
trans.head()
trans = trans[trans.msno.isin(members.msno)] #18,869,880
trans.transaction_date= pd.to_datetime(trans.transaction_date, format='%Y%m%d')
trans.membership_expire_date= pd.to_datetime(trans.membership_expire_date, format='%Y%m%d')


############################### Feture Eng. #########################################

#0. Past Churn indicators
len(set(trans_trimmed['msno'])) #77306 Trnas has LT 2M we will trim the trans 

trans_trimmed = trans_trimmed.sort_values(by=['msno','transaction_date']) #Sort by msno and transaction_date
#Shifting Transaction date forward (sft_regdate  - is shifted expiry date)
trans_trimmed['sft_regdate'] = trans_trimmed.groupby(['msno'])['membership_expire_date'].shift(1) #Check if the difference is 30 days.
trans_trimmed['over30']  =  pd.Series([ trans_trimmed['transaction_date']-(trans_trimmed['sft_regdate'])])[0] #The reason we had an issue was probably because the length was
#differnt because of the NAN or flase
trans_trimmed['daygap'] = trans_trimmed['over30']/np.timedelta64(1, 'D') #convvert to num of days
trans_trimmed['30yn'] = trans_trimmed['daygap']>30  #if Gaps is over 30 it means there was a churn
trans_trimmed = trans_trimmed.drop(['daygap','over30'],axis=1)

# 1. create a table with all users who churned =>have True in 30yn
churn_30 = trans_trimmed[trans_trimmed['30yn']==True].msno #churners who had 31 day gap
churn_30 = pd.DataFrame(churn_30) 
churn_30.drop_duplicates(inplace = True) #81459

#2. Count number of times a user has churned
churn_times = trans_trimmed[trans_trimmed['30yn']==True].groupby(['msno'])['30yn'].count()
churn_times = pd.DataFrame(churn_times)
churn_times.columns = churn_times.columns.str.replace('30yn','times_churn')
churn_times = churn_times.reset_index()

#3. Adding 1 to 'times_churn' for users whose last transaction was before  1/1/2017
# Add last exp_date and num of churns to the members table
temp = trans_trimmed.groupby(['msno']).last()
#boolean 1 if churned - wasn't active for 30 days (whole of Jan)

X = trans_trimmed['msno'].unique()
X = pd.DataFrame(X,columns=['msno'])
X =  pd.merge(X, churn_times, how='left', on=['msno']) #Add 30 days churn to X
#Legacy: X = X.merge(temp['last_date_churn'],how ='left', on ='msno')
X = X.fillna(0)

#4. Calculating sum of actual active days
trans_trimmed.sort_values(['msno','transaction_date'], inplace=True)
trans_2d = trans_trimmed[['msno','transaction_date','membership_expire_date']]
temp = pd.DataFrame()
temp['msno'] = 0 
temp['days'] = 0
temp={}
#for i in range(trans_trimmed.shape[0]):
days =0
i=0

def old_transaction_data(i):
    return trans_2d.iloc[i].msno, trans_2d.iloc[i].transaction_date, trans_2d.iloc[i].membership_expire_date

def new_transaction_date(i):
    return trans_2d.iloc[i].transaction_date, trans_2d.iloc[i].membership_expire_date

while i<len(trans_2d):
    old_msno, old_trans, old_exp = old_transaction_data(i)
    if trans_2d.iloc[i].msno == trans_2d.iloc[i+1].msno: #next line is still the same member
        i+=1
        new_trans, new_exp = new_transaction_date(i)
        if new_trans>old_exp: #old exp was complete
            days+=(old_exp-old_trans).days
            continue
        else: #old was overlapped
            days+= (new_trans - old_trans).days
            continue
    else:   #moved to a new msno
        days += (old_exp-old_trans).days
        temp[old_msno] = days
        #temp.append(pd.DataFrame({"msno":[old_msno],"days":[days]}))
        days = 0
        i+=1

temp = pd.DataFrame.from_dict(temp, orient='index')
temp.reset_index(inplace=True)
temp = temp.rename(index=str, columns={"index": "msno", 0: "days"})
X = X.merge(temp,how='left',on='msno')

#5. Join date
X = pd.merge(X,members_trimmed, how='left',on='msno')
X = X.rename(index=str, columns={"registration_init_time": "join_date"})

#6 Num of cancelletions
X=X.set_index('msno')
X['num_cancel'] = trans_trimmed.groupby(['msno'])['is_cancel'].sum()

#7 Num transactions
X['cnt_trans'] = trans_trimmed.groupby(['msno'])['msno'].count()

#8 Total price
X['tot_price'] = trans_trimmed.groupby(['msno'])['plan_list_price'].sum()
#9 Total Actual Paid
X['tot_paid'] = trans_trimmed.groupby(['msno'])['actual_amount_paid'].sum()

#10 Count paymnents
temp = trans_trimmed[trans_trimmed.actual_amount_paid >0]
X['ct_pmt'] = temp.groupby(['msno'])['actual_amount_paid'].count()

#11 Num plans <=31 days
temp = trans_trimmed[trans_trimmed.actual_amount_paid <=31]
X['ct_pln30'] = temp.groupby(['msno'])['msno'].count()

#12 Num plans >31 days
temp = trans_trimmed[trans_trimmed.actual_amount_paid >31]
X['ct_pln_ovr30'] = temp.groupby(['msno'])['msno'].count()

#13 Num of is_auto_renew transactions
X['cnt_autorenew'] = trans_trimmed.groupby(['msno'])['is_auto_renew'].sum()

#14 Last transaction weekday
temp = trans_trimmed.groupby(['msno'])['transaction_date'].max()
temp = pd.DataFrame(temp)

def weekday (x):
    return datetime.datetime.weekday(x)    
temp.transaction_date = temp.transaction_date.apply(weekday)
X = pd.merge(X,temp,how='left',on='msno')
X.rename(columns={'transaction_date':'last_trans_weekday'}, inplace=True)


#14 % cancellation transactions out of total transactions
X['can_ratio'] = X.num_cancel / X.cnt_trans

#15 Highest + lowest + mean  plan_days
X['plandays_max'] = trans_trimmed.groupby(['msno'])['payment_plan_days'].max()
X['plandays_min'] = trans_trimmed.groupby(['msno'])['payment_plan_days'].min()
X['plandays_mean'] = trans_trimmed.groupby(['msno'])['payment_plan_days'].mean()
X['plandays_std'] = trans_trimmed.groupby(['msno'])['payment_plan_days'].std()

#16 number of unique plan days
X['pln_uniq'] = trans_trimmed.groupby(['msno'])['payment_plan_days'].nunique()

#17  unique pmt methods
X['pmt_method_num'] = trans_trimmed.groupby(['msno'])['payment_method_id'].nunique()

#18 Higheset & lowest & Mean plan list price
X['plan_list_price_max'] = trans_trimmed.groupby(['msno'])['plan_list_price'].max()
X['plan_list_price_min'] = trans_trimmed.groupby(['msno'])['plan_list_price'].min()
X['plan_list_price_mean'] = trans_trimmed.groupby(['msno'])['plan_list_price'].mean()
X['plan_list_price_std'] = trans_trimmed.groupby(['msno'])['plan_list_price'].std()

#19 Higheset & lowest & MEAN actuak paid
X['actual_amount_paid_max'] = trans_trimmed.groupby(['msno'])['actual_amount_paid'].max()
X['actual_amount_paid_min'] = trans_trimmed.groupby(['msno'])['actual_amount_paid'].min()
X['actual_amount_paid_mean'] = trans_trimmed.groupby(['msno'])['actual_amount_paid'].mean()
X['actual_amount_paid_std'] = trans_trimmed.groupby(['msno'])['actual_amount_paid'].std()
X['actual_amount_paid_cnt'] = trans_trimmed.groupby(['msno'])['actual_amount_paid'].count()

#20 number of  auto renew
X['auto_renew_cnt'] = trans_trimmed.groupby(['msno'])['is_auto_renew'].sum()

#Num of days since last transaction
#21 last transaction data
trans_trimmed = trans_trimmed.sort_values(by=['msno','transaction_date']) #Sort by msno and transaction_date
temp = trans_trimmed.groupby(['msno']).last()
temp = temp.drop(['sft_regdate','30yn'], axis=1)
temp.columns = ['last_pmt_mtd','last_pln_days','last_plan_price','last_paid','last_auto_renew', 'last_trans_date','last_exp_date','last_is_cancel']
      
X = X.merge(temp,how='left',on='msno')

#22 time from first to last transaction

for i in X.index:
    end= X.last_trans_date[i]
    start = X.join_date[i]
    X.at[i,'days_first_last_trans'] = (end-start).days

#23 Days since registation
X['days_since_reg'] = (cutoff_date-X['join_date']).dt.days

#23 - Days since last transaction
X['days_since_last_trans'] = (cutoff_date - X.last_trans_date).dt.days
#24 - Days before is_churn
X['days_before_churn'] =31 -  (cutoff_date-X.last_exp_date).dt.days

                        ######################### LOG DATA Upload #############################
''' I uploaded the CSV and saved as pickle'''
log0 = pd.read_pickle('log0.pickle')
log0_trimmed = log0[log0.msno.isin(X.index)]
log1 = pd.read_pickle('log1.pickle')
log1_trimmed = log1[log1.msno.isin(X.index)]
log2 = pd.read_pickle('log2.pickle')
log2_trimmed = log2[log2.msno.isin(X.index)]
log3 = pd.read_pickle('log3.pickle')
log3_trimmed = log3[log3.msno.isin(X.index)]


log_trimmed = pd.concat([log0_trimmed,log1_trimmed,log2_trimmed,log3_trimmed])
log_trimmed.date = pd.to_datetime(log_trimmed.date , format='%Y%m%d')

log_trimmed = log_trimmed[log_trimmed.date<cutoff_date]

#remove negative seconds records
log_trimmed = log_trimmed[log_trimmed.total_secs>0]
#Removing outliers for sums and total seconds
for i in range(2,7):
    log_trimmed  = log_trimmed[remove_outliers(log_trimmed.iloc[:,i])]
log_trimmed  = log_trimmed[remove_outliers(log_trimmed['total_secs'])]

####################### Log Data Features #####################

#Sum of days logged in
log_trimmed.sort_values(by=['msno','date'],inplace = True)
#log_trimmed.to_pickle('log_trimmed.pickle')
X['days_logged'] = log_trimmed.groupby(['msno'])['date'].count()
X['last_logged'] = log_trimmed.groupby(['msno'])['date'].last()
X['days_since_last_log'] = (cutoff_date - X['last_logged']).dt.days
X['last_logged'] = log_trimmed.groupby(['msno'])['date'].last()

X[['sum_25','sum_50','sum_75','sum_985','sum_100','sum_unq','sum_seconds']] = log_trimmed.groupby(['msno'])['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs'].sum()
X[['mean_25','mean_50','mean_75','mean_985','mean_100','mean_unq','mean_seconds']]  =  log_trimmed.groupby(['msno'])['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs'].mean()
X[['max_25','max_50','max_75','max_985','max_100','max_unq','max_seconds']] = log_trimmed.groupby(['msno'])['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs'].max()
X[['min_25','min_50','min_75','min_985','min_100','min_unq','min_seconds']] =log_trimmed.groupby(['msno'])['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs'].min()
X[['std_25','std_50','std_75','std_985','std_100','std_unq','std_seconds']]  = log_trimmed.groupby(['msno'])['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs'].std()

#number of days loggedin past 30 days
temp = log_trimmed[(cutoff_date- log_trimmed.date).dt.days<30]
X['days_last_30'] = temp.groupby(['msno'])['msno'].count()

#number of days loggedin past 10 days
temp = log_trimmed[(cutoff_date- log_trimmed.date).dt.days<10]
X['days_last_10'] = temp.groupby(['msno'])['msno'].count()
#number of seconds played over past 30 days
temp = log_trimmed[(cutoff_date- log_trimmed.date).dt.days<30]
X['secs_last_30'] = temp.groupby(['msno'])['total_secs'].sum()

#seconds played last 10 days
temp = log_trimmed[(cutoff_date- log_trimmed.date).dt.days<10]
X['secs_last_10'] = temp.groupby(['msno'])['total_secs'].sum()

###################Creating y ###################################
###leave only dates after cutoff
trans_y = trans[(trans.transaction_date>=cutoff_date) & (trans.transaction_date<=cutoff_end_date)]
trans_y.sort_values(['msno','transaction_date'],inplace=True)
#trans_y.to_pickle('trans_y.pickle')
y = pd.DataFrame(X.index)
y.set_index('msno',inplace=True)
y['ind'] = trans_y.groupby(['msno'])['transaction_date'].first()
y= y.fillna(0)
y = y.ind.apply(lambda x: 1 if x!=0 else 0)
#y.to_pickle('y.pickle')


#########    cleaning the data    #########################

#City
X.groupby(['city'])['city'].count().sort_values()/len(X)
''' I'll bin using same size bins [1,13&5,rest]  ~33% each
city
19    0.000207
20    0.000841
16    0.001319
7     0.003402
17    0.006338
3     0.007774
21    0.009482
10    0.010348
18    0.010827
11    0.013181
8     0.013337
12    0.019468
14    0.028652
6     0.039078
9     0.049492
22    0.062647
15    0.063462
4     0.072815
5     0.121802
13    0.144232
1     0.321295
'''
X.city = X.city.apply(lambda x: 'A' if x==1 else 'B' if (x==5) or (x==13) else 'C')
#Categorizing
just_dummy = pd.get_dummies(X.city)
just_dummy= just_dummy.add_prefix('city_')
X = X.join(just_dummy,how='left')
X = X.drop('city',axis=1)

#BD
# all ages overf 80 and below 10 will be transformed to 0
len(X.bd[(X.bd>10) & (X.bd<80)]) #50260 are in the range of normal values
X.bd.apply(lambda x: 0 if (x<10) or (x>80) else x)

#Gender
X.gender.isna().sum()/len(X) #35% are nan we'll change to 'other'
X.gender.apply(lambda x: 'unknown' if pd.isnull(x) else x)
X = categorizing(X,X.gender,'gender')

#registered_via
X.groupby(['registered_via'])['registered_via'].count().sort_values()/len(X)
'''
registered_via
13    0.003143
4     0.146948
7     0.168409
3     0.239981
9     0.441518
13 is too small I'll add it to the most frquqent group 9
'''
X['registered_via'] = X['registered_via'].apply(lambda x: 9 if x==13 else x)
X = categorizing(X,X.registered_via,'registered_via')

#remove na
a = X.isna().any().sort_values()
a = a[a==True]
a.index
'''
All the below can be changed to 0
'std_25', 'min_seconds', 'min_unq', 'min_75', 'min_985', 'std_50',
       'min_50', 'min_100', 'std_75', 'std_seconds', 'std_100', 'std_unq',
       'days_last_30', 'days_last_10', 'secs_last_30', 'secs_last_10',
       'min_25', 'days', 'std_985', 'max_seconds', 'max_25', 'max_100',
       'plan_list_price_std', 'plandays_std', 'ct_pln_ovr30', 'ct_pln30',
       'days_logged', 'last_logged', 'days_since_last_log', 'sum_25', 'ct_pmt',
       'sum_75', 'sum_985', 'sum_100', 'sum_unq', 'sum_seconds', 'mean_25',
       'mean_50', 'mean_75', 'mean_985', 'mean_100', 'mean_unq',
       'mean_seconds', 'actual_amount_paid_std', 'max_50', 'max_75', 'max_985',
       'max_unq', 'sum_50']
'''    
X.fillna(0,inplace = True)

#Remove correlated features
def correlated_features(df, p):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > p)]
    # df = df.drop(to_drop, axis=1)
    return to_drop

correlated_features(X,.95)
X = X.drop(['tot_paid',
 'ct_pmt',
 'ct_pln_ovr30',
 'plandays_mean',
 'plan_list_price_max',
 'plan_list_price_min',
 'plan_list_price_mean',
 'plan_list_price_std',
 'actual_amount_paid_max',
 'actual_amount_paid_min',
 'actual_amount_paid_mean',
 'actual_amount_paid_std',
 'actual_amount_paid_cnt',
 'auto_renew_cnt',
 'last_plan_price',
 'last_paid',
 'days_since_reg',
 'days_since_last_trans'])
'


