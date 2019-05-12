import glob
import pandas as pd
import numpy as np
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import sklearn as sk
import time
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', 100)

X= pd.DataFrame()
for i in glob.glob("X2*.pickle"):
    data = pd.read_pickle(i)
    data['index2'] = i[1:]
    data.reset_index(inplace=True)
    X = pd.concat([X,data],axis=0)
X = X.drop(['registered_via_16'], axis=1) #Found in a row. I need to fix the dummy to include

y= pd.DataFrame()
for i in glob.glob("y2*.pickle"):
    data = pd.read_pickle(i)
    data = pd.DataFrame(data) 
    data['index2'] = i[1:]
    y = pd.concat([y,data],axis=0)

y.reset_index(inplace=True)

X.set_index(['msno','index2'], inplace = True)
y.set_index(['msno','index2'], inplace = True)

X = X.drop(['join_date','last_exp_date','last_logged','last_trans_date'],axis=1) #removing date variable

X.sort_index()
y.sort_index()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Sclaing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

################Phase 1 choosing the classifer that will be optimized##################

#Logistic regression
def LR (X_train,y_train,X_test):
    from sklearn.linear_model import LogisticRegression
    lrf = LogisticRegression(C=1e5,n_jobs=-1)
    lrf.fit(X_train,y_train)
    y_test_lrf = lrf.predict(X_test)
    return cm, (cm[0,0]  +cm[1,1]), y_test_lrf,lrf #sensitivity 0.9001

cm,accuracy,y_test_predict,lrclf = LR(X_train,y_train,X_test)

1 - y_test.mean() # Majority 0.8082

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_lrf)
metrics.auc(fpr, tpr)



'''
Logistic regression resutls:
Confusion Matrix
0.778574	0.0296965
0.0871388	0.104591

Accuracy = 0.8831
AUC = 0.754
''''

# Decision Tree
######### Decision Tree ########
def DT(X_train,y_train,X_test):
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn import tree
    tclf = DecisionTreeClassifier()
    tclf.fit(X_train,y_train)
    y_test_tf = tclf.predict(X_test)
    cm = sk.metrics.confusion_matrix(y_test,y_test_tf)/len(y_test)
    return cm, (cm[0,0]  +cm[1,1]),y_test_tf,tclf   #Accuracy 0.8831


cm,accuracy,y_test_predict,dtclf = LR(X_train,y_train,X_test)




from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


tclf = DecisionTreeClassifier()
tclf.fit(X_train,y_train)
y_test_tf = tclf.predict(X_test)
cm = sk.metrics.confusion_matrix(y_test,y_test_tf)/len(y_test)
(cm[0,0]  +cm[1,1]) #Accuracy 0.8831

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_tf)
metrics.auc(fpr, tpr)


'''
Decision Tree Results:

0.733787	0.0744831
0.0689533	0.122776
Accuracy = 0.856
AUC = 0.774
'''

#RAndom Forest
def RF(X_train,y_train,X_test):
    from sklearn.ensemble import RandomForestClassifier
    rfclf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rfclf.fit(X_train, y_train)
    y_test_predict_rf = rfclf.predict(X_test)
    cm = sk.metrics.confusion_matrix(y_test,y_test_predict_rf)/len(y_test)
    return cm, (cm[0,0]  +cm[1,1]), y_test_predict_rf,rfclf #sensitivity 0.9001
    ### Performance was the same wheter the data was standartized or not. Thus we will use non standartized

cm,accuracy,y_test_predict,rfclf = RF(X_train,y_train,X_test)

rf_score= rfclf.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predict_rf)
metrics.auc(fpr, tpr)

 
'''
0.782322	0.0259482
0.0733304	0.118399
Accuracy = 0.9007
AUC = 0.7925
'''


#Using CRoss Validation
scores = cross_val_score(rfclf, X_train, y_train, cv=10,n_jobs=-1,verbose =1)
scores.mean() #(0.9018)
scores.std() #0.000639



#Neural Network
def NN (X_train,y_train,X_test):
    from sklearn.neural_network import MLPClassifier
    nnclf = MLPClassifier(hidden_layer_sizes = [20], solver='lbfgs',
                          random_state = 0).fit(X_train, y_train) 
    y_test_predict_nn = nnclf.predict(X_test)
    cm = sk.metrics.confusion_matrix(y_test,y_test_predict_nn)/len(y_test)
    (cm[0,0]  +cm[1,1]) #sensitivity 0.8985
    return cm, (cm[0,0]  +cm[1,1]), y_test_predict_nn,nnclf

cm,accuracy,y_test_predict,nnclf = RF(X_train,y_train,X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predict_nn)
metrics.auc(fpr, tpr)

scores = cross_val_score(nnclf, X_train, y_train, cv=10,n_jobs=-1,verbose =1)
scores.mean() #(0.902)
scores.std() #0.0013
#Got same results for Standartized data


'''
ANN
Confusion Matrix
0.779662	0.0286083
0.072887	0.118842

Accuracy = 0.8985
AUC = 0.7914
'''

#XGBoost

'''*&&*&*&* important notice: for this to work you have to upload the data without importing 
XGBCLASSIFER - it changes the data type. The import must be done at this stage'''

def XG (X_train,y_train,X_test):
    from xgboost import XGBClassifier
    xgbclf = XGBClassifier(n_jobs=-1)
    xgbclf.fit(X_train,y_train)
    y_test_predict_xgb = xgbclf.predict(X_test)
    cm = sk.metrics.confusion_matrix(y_test,y_test_predict_xgb)/len(y_test)
    (cm[0,0]  +cm[1,1]) #accuracy 0.8956
    return cm,(cm[0,0]  +cm[1,1]),y_test_predict_xgb,xgbclf

cm,accuracy,y_test_predict,xgclf = RF(X_train,y_train,X_test)


#Performance was the same when data was not scaled. so we will continue without scaling the data
#AUC metric
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predict_xgb)
metrics.auc(fpr, tpr)

'''
XGBoost
Confusion Matrix:
0.782097	0.0261739
0.0736689	0.118061

Accuracy 0.900
AUC = 0.7817
'''

#Using CRoss Validation
scores = cross_val_score(xgclf, X_train, y_train, cv=10,n_jobs=-1,verbose =1)
cross_val_score()
scores.mean() #(0.9021)
scores.std() #0.0013


#Results for training set almost 100% accuracy 
y_train_predict_rf = rfclf.predict(X_train)

cm = sk.metrics.confusion_matrix(y_train,y_train_predict_rf)/len(y_train)
(cm[0,0]  +cm[1,1]) #sensitivity 0.9001
rf_score= rfclf.predict_proba(X_test)

'''
0.809777	8.06096e-06
0.00030363	0.189911
Accuracy = 0.9996883094549716
'''






################### Is cross validation necessary?##################

#Corssvalidation shows that there's no need for CV, the dataset is large enough to show low conf_interval









#####################################PHASE 2: Overfitting #######################
#We'll try to run again after removing dimensions with low performance.

#Removing correlated features


def correlated_features(df, p):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > p)]
    # df = df.drop(to_drop, axis=1)
    return to_drop

correlated_features(X,.95)
'['tot_paid',
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
 'days_since_last_trans']
'
X=X.drop(['tot_paid',
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
 'days_since_last_trans'],axis = 1)

#Reloading the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Sclaing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Still getting overfitted classifier
rfclf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rfclf.fit(X_train, y_train)
y_train_predict_rf = rfclf.predict(X_train)
cm = sk.metrics.confusion_matrix(y_train,y_train_predict_rf)/len(y_train)
(cm[0,0]  +cm[1,1]) #sensitivity 0.9001
rf_score= rfclf.predict_proba(X_test)
'''0.809775	1.07479e-05
0.000370804	0.189844

cm = 0.999672
'''

#Test data result are close to the data before the correlated variable elimination but since results are still better we'll continue without them
y_test_predict_rf = rfclf.predict(X_test)
cm = sk.metrics.confusion_matrix(y_test,y_test_predict_rf)/len(y_test)
(cm[0,0]  +cm[1,1]) #sensitivity 0.89755
'''
0.781839	0.0264318
0.0760147	0.115715

Accuracy = 0.8975535044939744
'''




#Reducing dimensions using PCA
########################PCA################################
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
dfpca = pca.fit(X_train) 
pca_dim_exp = dfpca.explained_variance_
'''
[19.11547053,  5.50112502,  4.72755298,  3.77838923,  3.27332763,
        3.12516187,  2.6520282 ,  2.09407754,  1.95655989,  1.57424242,
        1.46407698,  1.3941713 ,  1.37292412,  1.28255479,  1.26668881,
        1.14787538,  1.03580742,  0.97439399,  0.96857521,  0.91260648,
        0.81427887,  0.80155594,  0.73736556,  0.7301202 ,  0.70963208,
        0.64785928,  0.58173252,  0.51965906,  0.50017822,  0.45875079,
        0.44661864,  0.41093038,  0.3869563 ,  0.35346581,  0.32264099,
        0.29756472,  0.27509272,  0.24220399,  0.21644079,  0.18511961,
        0.16636006,  0.1452397 ,  0.14241723,  0.13750901,  0.13131091,
        0.12375124,  0.09675516,  0.08976093,  0.08197492,  0.07794969]
'''

plt.plot(pca_dim_exp)

#Using Elbow criteria we will stop at 9 components

pca = PCA(n_components=9)
dfpca = pca.fit(X_train) 
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#Running Random Forest again
rfclf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rfclf.fit(X_train_pca, y_train)

y_test_predict_rf = rfclf.predict(X_test_pca)
cm = sk.metrics.confusion_matrix(y_test,y_test_predict_rf)/len(y_test)
(cm[0,0]  +cm[1,1]) #Accuracy = 0.87279

# PCA hurt test results so we'll not use it.

#We'll try to reduce dimensionality using the feature importance attribute
feature_importances2 = pd.DataFrame(rfclf.feature_importances_,index = X.columns,columns=['importance']).sort_values('importance',ascending=False)
plt.cla()
plt.plot(feature_importances.importance)
#We will try to stop at 0.01 as importnace cutoff point
'''num_cancel
max_985
max_50
registered_via_7
times_churn
pmt_method_num
city_A
max_75
pln_uniq
registered_via_9
city_C
gender_male
city_B
gender_female
registered_via_3
min_unq
min_100
registered_via_4
min_25
min_50
min_75
min_985'
'''

#Run this after removing the correlated variables
X = X.drop(['num_cancel','max_985','max_50','registered_via_7','times_churn','pmt_method_num','city_A','max_75',
'pln_uniq','registered_via_9','city_C','gender_male','city_B','gender_female','registered_via_3','min_unq','min_100',
'registered_via_4','min_25','min_50','min_75','min_985'],axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Sclaing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rfclf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rfclf.fit(X_train, y_train)
y_test_predict_rf = rfclf.predict(X_test)
cm = sk.metrics.confusion_matrix(y_test,y_test_predict_rf)/len(y_test)
(cm[0,0]  +cm[1,1]) #sensitivity 0.89702
rf_score= rfclf.predict_proba(X_test)


#We removed 30 variables and wend from Accuracy of 0.900 to 0.897 so the model is much more robust.

#By looking at the variables by importance it looks like we might be able to improve results if we create more features:
#Especially those that relate to the last days activity
'''	importance
days_before_churn	0.05814723060949504
last_pmt_mtd	0.05344452924863981
last_pln_days	0.050312720750242894
cnt_trans	0.04035719930608117
tot_price	0.03603393316822273
days_last_10	0.03449191347101551
plandays_max	0.031662635027396
days_since_last_log	0.030648469363125246
days_last_30	0.02909911879342835
secs_last_10	0.024202427442022466
secs_last_30	0.023164260533495625
days	0.022379527027281364
'''
'''A few ideas:
    
days_before_churn^2
log(days_before_churn)
is_last payment method different then the one before
days_last_10/days_last_30

'''
X['days_before_churn_sqrt'] = X['days_before_churn'].apply(lambda x: np.sqrt(x) if x>0 else 0)
X['days_before_churn_log'] = X['days_before_churn'].apply(lambda x: np.log(x) if x>0 else 0)
X['days_before_churn_cub'] = X['days_before_churn'].apply(lambda x: x*x)

X.fillna(0,inplace = True)

#Results were 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


rfclf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rfclf.fit(X_train, y_train)
y_test_predict_rf = rfclf.predict(X_test)
cm = sk.metrics.confusion_matrix(y_test,y_test_predict_rf)/len(y_test)
(cm[0,0]  +cm[1,1]) #sensitivity 0.89702



#Second attempt - Create a PCA variable based on all the variables we dropped due to low importance

#Reload the data

temp_X_test = X_test[['num_cancel','max_985','max_50','registered_via_7','times_churn','pmt_method_num','city_A','max_75',
'pln_uniq','registered_via_9','city_C','gender_male','city_B','gender_female','registered_via_3','min_unq','min_100',
'registered_via_4','min_25','min_50','min_75','min_985']]

temp_X_train =X_train[['num_cancel','max_985','max_50','registered_via_7','times_churn','pmt_method_num','city_A','max_75',
'pln_uniq','registered_via_9','city_C','gender_male','city_B','gender_female','registered_via_3','min_unq','min_100',
'registered_via_4','min_25','min_50','min_75','min_985']]
 


X_test = X_test.drop(['num_cancel','max_985','max_50','registered_via_7','times_churn','pmt_method_num','city_A','max_75',
'pln_uniq','registered_via_9','city_C','gender_male','city_B','gender_female','registered_via_3','min_unq','min_100',
'registered_via_4','min_25','min_50','min_75','min_985'],axis=1)

X_train = X_train.drop(['num_cancel','max_985','max_50','registered_via_7','times_churn','pmt_method_num','city_A','max_75',
'pln_uniq','registered_via_9','city_C','gender_male','city_B','gender_female','registered_via_3','min_unq','min_100',
'registered_via_4','min_25','min_50','min_75','min_985'],axis=1)


pca = PCA(n_components=3)
dfpca = pca.fit(temp_X_train) 
X_train_pca = pca.transform(temp_X_train)
X_test_pca = pca.transform(temp_X_test)

X_train['pca'] = X_train_pca
X_test['pca'] = X_test_pca

rfclf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rfclf.fit(X_train, y_train)
y_test_predict_rf = rfclf.predict(X_test)
cm = sk.metrics.confusion_matrix(y_test,y_test_predict_rf)/len(y_test)
(cm[0,0]  +cm[1,1]) #Accuracy  = 0.9002


### Try to remove more variables  -cutoff point >0.015

X = X[['days_before_churn','last_pmt_mtd','last_pln_days','cnt_trans','tot_price',
'days_last_10','plandays_max','days_since_last_log','days_last_30','secs_last_10',
'secs_last_30','days','last_is_cancel','sum_50','sum_unq','days_first_last_trans','last_auto_renew',
'days_logged','can_ratio','sum_100','sum_75']]

X_test = pd.DataFrame(X_test)

temp_X_test = X_test.drop(['days_before_churn','last_pmt_mtd','last_pln_days','cnt_trans','tot_price',
'days_last_10','plandays_max','days_since_last_log','days_last_30','secs_last_10',
'secs_last_30','days','last_is_cancel','sum_50','sum_unq','days_first_last_trans','last_auto_renew',
'days_logged','can_ratio','sum_100','sum_75'],axis=1)
        
temp_X_train = X_train.drop(['days_before_churn','last_pmt_mtd','last_pln_days','cnt_trans','tot_price',
'days_last_10','plandays_max','days_since_last_log','days_last_30','secs_last_10',
'secs_last_30','days','last_is_cancel','sum_50','sum_unq','days_first_last_trans','last_auto_renew',
'days_logged','can_ratio','sum_100','sum_75'],axis=1) 


X_test = X_test[['days_before_churn','last_pmt_mtd','last_pln_days','cnt_trans','tot_price',
'days_last_10','plandays_max','days_since_last_log','days_last_30','secs_last_10',
'secs_last_30','days','last_is_cancel','sum_50','sum_unq','days_first_last_trans','last_auto_renew',
'days_logged','can_ratio','sum_100','sum_75']]

X_train = X_train[['days_before_churn','last_pmt_mtd','last_pln_days','cnt_trans','tot_price',
'days_last_10','plandays_max','days_since_last_log','days_last_30','secs_last_10',
'secs_last_30','days','last_is_cancel','sum_50','sum_unq','days_first_last_trans','last_auto_renew',
'days_logged','can_ratio','sum_100','sum_75']]


pca = PCA(n_components=1)
dfpca = pca.fit(temp_X_train) 
X_train_pca = pca.transform(temp_X_train)
X_test_pca = pca.transform(temp_X_test)


X_train['pca'] = X_train_pca
X_test['pca'] = X_test_pca

rfclf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rfclf.fit(X_train, y_train)
y_test_predict_rf = rfclf.predict(X_test)
cm = sk.metrics.confusion_matrix(y_test,y_test_predict_rf)/len(y_test)
(cm[0,0]  +cm[1,1]) #Accuracy  = 0.8968

feature_importances3 = pd.DataFrame(rfclf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
#Down from 89 to 22 variables  with almost the same performance

X['ratio_days_before_churn_and_days'] = X['days_before_churn']/X['days']



# Attempt to create more time dependant variable didn't increase results
X['days_since_last_trans_div_days_since_reg'] = X['days_since_last_trans']/X['days_since_reg']
X['pln_uniq_div_days_since_reg']= X['pln_uniq']/X['days_since_reg']
X['pmt_method_num_div_days_since_reg']= X['pmt_method_num']/X['days_since_reg']
X['tot_paid_div_days_since_reg']= X['tot_paid']/X['days_since_reg']
X['times_churn_div_days_since_reg']= X['times_churn']/X['days_since_reg']
X['sum_seconds_div_days_since_reg']= X['sum_seconds']/X['days_since_reg']
X['days_div_days_since_reg']= X['days']/X['days_since_reg']
X['sum_unq_div_days_since_reg']= X['sum_unq']/X['days_since_reg']




#Over sampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)
X_train = pd.DataFrame(X_train,columns = X.columns)
y_train = pd.DataFrame(y_train,columns = y.columns)



'''
from imblearn.over_sampling import SMOTE
X_train,y_train = SMOTE().fit_resample(X_train,y_train)
'''



#Random Forest

cm,t,y_test_predict_rf,rfclf = RF(X_train,y_train)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predict_rf)
metrics.auc(fpr, tpr)
'''
0.763395	0.0448753
0.0604006	0.131329
Accuracy = 0.8947241143041393
AUC = 0.814 up from 0.79

There's a small decreaase in accuracy from 0.9 to .8947 but the TP went up from 0.11 to 0.13 
thile the FP went up from 0.02 to 0.045
Which means that we can identify more cases of churn.
Since the cost misclassifying FN (identfying people who didn't churn as churners) is much
smaller compared to misclassfying TP we'll absorb the depreciation in Accuracy and use the oversampling
'''

#XGBoost
cm,t,y_test_predict_xg,xgclf = XG(X_train,y_train)
fpr, tpr, thresholds = metrics.r

oc_curve(y_test, y_test_predict_xg)
metrics.auc(fpr, tpr)
'''
0.669163	0.139108
0.0225787	0.169151
Accuracy 0.8383
'''



''' Analyze results '''

y_score = rfclf.predict_proba(X_test)[:,1]

temp = pd.DataFrame()
temp['actual'] = y_test.reset_index().iloc[:,2]
temp['score'] = rfclf.predict_proba(X_test)[:,1]

temp = temp.sort_values('score',ascending=False)
temp = temp.reset_index()
percentile = len(temp)/100

a = 0 # count rows in each percentile
b=0 #sum of churners
c= 0 #pecenntile
d = 0
for i in range (len(temp)):
    if a<percentile:
        a+=1
        b+=temp.actual[i]
    else:
        print('percentile '+str(c) +'% of actual churners ='+ str(b/percentile) + 'Actual Churners '+str(b)
        +'Non Churners:'+str(percentile - b))
        a=0
        b=0
        c+=1
print('percentile '+str(c) +'% of actual churners ='+ str(b/percentile) + 'Actual Churners '+str(b)
        +'Non Churners:'+str(percentile - b))

'''
percentile 0% of actual churners =1.0003627423320303Actual Churners 1241Non Churners:-0.4500000000000455
percentile 1% of actual churners =1.0003627423320303Actual Churners 1241Non Churners:-0.4500000000000455
percentile 2% of actual churners =0.998750554189674Actual Churners 1239Non Churners:1.5499999999999545
percentile 3% of actual churners =0.998750554189674Actual Churners 1239Non Churners:1.5499999999999545
percentile 4% of actual churners =0.9906896134778929Actual Churners 1229Non Churners:11.549999999999955
percentile 5% of actual churners =0.9753738261255089Actual Churners 1210Non Churners:30.549999999999955
percentile 6% of actual churners =0.9124984885736166Actual Churners 1132Non Churners:108.54999999999995
percentile 7% of actual churners =0.8512353391640805Actual Churners 1056Non Churners:184.54999999999995
percentile
''''