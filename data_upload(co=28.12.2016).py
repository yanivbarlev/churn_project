'''Files-
trans.pickle - Transaction data
members.pickle - Members Data

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
#cutoff_date = pd.Timestamp('20161126') #Add one day so there are no zeros


####################Uploading members table########################################
members = pd.read_pickle('members.pickle')
#Originally used: members = pd.read_csv('members_v3.csv')
members.sort_values('registration_init_time', inplace=True)
# Convert date & time format
members.registration_init_time= pd.to_datetime(members.registration_init_time, format='%Y%m%d')
#Remove all members that have reg_init_time<28/1/2017 because this is the cutoffpoint
members = members[members.registration_init_time<cutoff_date] #6434594

sns.countplot(members.registration_init_time.apply(lambda x: x.year) )

#members.to_pickle('members.pickle')
#TODO:How to use logfiles?

######################Uploading transactions ######################################
trans = pd.read_pickle('trans.pickle')
#Originally trans = pd.read_csv('transactions.csv')  #21,547,746
trans.head()
trans = trans[trans.msno.isin(members.msno)] #18,869,880
trans.transaction_date= pd.to_datetime(trans.transaction_date, format='%Y%m%d')
trans.membership_expire_date= pd.to_datetime(trans.membership_expire_date, format='%Y%m%d')

#trans.to_pickle('trans.pickle')