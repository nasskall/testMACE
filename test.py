import pickle
from datetime import date

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import re

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
train = pd.read_csv('resources/ML_Athens_final.csv')
train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
obj_df = train.select_dtypes('object')
obj_df= obj_df.fillna('2')
obj_df = obj_df.astype(float)
int_df = train.select_dtypes('int64')
float_df = train.select_dtypes('float64')
int_float_df = pd.concat([int_df, float_df], axis=1)
df = pd.concat([int_float_df, obj_df], axis=1)
train.loc[(train['MACE_10years'] == 0) & (train['MACE_Time_10years'] >= 120.0), 'MACE_Time_10years'] = 400
train.loc[((train['MACE_10years'] == 1) & (train['MACE_Time_10years'] < 120.0)), 'MACE_Time_10years'] = 300
train = train.loc[((train['MACE_Time_10years'] == 400) | (train['MACE_Time_10years'] == 300))]
train = train.drop(columns=['MACE_10years'])
train = train.drop(
    columns=['BMI', 'Height', 'Weight', 'LDL', 'HDL', 'TG', 'Left_Atrial_diameter',
             'Hypertrophy_by_ECHO', 'LeftVentricular_Wall_Abnormalities'])
train = train.drop(
    columns=['ECG_Ischemic_abnorm', 'Dyslipidemia', 'ACE_ARB_discharge', 'B_block_diascharge',
             'MI_by_ECG', 'AF_types'])  # Excluded by LassoCV
train = train.drop(
    columns=['Smoking', 'Diouretics_discharge', 'Obesity', 'LeftVentricle_Hypertr_by_ECG',
             'TIA',
             'Alcohol', 'Hemorhag_Transformation', 'Antiplatelets_discharge', 'diabetes_melitus', 'Chol_adm',
             'Urea', 'Cr_adm', 'TOAST', 'ASA_Prior', 'CAD', 'PWML', 'EF', 'CA_Blocker_discharge',
             'Vascular_Imaging_arterial_all'])
train['Anti_Tr_Discharge'] = np.where((train['Cumadin_discharge'] == 0) & (train['NOACs_discharge'] == 0), 0, 1)
train = train.reindex(columns=['Age', 'Sex', 'Hypertension', 'Heart_Failure', 'PAD', 'Vascular_Stenosis_Degree_all', 'eGFR','Statin_prior','Cumadin_Prior','Statin_discharge','Anti_Tr_Discharge' ,'AF','MACE_Time_10years'])
d_var = train.iloc[:, len(train.columns) - 1]
ind_var = train.iloc[:, :len(train.columns) - 1]
ind_var = ind_var.astype('float64')
X_train, X_test, y_train, y_test= train_test_split(ind_var, d_var, test_size=0.1,
                                                                          random_state=22)
cls=XGBClassifier()
cls.fit(X_train,y_train)
predictions= cls.predict(X_test)
# df = pd.read_csv('resources/test.csv')
test_df = pd.read_csv('resources/test1.csv')
temp = re.compile("([a-zA-Z]+)([0-9]+)")
subs = {
    "1": "January",
    "2": "February",
    "3":"March",
    "4":"April",
    "5":"May",
    "6":"June",
    "7":"July",
    "8":"August",
    "9":"September",
    "10":"October",
    "11":"November",
    "12":"December"}
delta =[]
for i in range(test_df.shape[0]):
        dat2 = test_df['LAST_FOLLOW_UP'][i]
        dat1 = test_df['Index_stroke_date'][i]
        dat2 = dat2.split(',')
        dat1 = dat1.split(',')
        for j in range(3):
            dat2[j] = dat2[j].replace(" ", "")
            dat1[j] = dat1[j].replace(" ", "")
        del dat2[0]
        del dat1[0]
        dat1[0:1] = list(temp.match(dat1[0]).groups())
        dat2[0:1] = list(temp.match(dat2[0]).groups())
        rev_subs = {v: k for k, v in subs.items()}
        dat1 = [rev_subs.get(item, item) for item in dat1]
        dat2 = [rev_subs.get(item, item) for item in dat2]
        f_date = date(int(dat2[2]), int(dat2[0]), int(dat2[1]))
        i_date = date(int(dat1[2]), int(dat1[0]), int(dat1[1]))
        delta_c = f_date - i_date
        delta.append(delta_c.days)
test_df['last_follow_up'] = delta
test_df.drop(columns='Index_stroke_date')
test_df[['Hypertension']] = test_df[['Hypertension']].apply(pd.to_numeric)
test_df.loc[((test_df['time_to_mace'].isna()) & (test_df['last_follow_up'] > 680)), 'time_to_mace2'] = 4444444
test_df.loc[((test_df['time_to_mace'].isna()) & (test_df['last_follow_up'] < 1800)), 'time_to_mace'] = 6666666
test_df.loc[((test_df['time_to_mace'].isna()) & (test_df['last_follow_up'] < 680)), 'time_to_mace'] = 5555555
test_df.loc[(test_df['time_to_mace'] < 680), 'time_to_mace2'] = 2222222 #positive for all groups
test_df.loc[(test_df['time_to_mace'] < 680), 'time_to_mace5'] = 2222222
test_df.loc[(test_df['time_to_mace'] > 680) & (test_df['time_to_mace'] < 1800), 'time_to_mace2'] = 4444444 #negative for two years group and positive for five years group
test_df.loc[(test_df['time_to_mace'] > 680) & (test_df['time_to_mace'] < 1800), 'time_to_mace5'] = 2222222
test_df.loc[(test_df['time_to_mace'] > 1800) & (test_df['time_to_mace'] < 3000), 'time_to_mace2'] = 4444444 #negative for all groups
test_df.loc[(test_df['time_to_mace'] > 1800) & (test_df['time_to_mace'] < 3000), 'time_to_mace5'] = 4444444
test_df = test_df.loc[test_df['time_to_mace'] != 5555555]
test_df = test_df.dropna()
test_df=test_df.drop(columns=['Index_stroke_date', 'LAST_FOLLOW_UP','last_follow_up'])
df_af = test_df.loc[test_df['AF'] == 1]
df_af_no = test_df.loc[test_df['AF'] == 0]
df_af_2 = df_af.drop(columns=['time_to_mace','time_to_mace5'])
df_af_no_2 = df_af_no.drop(columns=['time_to_mace','time_to_mace5'])
test_var_af_2 = df_af_2.iloc[:, :len(df_af_2.columns) - 1]
test_var_af_2 = test_var_af_2.astype('float64')
label_var_af_2 = df_af_2.iloc[:, len(df_af_2.columns) - 1]
test_var_2 = df_af_no_2.iloc[:, :len(df_af_no_2.columns) - 1]
label_var_2 = df_af_no_2.iloc[:, len(df_af_no_2.columns) - 1]
test_var_2 = test_var_2.astype('float64')
test_var_2 =test_var_2.drop(columns=['Cumadin_Prior', 'Anti_Tr_Discharge'])

filename5AF = 'resources/xg_model_05.sav'
filename5 = 'resources/xg_model_05out.sav'
filename2AF = 'resources/xg_model_02.sav'
filename2 = 'resources/xg_model_02out.sav'
cls5AF = pickle.load(open(filename5AF, 'rb'))
cls5 = pickle.load(open(filename5, 'rb'))
cls2AF = pickle.load(open(filename2AF, 'rb'))
cls2 = pickle.load(open(filename2, 'rb'))

y_pred1 = cls2AF.predict(test_var_af_2)
y_pred2 = cls2.predict(test_var_2)
subs={4444444:400, 2222222:300}
rev_subs = {v:k for k,v in subs.items()}
y_pred1 = [rev_subs.get(item,item) for item in y_pred1]
y_pred2 = [rev_subs.get(item,item) for item in y_pred2]
print(accuracy_score(label_var_af_2, y_pred1))
print(confusion_matrix(label_var_af_2, y_pred1))
print(accuracy_score(label_var_2,y_pred2))
print(confusion_matrix(label_var_2, y_pred2))

df_af = df_af.loc[test_df['time_to_mace'] != 6666666]
df_af_no = df_af_no.loc[test_df['time_to_mace'] != 6666666]
df_af_5 = df_af.drop(columns=['time_to_mace','time_to_mace2'])
df_af_no_5 = df_af_no.drop(columns=['time_to_mace','time_to_mace2'])
test_var_af_5 = df_af_5.iloc[:, :len(df_af_5.columns) - 1]
test_var_af_5 = test_var_af_5.astype('float64')
label_var_af_5 = df_af_5.iloc[:, len(df_af_5.columns) - 1]
test_var_5 = df_af_no_5.iloc[:, :len(df_af_no_5.columns) - 1]
label_var_5 = df_af_no_5.iloc[:, len(df_af_no_5.columns) - 1]
test_var_5 = test_var_5.astype('float64')
test_var_5 =test_var_5.drop(columns=['Cumadin_Prior', 'Anti_Tr_Discharge'])

y_pred1 = cls5AF.predict(test_var_af_5)
y_pred2 = cls5.predict(test_var_5)
subs={4444444:400, 2222222:300}
rev_subs = {v:k for k,v in subs.items()}
y_pred1 = [rev_subs.get(item,item) for item in y_pred1]
y_pred2 = [rev_subs.get(item,item) for item in y_pred2]
print(accuracy_score(label_var_af_5, y_pred1))
print(confusion_matrix(label_var_af_5, y_pred1))
print(accuracy_score(label_var_5,y_pred2))
print(confusion_matrix(label_var_5, y_pred2))



