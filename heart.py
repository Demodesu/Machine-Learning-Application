import pandas as pd
import numpy as np
import os

os.chdir("E:\Gun's stuff\Machine Learning and Deep Learning\Machine Learning\practice using machine learning\heart disease")

#dataset from
#https://archive.ics.uci.edu/ml/datasets/Heart+Disease
#https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

unmod_df = pd.read_csv('new_heart.csv')
df = pd.read_csv('new_heart.csv')

#print(df.columns)
#print(len(df.columns))

# in this data frame we have 11 features and 1 output
# the features are:
# Age, Sex, Chest Pain Type, Cholesterol, Fasting Blood Sugar, 
# Resting Electrocardio Graphic Results, Max Heart Rate, Excercise, 
# Old Peak, Slope of ST

# Explanation
# Age(years)
# Sex(male/female)
# Chest Pain Type TA(typical angina), ATA(atypical angina), NAP(non-angina pain), ASY(asymptomatic)
# Cholesterol(mm/dl)
# Fasting Blood Sugar If > 120 mg/dl = 1, else = 0
# Resting Electrographic Results Normal or ST(ST-T wave abnormality) or LVH(left ventricular hypertrophy)
# Max Heart Rate btw 60 to 202
# Excercise Yes or No
# Old Peak(ST depression induced by excercise relative to rest) 
# Slope of ST Up = upsloping, Flat = flat, Down = downsloping

        #machine learning

    #cleaning data
np.where(df.duplicated(keep='first')==True) #finds the duplicates but keeps the first value as False
df = df.drop(np.where(df.duplicated(keep='first')==True)[0],axis=0).reset_index(drop=True) #drop all duplicates 

np.where(df.isna()==True) #find out where is NAN value in data frame
df = df.drop(np.where(df.isna()==True)[0],axis=0).reset_index(drop=True) #drop the NAN values in data frame and then reset index and assign to new data frame

#     #transform categorical data to numbers
# from sklearn.preprocessing import LabelEncoder
# hotencode = LabelEncoder()

# df_not_encode = df.select_dtypes(include=[np.number]) #create a df that doesn't need to be encoded
# df_encode = df.select_dtypes(exclude=[np.number]) #create a df that needs to be encoded

# for col in df_encode: #encode the categorical data
#     df_encode[col] = hotencode.fit_transform(df_encode[col])

# df = pd.concat([df_not_encode,df_encode],axis='columns')

# col_to_move = df.pop('HeartDisease')
# df['HeartDisease'] = col_to_move

    #train test split
from sklearn.model_selection import train_test_split

X = df.drop(['HeartDisease'],axis='columns')
Y = df.HeartDisease

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    #model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

    #select best model
from sklearn.model_selection import cross_val_score

#     #estimate best model
# print(cross_val_score(DecisionTreeClassifier(),X_train,Y_train).mean())
# print(cross_val_score(SVC(),X_train,Y_train).mean())
# print(cross_val_score(LogisticRegression(max_iter=2000),X_train,Y_train).mean())
# print(cross_val_score(RandomForestClassifier(),X_train,Y_train).mean())
# #random forest is best

# from sklearn.ensemble import BaggingClassifier
# bag_model = BaggingClassifier(
#     base_estimator=RandomForestClassifier(n_estimators=200),
#     n_estimators=1000,
#     max_samples=0.5,
#     max_features=0.5,
#     oob_score=True,
#     random_state=0
# ) 

# bag_model.fit(X_train.values,Y_train.values)

# print(bag_model.score(X_test.values,Y_test.values))
# print(bag_model.predict([[21,100,150,0,160,2,1,3,1,0,2]]))
# print(bag_model.predict_proba([[21,100,150,0,160,2,1,3,1,0,2]]))

# import joblib
# # joblib.dump(bag_model, 'model_joblib')

# load_model = joblib.load('model_joblib')

# print(load_model.score(X_test.values,Y_test.values))
# print(load_model.predict([[21,120,175,0,144,2,0,3,1,0,2]]))
# print(load_model.predict_proba([[21,120,175,0,144,2,0,3,1,0,2]]))

# # [[21,100,150,0,160,2,1,3,1,0,2]]
# # [[21,120,175,0,144,2,0,3,1,0,2]] 
# # Age - your age
# # RestingBP - your resting blood pressure (mean = 131.344)
# # Cholesterol - your cholesterol (mean = 249.659) more than 150 is high
# # FastingBS - > 120 mg/dl = 1, else = 0 -> blood sugar
# # MaxHR - your max heart rate (mean = 149.678)
# # Oldpeak - ST depression induced by exercise relative to rest (mean = 1.05) -> should slope upwards, more excercise
# # Sex - M or F (M = 1, F = 0)
# # ChestPainType - (ASY = 0, ATA = 1, NAP = 2, TA = 3) typical is normal
# # RestingECG - (Normal = 1, ST = 2, LVH = 0)
# # ExerciseAngina - (0 = No, 1 = Yes) does your chest hurt when you excercise?	
# # ST_Slope - (Up = 2, Flat = 1, Down = 0)	

        #univariate analysis

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

    #analysis
fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

    #age
age_df = df[['Age','HeartDisease']]
age_no_df = age_df.drop(age_df.index[age_df['HeartDisease']==1])
age_yes_df = age_df.drop(age_df.index[age_df['HeartDisease']==0])

ax0 = fig.add_subplot(3,4,1,title='Age')
ax0.boxplot(age_no_df.drop(columns='HeartDisease'),positions=[0])
ax0.boxplot(age_yes_df.drop(columns='HeartDisease'),positions=[1])

    #sex
sex_df = df[['Sex','HeartDisease']]
sex_m_df = sex_df.drop(sex_df.index[sex_df['HeartDisease']==0])
sex_m_df = sex_m_df.drop(sex_m_df.index[sex_m_df['Sex']==0])
sex_f_df = sex_df.drop(sex_df.index[sex_df['HeartDisease']==0])
sex_f_df = sex_f_df.drop(sex_f_df.index[sex_f_df['Sex']==1])
sex_data = [len(sex_f_df.index),len(sex_m_df.index)]

ax1 = fig.add_subplot(3,4,2,title='Sex')
ax1.pie(sex_data,labels=['Female','Male'],autopct='%.2f%%',shadow='True')

    #chestpain
chp_df = df[['ChestPainType','HeartDisease']]
chp_AYS_df = chp_df.drop(chp_df.index[chp_df['ChestPainType']!=0])
chp_AYS_df = chp_AYS_df.drop(chp_AYS_df.index[chp_AYS_df['HeartDisease']==0])
chp_ATA_df = chp_df.drop(chp_df.index[chp_df['ChestPainType']!=1])
chp_ATA_df = chp_ATA_df.drop(chp_ATA_df.index[chp_ATA_df['HeartDisease']==0])
chp_NAP_df = chp_df.drop(chp_df.index[chp_df['ChestPainType']!=2])
chp_NAP_df = chp_NAP_df.drop(chp_NAP_df.index[chp_NAP_df['HeartDisease']==0])
chp_TA_df = chp_df.drop(chp_df.index[chp_df['ChestPainType']!=3])
chp_TA_df = chp_TA_df.drop(chp_TA_df.index[chp_TA_df['HeartDisease']==0])
chp_data = [len(chp_AYS_df.index),len(chp_ATA_df.index),len(chp_NAP_df.index),len(chp_TA_df.index)]

ax2 = fig.add_subplot(3,4,3,title='Chest Pain')
ax2.pie(chp_data,labels=['AYS','ATA','NAP','TA'],autopct='%.2f%%',shadow='True')

    #resting blood pressure
resting_bp_df = df[['RestingBP','HeartDisease']]
resting_bp_no_df = resting_bp_df.drop(resting_bp_df.index[resting_bp_df['HeartDisease']==1])
resting_bp_yes_df = resting_bp_df.drop(resting_bp_df.index[resting_bp_df['HeartDisease']==0])

ax3 = fig.add_subplot(3,4,4,title='Resting BP')
ax3.boxplot(resting_bp_no_df.drop(columns='HeartDisease'),positions=[0])
ax3.boxplot(resting_bp_yes_df.drop(columns='HeartDisease'),positions=[1])

    #cholesterol
cholesterol_df = df[['Cholesterol','HeartDisease']]
cholesterol_no_df = cholesterol_df.drop(cholesterol_df.index[cholesterol_df['HeartDisease']==1])
cholesterol_yes_df = cholesterol_df.drop(cholesterol_df.index[cholesterol_df['HeartDisease']==0])

ax4 = fig.add_subplot(3,4,5,title='Cholesterol')
ax4.boxplot(resting_bp_no_df.drop(columns='HeartDisease'),positions=[0])
ax4.boxplot(resting_bp_yes_df.drop(columns='HeartDisease'),positions=[1])

    #fasting BS
fasting_bs_df = df[['FastingBS','HeartDisease']]
fasting_bs_no_df = fasting_bs_df.drop(fasting_bs_df.index[fasting_bs_df['HeartDisease']==0])
fasting_bs_no_df = fasting_bs_no_df.drop(fasting_bs_no_df.index[fasting_bs_no_df['FastingBS']==1])
fasting_bs_yes_df = fasting_bs_df.drop(fasting_bs_df.index[fasting_bs_df['HeartDisease']==0])
fasting_bs_yes_df = fasting_bs_yes_df.drop(fasting_bs_yes_df.index[fasting_bs_yes_df['FastingBS']==0])
fasting_bs_data = [len(fasting_bs_no_df.index),len(fasting_bs_yes_df.index)]

ax5 = fig.add_subplot(3,4,6,title='Fasting BS')
ax5.pie(fasting_bs_data,labels=['<120mg/dl','>120mg/dl'],autopct='%.2f%%',shadow='True')

    #resting ECG
ecg_df = df[['RestingECG','HeartDisease']]
ecg_nor_df = ecg_df.drop(ecg_df.index[ecg_df['RestingECG']!=1])
ecg_nor_df = ecg_nor_df.drop(ecg_nor_df.index[ecg_nor_df['HeartDisease']==0])
ecg_st_df = ecg_df.drop(ecg_df.index[ecg_df['RestingECG']!=2])
ecg_st_df = ecg_st_df.drop(ecg_st_df.index[ecg_st_df['HeartDisease']==0])
ecg_lvh_df = ecg_df.drop(ecg_df.index[ecg_df['RestingECG']!=0])
ecg_lvh_df = ecg_lvh_df.drop(ecg_lvh_df.index[ecg_lvh_df['HeartDisease']==0])
ecg_data = [len(ecg_nor_df.index),len(ecg_st_df.index),len(ecg_lvh_df.index)]

ax6 = fig.add_subplot(3,4,7,title='Resting ECG')
ax6.pie(ecg_data,labels=['NOR','ST','LVH'],autopct='%.2f%%',shadow='True')

    #max HR
max_hr_df = df[['MaxHR','HeartDisease']]
max_hr_no_df = max_hr_df.drop(max_hr_df.index[max_hr_df['HeartDisease']==1])
max_hr_yes_df = max_hr_df.drop(max_hr_df.index[max_hr_df['HeartDisease']==0])

ax7 = fig.add_subplot(3,4,8,title='Max HR')
ax7.boxplot(max_hr_no_df.drop(columns='HeartDisease'),positions=[0])
ax7.boxplot(max_hr_yes_df.drop(columns='HeartDisease'),positions=[1])

    #exercise angina
ex_an_df = df[['ExerciseAngina','HeartDisease']]
ex_an_no_df = ex_an_df.drop(ex_an_df.index[ex_an_df['HeartDisease']==0])
ex_an_no_df = ex_an_no_df.drop(ex_an_no_df.index[ex_an_no_df['ExerciseAngina']==1])
ex_an_yes_df = ex_an_df.drop(ex_an_df.index[ex_an_df['HeartDisease']==0])
ex_an_yes_df = ex_an_yes_df.drop(ex_an_yes_df.index[ex_an_yes_df['ExerciseAngina']==0])
ex_an_data = [len(ex_an_no_df.index),len(ex_an_yes_df.index)]

ax8 = fig.add_subplot(3,4,9,title='Excercise Angina')
ax8.pie(ex_an_data,labels=['No','Yes'],autopct='%.2f%%',shadow='True')

    #old peak
old_df = df[['Oldpeak','HeartDisease']]
old_no_df = old_df.drop(old_df.index[old_df['HeartDisease']==1])
old_yes_df = old_df.drop(old_df.index[old_df['HeartDisease']==0])

ax9 = fig.add_subplot(3,4,10,title='Old Peak')
ax9.boxplot(old_no_df.drop(columns='HeartDisease'),positions=[0])
ax9.boxplot(old_yes_df.drop(columns='HeartDisease'),positions=[1])

    #ST slope
st_slope_df = df[['ST_Slope','HeartDisease']]
st_slope_up_df = st_slope_df.drop(st_slope_df.index[st_slope_df['ST_Slope']!=2])
st_slope_up_df = st_slope_up_df.drop(st_slope_up_df.index[st_slope_up_df['HeartDisease']==0])
st_slope_flat_df = st_slope_df.drop(st_slope_df.index[st_slope_df['ST_Slope']!=1])
st_slope_flat_df = st_slope_flat_df.drop(st_slope_flat_df.index[st_slope_flat_df['HeartDisease']==0])
st_slope_down_df = st_slope_df.drop(st_slope_df.index[st_slope_df['ST_Slope']!=0])
st_slope_down_df = st_slope_down_df.drop(st_slope_down_df.index[st_slope_down_df['HeartDisease']==0])
st_slope_data = [len(st_slope_up_df.index),len(st_slope_flat_df.index),len(st_slope_down_df.index)]

ax10 = fig.add_subplot(3,4,11,title='ST Slope')
ax10.pie(st_slope_data,labels=['UP','FLAT','DOWN'],autopct='%.2f%%',shadow='True')

plt.show()

# 1 = male, 0 = female

# -- Value 1: typical angina
# -- Value 2: atypical angina
# -- Value 3: non-anginal pain
# -- Value 4: asymptomatic

# fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

# 19 restecg: resting electrocardiographic results
# -- Value 0: normal
# -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

# 38 exang: exercise induced angina (1 = yes; 0 = no)

# 40 oldpeak = ST depression induced by exercise relative to rest

# 41 slope: the slope of the peak exercise ST segment
# -- Value 1: upsloping
# -- Value 2: flat
# -- Value 3: downsloping

# Age - your age
# RestingBP - your resting blood pressure (mean = 131.344)
# Cholesterol - your cholesterol (mean = 249.659) more than 150 is high
# FastingBS - > 120 mg/dl = 1, else = 0 -> blood sugar
# MaxHR - your max heart rate (mean = 149.678)
# Oldpeak - ST depression induced by exercise relative to rest (mean = 1.05) -> should slope upwards, more excercise
# Sex - M or F (M = 1, F = 0)
# ChestPainType - (ASY = 0, ATA = 1, NAP = 2, TA = 3) typical is normal
# RestingECG - (Normal = 1, ST = 2, LVH = 0)
# ExerciseAngina - (0 = No, 1 = Yes) does your chest hurt when you excercise?	
# ST_Slope - (Up = 2, Flat = 1, Down = 0)	

