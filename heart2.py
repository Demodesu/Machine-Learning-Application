import pandas as pd
import numpy as np
import os

        # misc
# [[age,gender,height,weight,ap_hi,ap_low,cholest,gluc,smoke,alco,active]]
# years, M = 2 F = 1, cm, systolic, diatolic, 1 = normal 2 = above 3 = well above, 1 = normal 2 = above 3 = well above, 0 = no 1 = yes, 0 = no 1 = yes, 0 = no 1 = yes
#[[62,1,160,58,108,78,1,1,0,0,0]] mom
#[[61,2,176,80,160,90,2,1,1,0,0]] dad
#

# nu simply shows the corresponding parameter. More details are in libsvm document
# https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf
# C is a hypermeter which is set before the training model and used to control error and Gamma is also a hypermeter which is set before the training model and used to give curvature weight of the decision boundary.
# https://ai.stackexchange.com/questions/7202/why-does-training-an-svm-take-so-long-how-can-i-speed-it-up

# optimization finished, #iter = 798630523 obj is the optimal objective value of the dual SVM problem
# obj = -279124.116889, rho = -8.133482 rho is the bias term in the decision function sgn(w^Tx - rho)
# nSV = 19308, nBSV = 14878 SV and nBSV are number of support vectors and bounded support vectors (i.e., alpha_i = C)
# Total nSV = 19308 nu-svm is a somewhat equivalent form of C-SVM where C is replaced by nu

# optimization finished, #iter = 728753435
# obj = -275588.396698, rho = -9.020698
# nSV = 18845, nBSV = 14497
# Total nSV = 18845

# training time around 8 hrs

# 0.14372735	0.003068065	0.030737066	0.00220126	entropy	300	sqrt	4	2	{'criterion': 'entropy', 'max_depth': 300, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2}	0.737142857	0.748571429	0.751428571	0.746781116	0.708154506	0.738415696	0.015873335	1
# best run time + score

# dataset from
# https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?resource=download

os.chdir("E:\Gun's stuff\Machine Learning and Deep Learning\Machine Learning\practice using machine learning\heart disease 2")

df = pd.read_csv('cardio_train.csv')

# Age | Objective Feature | age | int (days)
# Height | Objective Feature | height | int (cm) |
# Weight | Objective Feature | weight | float (kg) |
# Gender | Objective Feature | gender | categorical code |
# Systolic blood pressure | Examination Feature | ap_hi | int |
# Diastolic blood pressure | Examination Feature | ap_lo | int |
# Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
# Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
# Smoking | Subjective Feature | smoke | binary |
# Alcohol intake | Subjective Feature | alco | binary |
# Physical activity | Subjective Feature | active | binary |
# Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

        #data manipulation
    #cleaning data
np.where(df.duplicated(keep='first')==True) #finds the duplicates but keeps the first value as False
df = df.drop(np.where(df.duplicated(keep='first')==True)[0],axis=0).reset_index(drop=True) #drop all duplicates 

np.where(df.isna()==True) #find out where is NAN value in data frame
df = df.drop(np.where(df.isna()==True)[0],axis=0).reset_index(drop=True) #drop the NAN values in data frame and then reset index and assign to new data frame

    #transform data
df['age'] = df['age'].apply(lambda x: x / 365)

# normal systolic/diastolic 120/80 mmHg
# above 220 is considered dead
# below 20 considered dead
df = df.drop(df.index[df['ap_hi'].gt(220)])
df = df.drop(df.index[df['ap_lo'].gt(220)])
df = df.drop(df.index[df['ap_hi'].lt(20)])
df = df.drop(df.index[df['ap_lo'].lt(20)])

#     #train test split
# from sklearn.model_selection import train_test_split

# X = df.drop(['cardio'],axis='columns')
# Y = df.cardio

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#         #model selection
#     #model
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier

#     #select best model
# from sklearn.model_selection import cross_val_score

#     #estimate best model
# print(cross_val_score(DecisionTreeClassifier(),X_train,Y_train).mean())
# print(cross_val_score(SVC(),X_train,Y_train).mean())
# print(cross_val_score(LogisticRegression(max_iter=2000),X_train,Y_train).mean())
# print(cross_val_score(RandomForestClassifier(),X_train,Y_train).mean())
# 0.6368524473026081
# 0.7203465523401216
# 0.7188996070025009
# 0.7065023222579493
# svm is the best

#         #hyper parameter tuning
# from sklearn.model_selection import GridSearchCV

# X_dummy = df.drop(['cardio'],axis='columns')
# Y_dummy = df.cardio

# X_train_dummy, X_test_dummy, Y_train_dummy, Y_test_dummy = train_test_split(X,Y,train_size=0.05,random_state=0)

# GS = GridSearchCV(RandomForestClassifier(n_jobs=-1,random_state=0,verbose=True), {
#     'criterion':['gini','entropy'],
#     'max_depth':[100,200,300,None],
#     'min_samples_split':[2,3,4,5],
#     'min_samples_leaf':[1,2,3,4],
#     'max_features':['auto','sqrt','log2']
# },cv=5,return_train_score=False)

# GS.fit(X_train_dummy.values,Y_train_dummy.values)
# #pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
# pd.DataFrame(GS.cv_results_).to_csv('rf_hyperparameters.csv', index=False)
# print(pd.DataFrame(GS.cv_results_))

# GS = GridSearchCV(BernoulliNB(), {
#     'binarize':[0,0.25,0.75,1],
#     'fit_prior':[True,False],
# },cv=5,return_train_score=False,n_jobs=-1,verbose=True)

# GS.fit(X_train_dummy.values,Y_train_dummy.values)
# #pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
# pd.DataFrame(GS.cv_results_).to_csv('bnb_hyperparameters.csv', index=False)
# print(pd.DataFrame(GS.cv_results_))

# GS = GridSearchCV(MultinomialNB(), {
#     'fit_prior':[True,False],
# },cv=5,return_train_score=False,n_jobs=-1,verbose=True)

# GS.fit(X_train_dummy.values,Y_train_dummy.values)
# #pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
# pd.DataFrame(GS.cv_results_).to_csv('mnb_hyperparameters.csv', index=False)
# print(pd.DataFrame(GS.cv_results_))

# GS = GridSearchCV(LogisticRegression(max_iter=10000), {
#     'C':[1,5,10,15],
#     'solver':['newton-cg','lbfgs','liblinear','sag','saga']
# },cv=5,return_train_score=False,n_jobs=-1,verbose=True)

# GS.fit(X_train_dummy.values,Y_train_dummy.values)
# #pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
# pd.DataFrame(GS.cv_results_).to_csv('lg_hyperparameters.csv', index=False)
# print(pd.DataFrame(GS.cv_results_))

# GS = GridSearchCV(KNeighborsClassifier(), {
#     'n_neighbors':[1,5,10,15,20,30,50,70],
#     'weights':['uniform','distance'],
#     'algorithm':['ball_tree','kd_tree','brute'],
#     'leaf_size':[30,40,50,60],
#     'p':[1,2,3]
# },cv=5,return_train_score=False,n_jobs=-1,verbose=True)

# GS.fit(X_train_dummy.values,Y_train_dummy.values)
# #pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
# pd.DataFrame(GS.cv_results_).to_csv('kn_hyperparameters.csv', index=False)
# print(pd.DataFrame(GS.cv_results_))

#         #random forest
# dt_model = RandomForestClassifier(n_estimators=1000,verbose=True,n_jobs=-1)
# dt_model.fit(X_train.values,Y_train.values)

# print(dt_model.score(X_test,Y_test))
# print(dt_model.predict([[21,2,175,95,100,80,2,2,0,0,0]]))
# print(dt_model.predict_proba([[21,2,175,95,100,80,2,2,0,0,0]]))

        #bagging classifier
   #parameters
# from sklearn.ensemble import BaggingClassifier
# bag_model = BaggingClassifier(
#     base_estimator=SVC(C=10,kernel='linear',gamma='auto',verbose=2),
#     n_estimators=2,
#     max_samples=0.7,
#     oob_score=True,
#     n_jobs=1,
#     random_state=0
# ) 

#         #bagging classifier
#    #parameters
# from sklearn.ensemble import BaggingClassifier
# bag_model = BaggingClassifier(
#     base_estimator=RandomForestClassifier(criterion='entropy',max_depth=300,max_features='sqrt',min_samples_leaf=4,min_samples_split=2),
#     n_estimators=50,
#     max_samples=0.5,
#     oob_score=True,
#     n_jobs=-1,
#     random_state=0,
#     verbose=True
# ) 

# bag_model.fit(X_train.values,Y_train.values)

#         #loading and saving
#     #joblib
# import joblib

# # joblib.dump(bag_model, 'rf_bagging')

# load_model = joblib.load('rf_bagging')

# print(load_model.score(X_test.values,Y_test.values))
# print(load_model.predict([[61,2,176,80,160,90,2,1,1,0,0]]))
# print(load_model.predict_proba([[61,2,176,80,160,90,2,1,1,0,0]]))

        #univariate analysis
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

    #analysis
fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

#     #age
# age_df = df[['age','cardio']]
# age_no_df = age_df.drop(age_df.index[age_df['cardio']==1])
# age_yes_df = age_df.drop(age_df.index[age_df['cardio']==0])

# ax0 = fig.add_subplot(3,4,1,title='Age')
# ax0.boxplot(age_no_df.drop(columns='cardio'),positions=[0])
# ax0.boxplot(age_yes_df.drop(columns='cardio'),positions=[1])
# ax0.text(0, 1, f'No Mean = {age_no_df["age"].mean():.2f}', transform=ax0.transAxes)
# ax0.text(1, 1, f'Yes Mean = {age_yes_df["age"].mean():.2f}', transform=ax0.transAxes, horizontalalignment='right')

#     #sex
# sex_df = df[['gender','cardio']]
# sex_m_df = sex_df.drop(sex_df.index[sex_df['cardio']==0])
# sex_m_df = sex_m_df.drop(sex_m_df.index[sex_m_df['gender']==1])
# sex_f_df = sex_df.drop(sex_df.index[sex_df['cardio']==0])
# sex_f_df = sex_f_df.drop(sex_f_df.index[sex_f_df['gender']==2])
# sex_data = [len(sex_f_df.index),len(sex_m_df.index)]

# ax1 = fig.add_subplot(3,4,2,title='Sex')
# ax1.pie(sex_data,labels=['Female','Male'],autopct='%.2f%%',shadow='True')

#     #height
# height_df = df[['height','cardio']]
# height_no_df = height_df.drop(height_df.index[height_df['cardio']==1])
# height_yes_df = height_df.drop(height_df.index[height_df['cardio']==0])

# ax2 = fig.add_subplot(3,4,3,title='Height')
# ax2.boxplot(height_no_df.drop(columns='cardio'),positions=[0])
# ax2.boxplot(height_yes_df.drop(columns='cardio'),positions=[1])
# ax2.text(0, 1, f'No Mean = {height_no_df["height"].mean():.2f}', transform=ax2.transAxes)
# ax2.text(1, 1, f'Yes Mean = {height_yes_df["height"].mean():.2f}', transform=ax2.transAxes, horizontalalignment='right')

#     #weight
# weight_df = df[['weight','cardio']]
# weight_no_df = weight_df.drop(weight_df.index[weight_df['cardio']==1])
# weight_yes_df = weight_df.drop(weight_df.index[weight_df['cardio']==0])

# ax3 = fig.add_subplot(3,4,4,title='Weight')
# ax3.boxplot(weight_no_df.drop(columns='cardio'),positions=[0])
# ax3.boxplot(weight_yes_df.drop(columns='cardio'),positions=[1])
# ax3.text(0, 1, f'No Mean = {weight_no_df["weight"].mean():.2f}', transform=ax3.transAxes)
# ax3.text(1, 1, f'Yes Mean = {weight_yes_df["weight"].mean():.2f}', transform=ax3.transAxes, horizontalalignment='right')

#     #blood pressure high (bph)
# bph_df = df[['ap_hi','cardio']]
# bph_no_df = bph_df.drop(bph_df.index[bph_df['cardio']==1])
# bph_yes_df = bph_df.drop(bph_df.index[bph_df['cardio']==0])

# ax4 = fig.add_subplot(3,4,5,title='BPH')
# ax4.boxplot(bph_no_df.drop(columns='cardio'),positions=[0])
# ax4.boxplot(bph_yes_df.drop(columns='cardio'),positions=[1])
# ax4.text(0, 1, f'No Mean = {bph_no_df["ap_hi"].mean():.2f}', transform=ax4.transAxes)
# ax4.text(1, 1, f'Yes Mean = {bph_yes_df["ap_hi"].mean():.2f}', transform=ax4.transAxes, horizontalalignment='right')

#     #blood pressure low (bpl)
# bpl_df = df[['ap_lo','cardio']]
# bpl_no_df = bpl_df.drop(bpl_df.index[bpl_df['cardio']==1])
# bpl_yes_df = bpl_df.drop(bpl_df.index[bpl_df['cardio']==0])

# ax5 = fig.add_subplot(3,4,6,title='BPL')
# ax5.boxplot(bpl_no_df.drop(columns='cardio'),positions=[0])
# ax5.boxplot(bpl_yes_df.drop(columns='cardio'),positions=[1])
# ax5.text(0, 1, f'No Mean = {bpl_no_df["ap_lo"].mean():.2f}', transform=ax5.transAxes)
# ax5.text(1, 1, f'Yes Mean = {bpl_yes_df["ap_lo"].mean():.2f}', transform=ax5.transAxes, horizontalalignment='right')

#     #cholesterol
# chol_df = df[['cholesterol','cardio']]
# chol_nor_df = chol_df.drop(chol_df.index[chol_df['cholesterol']!=1])
# chol_nor_df = chol_nor_df.drop(chol_nor_df.index[chol_nor_df['cardio']==0])
# chol_abv_df = chol_df.drop(chol_df.index[chol_df['cholesterol']!=2])
# chol_abv_df = chol_abv_df.drop(chol_abv_df.index[chol_abv_df['cardio']==0])
# chol_wabv_df = chol_df.drop(chol_df.index[chol_df['cholesterol']!=3])
# chol_wabv_df = chol_wabv_df.drop(chol_wabv_df.index[chol_wabv_df['cardio']==0])
# chol_data = [len(chol_nor_df.index),len(chol_abv_df.index),len(chol_wabv_df.index)]

# ax6 = fig.add_subplot(3,4,7,title='Cholesterol')
# ax6.pie(chol_data,labels=['NOR','ABV','WABV'],autopct='%.2f%%',shadow='True')

#     #glucose
# gluc_df = df[['gluc','cardio']]
# gluc_nor_df = gluc_df.drop(gluc_df.index[gluc_df['gluc']!=1])
# gluc_nor_df = gluc_nor_df.drop(gluc_nor_df.index[gluc_nor_df['cardio']==0])
# gluc_abv_df = gluc_df.drop(gluc_df.index[gluc_df['gluc']!=2])
# gluc_abv_df = gluc_abv_df.drop(gluc_abv_df.index[gluc_abv_df['cardio']==0])
# gluc_wabv_df = gluc_df.drop(gluc_df.index[gluc_df['gluc']!=3])
# gluc_wabv_df = gluc_wabv_df.drop(gluc_wabv_df.index[gluc_wabv_df['cardio']==0])
# gluc_data = [len(gluc_nor_df.index),len(gluc_abv_df.index),len(gluc_wabv_df.index)]

# ax7 = fig.add_subplot(3,4,8,title='Glucose')
# ax7.pie(gluc_data,labels=['NOR','ABV','WABV'],autopct='%.2f%%',shadow='True')

#     #smoke
# smoke_df = df[['smoke','cardio']]
# smoke_no_df = smoke_df.drop(smoke_df.index[smoke_df['cardio']==0])
# smoke_no_df = smoke_no_df.drop(smoke_no_df.index[smoke_no_df['smoke']==1])
# smoke_yes_df = smoke_df.drop(smoke_df.index[smoke_df['cardio']==0])
# smoke_yes_df = smoke_yes_df.drop(smoke_yes_df.index[smoke_yes_df['smoke']==0])
# smoke_data = [len(smoke_no_df.index),len(smoke_yes_df.index)]

# ax8 = fig.add_subplot(3,4,9,title='Smoke')
# ax8.pie(smoke_data,labels=['No','Yes'],autopct='%.2f%%',shadow='True')

#     #alcohol
# alcohol_df = df[['alco','cardio']]
# alcohol_no_df = alcohol_df.drop(alcohol_df.index[alcohol_df['cardio']==0])
# alcohol_no_df = alcohol_no_df.drop(alcohol_no_df.index[alcohol_no_df['alco']==1])
# alcohol_yes_df = alcohol_df.drop(alcohol_df.index[alcohol_df['cardio']==0])
# alcohol_yes_df = alcohol_yes_df.drop(alcohol_yes_df.index[alcohol_yes_df['alco']==0])
# alcohol_data = [len(alcohol_no_df.index),len(alcohol_yes_df.index)]

# ax9 = fig.add_subplot(3,4,10,title='Alcohol')
# ax9.pie(alcohol_data,labels=['No','Yes'],autopct='%.2f%%',shadow='True')

#     #excercise
# excercise_df = df[['active','cardio']]
# excercise_no_df = excercise_df.drop(excercise_df.index[excercise_df['cardio']==0])
# excercise_no_df = excercise_no_df.drop(excercise_no_df.index[excercise_no_df['active']==1])
# excercise_yes_df = excercise_df.drop(excercise_df.index[excercise_df['cardio']==0])
# excercise_yes_df = excercise_yes_df.drop(excercise_yes_df.index[excercise_yes_df['active']==0])
# excercise_data = [len(excercise_no_df.index),len(excercise_yes_df.index)]

# ax10 = fig.add_subplot(3,4,11,title='Excercise')
# ax10.pie(alcohol_data,labels=['No','Yes'],autopct='%.2f%%',shadow='True')

    #age vs weight vs height

    #train test split
from sklearn.model_selection import train_test_split

df_a_w_h = df[['age','weight','height','cardio']]
df_a_w_h_x = df_a_w_h.drop('cardio',axis='columns')
df_a_w_h_y = df_a_w_h.cardio

X_train, X_test, Y_train, Y_test = train_test_split(df_a_w_h_x,df_a_w_h_y,train_size=0.2,random_state=0)

concat_df_awh = pd.concat([X_train,Y_train],axis='columns')

df_a_w_h_no = concat_df_awh.drop(concat_df_awh.index[concat_df_awh['cardio']==1])
df_a_w_h_yes = concat_df_awh.drop(concat_df_awh.index[concat_df_awh['cardio']==0])

ax11 = fig.add_subplot(1,2,2,projection='3d',title='Age vs. Weight vs. Height')
ax12 = fig.add_subplot(1,2,1,projection='3d',title='Age vs. Weight vs. Height')

ax11.scatter3D(df_a_w_h_no['age'], df_a_w_h_no['weight'], df_a_w_h_no['height'], color='green',marker='^',label='No')
ax12.scatter3D(df_a_w_h_yes['age'], df_a_w_h_yes['weight'], df_a_w_h_yes['height'], color='red',marker='D',label='Yes')

ax11.legend()
ax12.legend()
ax11.set_xlabel('Age (years)')
ax11.set_ylabel('Weight (kg)')
ax11.set_zlabel('Height (cm)')
ax12.set_xlabel('Age (years)')
ax12.set_ylabel('Weight (kg)')
ax12.set_zlabel('Height (cm)')

plt.show()