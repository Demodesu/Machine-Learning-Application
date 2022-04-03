import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def get_properties(model):   
  return [i for i in model.__dict__ if i.endswith('_')] 

#print(get_properties(model))

iris = load_iris()

#print(iris) #get the information

df = pd.DataFrame(iris.data,columns=iris.feature_names).drop('petal width (cm)',axis='columns') #our data frame, putting iris data into columns with feature names

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

#df[:] = df_scaled

df['target'] = iris.target #the species we want to identify, 0 = setosa, 1 = versicolor, 2 = virginica

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]
df_concat0 = pd.concat([df0,df1])
df_concat1 = pd.concat([df0,df2])
df_concat2 = pd.concat([df1,df2])

from sklearn.model_selection import train_test_split

  #first dataset btw setosa and versicolor
X0 = df_concat0.drop(['target'],axis='columns')
Y0 = df_concat0.target

X0_train, X0_test, Y0_train, Y0_test = train_test_split(X0,Y0,test_size=0.2,random_state=0)

  #second dataset btw setosa and virginica
X1 = df_concat1.drop(['target'],axis='columns')
Y1 = df_concat1.target

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1,test_size=0.2,random_state=0)

  #third dataset btw versicolor and virginica
X2 = df_concat2.drop(['target'],axis='columns')
Y2 = df_concat2.target

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2,test_size=0.2,random_state=0)

from sklearn.svm import SVC

  #hyper parameter tuning
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(gamma='auto'),{
  'C':[1,5,10,15,20]
}, cv=5, return_train_score=False) #cv = how many cross validation
clf.fit(iris.data,iris.target) #oarameter tuning training
results = pd.DataFrame(clf.cv_results_)
#print(results) 
important_results = results[['param_C','mean_test_score']]
#print(important_results) #get results
#print(clf.best_params_)

  #first model
model0 = SVC(C=clf.best_params_['C'],kernel='linear') #c = regularization, can tune other parameters, gamma, kernal?
model0.fit(X0_train,Y0_train)

  #second model
model1 = SVC(C=clf.best_params_['C'],kernel='linear') #c = regularization, can tune other parameters, gamma, kernal?
model1.fit(X1_train,Y1_train)

  #third model
model2 = SVC(C=clf.best_params_['C'],kernel='linear') #c = regularization, can tune other parameters, gamma, kernal?
model2.fit(X2_train,Y2_train)

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

  #create the place to add plots
fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

  #first plot
ax0 = fig.add_subplot(1,4,1,projection='3d',title='Setosa vs. Versicolor')

svm0 = model0.support_
x_dfconcat0_setosa = df_concat0['sepal length (cm)'].drop(df_concat0.index[df_concat0['target']==1])
y_dfconcat0_setosa = df_concat0['sepal width (cm)'].drop(df_concat0.index[df_concat0['target']==1])
z_dfconcat0_setosa = df_concat0['petal length (cm)'].drop(df_concat0.index[df_concat0['target']==1])
x_dfconcat0_versicolor = df_concat0['sepal length (cm)'].drop(df_concat0.index[df_concat0['target']==0])
y_dfconcat0_versicolor = df_concat0['sepal width (cm)'].drop(df_concat0.index[df_concat0['target']==0])
z_dfconcat0_versicolor = df_concat0['petal length (cm)'].drop(df_concat0.index[df_concat0['target']==0])

ax0.scatter3D(x_dfconcat0_setosa, y_dfconcat0_setosa, z_dfconcat0_setosa, color='green',marker='^',label='setosa')
ax0.scatter3D(x_dfconcat0_versicolor, y_dfconcat0_versicolor, z_dfconcat0_versicolor, color='red',label='versicolor')
ax0.scatter3D(X0_train['sepal length (cm)'].iloc[svm0], X0_train['sepal width (cm)'].iloc[svm0], X0_train['petal length (cm)'].iloc[svm0], color='black', marker='D', label='support vectors')

z0 = lambda x,y: (-model0.intercept_[0]-model0.coef_[0][0]*x -model0.coef_[0][1]*y) / model0.coef_[0][2] #lambda is a small function
tmp0 = np.linspace(ax0.get_xlim()[1], ax0.get_ylim()[0], 2)
x0,y0 = np.meshgrid(tmp0,tmp0)

ax0.plot_surface(x0, y0, z0(x0,y0), color='yellow', alpha=0.3)

ax0.legend()
ax0.text((ax0.get_xlim()[0] + ax0.get_xlim()[1]) / 2, (ax0.get_ylim()[0] + ax0.get_ylim()[1]) / 2, ax0.get_zlim()[1], f'model score = {model0.score(X0_test,Y0_test)}')
ax0.set_xlabel('sepal length (cm)')
ax0.set_ylabel('sepal width (cm)')
ax0.set_zlabel('petal length (cm)')

  #second plot
ax1 = fig.add_subplot(1,4,2,projection='3d',title='Setosa vs. Virginica')

svm1 = model1.support_
x_dfconcat1_setosa = df_concat1['sepal length (cm)'].drop(df_concat1.index[df_concat1['target']==2])
y_dfconcat1_setosa = df_concat1['sepal width (cm)'].drop(df_concat1.index[df_concat1['target']==2])
z_dfconcat1_setosa = df_concat1['petal length (cm)'].drop(df_concat1.index[df_concat1['target']==2])
x_dfconcat1_virginica = df_concat1['sepal length (cm)'].drop(df_concat1.index[df_concat1['target']==0])
y_dfconcat1_virginica = df_concat1['sepal width (cm)'].drop(df_concat1.index[df_concat1['target']==0])
z_dfconcat1_virginica = df_concat1['petal length (cm)'].drop(df_concat1.index[df_concat1['target']==0])

ax1.scatter3D(x_dfconcat1_setosa, y_dfconcat1_setosa, z_dfconcat1_setosa, color='green',marker='^',label='setosa')
ax1.scatter3D(x_dfconcat1_virginica, y_dfconcat1_virginica, z_dfconcat1_virginica, color='blue',marker='+',label='virginica')
ax1.scatter3D(X1_train['sepal length (cm)'].iloc[svm1], X1_train['sepal width (cm)'].iloc[svm1], X1_train['petal length (cm)'].iloc[svm1], color='black', marker='D', label='support vectors')

z1 = lambda x,y: (-model1.intercept_[0]-model1.coef_[0][0]*x -model1.coef_[0][1]*y) / model1.coef_[0][2] #lambda is a small function
tmp1 = np.linspace(ax1.get_xlim()[1], ax1.get_ylim()[0], 2)
x1,y1 = np.meshgrid(tmp1,tmp1)

ax1.plot_surface(x1, y1, z1(x1,y1), color='cyan', alpha=0.3)

ax1.legend()
ax1.text((ax1.get_xlim()[0] + ax1.get_xlim()[1]) / 2, (ax1.get_ylim()[0] + ax1.get_ylim()[1]) / 2, ax1.get_zlim()[1], f'model score = {model1.score(X1_test,Y1_test)}')
ax1.set_xlabel('sepal length (cm)')
ax1.set_ylabel('sepal width (cm)')
ax1.set_zlabel('petal length (cm)')

  #third plot
ax2 = fig.add_subplot(1,4,3,projection='3d',title='Versicolor vs. Virginica')

svm2 = model2.support_
x_dfconcat2_versicolor = df_concat2['sepal length (cm)'].drop(df_concat2.index[df_concat2['target']==2])
y_dfconcat2_versicolor = df_concat2['sepal width (cm)'].drop(df_concat2.index[df_concat2['target']==2])
z_dfconcat2_versicolor = df_concat2['petal length (cm)'].drop(df_concat2.index[df_concat2['target']==2])
x_dfconcat2_virginica = df_concat2['sepal length (cm)'].drop(df_concat2.index[df_concat2['target']==1])
y_dfconcat2_virginica = df_concat2['sepal width (cm)'].drop(df_concat2.index[df_concat2['target']==1])
z_dfconcat2_virginica = df_concat2['petal length (cm)'].drop(df_concat2.index[df_concat2['target']==1])

ax2.scatter3D(x_dfconcat2_versicolor, y_dfconcat2_versicolor, z_dfconcat2_versicolor, color='red', label='versicolor')
ax2.scatter3D(x_dfconcat2_virginica, y_dfconcat2_virginica, z_dfconcat2_virginica, color='blue', marker='+', label='virginica')
ax2.scatter3D(X2_train['sepal length (cm)'].iloc[svm2], X2_train['sepal width (cm)'].iloc[svm2], X2_train['petal length (cm)'].iloc[svm2], color='black', marker='D', label='support vectors')

z2 = lambda x,y: (-model2.intercept_[0]-model2.coef_[0][0]*x -model2.coef_[0][1]*y) / model2.coef_[0][2] #lambda is a small function
tmp2 = np.linspace(ax2.get_xlim()[1], ax2.get_ylim()[0], 2)
x2,y2 = np.meshgrid(tmp2,tmp2)

ax2.plot_surface(x2, y2, z2(x2,y2), color='magenta', alpha=0.3)

ax2.legend()
ax2.text((ax2.get_xlim()[0] + ax2.get_xlim()[1]) / 2, (ax2.get_ylim()[0] + ax2.get_ylim()[1]) / 2, ax2.get_zlim()[1], f'model score = {model2.score(X2_test,Y2_test)}')
ax2.set_xlabel('sepal length (cm)')
ax2.set_ylabel('sepal width (cm)')
ax2.set_zlabel('petal length (cm)')

  #fourth plot
ax3 = fig.add_subplot(1,4,4,projection='3d',title='Classification of Irises')

ax3.scatter3D(df0['sepal length (cm)'],df0['sepal width (cm)'],df0['petal length (cm)'],color='green',marker='^',label='setosa')
ax3.scatter3D(df1['sepal length (cm)'],df1['sepal width (cm)'],df1['petal length (cm)'],color='red',label='versicolor')
ax3.scatter3D(df2['sepal length (cm)'],df2['sepal width (cm)'],df2['petal length (cm)'],color='blue',marker='+',label='virginica')

  #fourth plot, first plane
z3_1 = lambda x,y: (-model0.intercept_[0]-model0.coef_[0][0]*x -model0.coef_[0][1]*y) / model0.coef_[0][2] #lambda is a small function
tmp3_1 = np.linspace(ax3.get_xlim()[1], ax3.get_ylim()[0], 2)
x3_1,y3_1 = np.meshgrid(tmp3_1,tmp3_1)

ax3.plot_surface(x3_1, y3_1, z3_1(x3_1,y3_1), color='yellow', alpha=0.3)

  #fourth plot, second plane
z3_2 = lambda x,y: (-model1.intercept_[0]-model1.coef_[0][0]*x -model1.coef_[0][1]*y) / model1.coef_[0][2] #lambda is a small function
tmp3_2 = np.linspace(ax3.get_xlim()[1], ax3.get_ylim()[0], 2)
x3_2,y3_2 = np.meshgrid(tmp3_2,tmp3_2)

ax3.plot_surface(x3_2, y3_2, z3_2(x3_2,y3_2), color='cyan', alpha=0.3)

  #fourth plot, third plane
z3_3 = lambda x,y: (-model2.intercept_[0]-model2.coef_[0][0]*x -model2.coef_[0][1]*y) / model2.coef_[0][2] #lambda is a small function
tmp3_3 = np.linspace(ax3.get_xlim()[1], ax3.get_ylim()[0], 2)
x3_3,y3_3 = np.meshgrid(tmp3_3,tmp3_3)

ax3.plot_surface(x3_3, y3_3, z3_3(x3_3,y3_3), color='magenta', alpha=0.3)

ax3.legend()
ax3.set_xlabel('sepal length (cm)')
ax3.set_ylabel('sepal width (cm)')
ax3.set_zlabel('petal length (cm)')

plt.show()