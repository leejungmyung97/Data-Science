import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
# import plotly.express as px


df=pd.read_csv('C:/Users/ASUS/Desktop/이정명/3-1/3-1/데이터과학/팀플/healthcare-dataset-stroke-data.csv', na_values=['Unknown', 'Other'])
dataset=pd.read_csv('C:/Users/ASUS/Desktop/이정명/3-1/3-1/데이터과학/팀플/healthcare-dataset-stroke-data.csv')

print("####dataset#### ")
print(dataset.head())
print("####dataset describe####")
print(dataset.describe())
print("####dataset info####")
print(dataset.info())
print("####null data####")
print(dataset.isnull().sum())

Stroke_plot = dataset['stroke'].value_counts().reset_index()
Stroke_plot.columns = ['stroke', 'count']

# px.pie(Stroke_plot, values='count', names='stroke', template='plotly', title='Stroke')

# plt.figure(figsize=(7, 7))
# sns.countplot(x=dataset['stroke'])
# plt.title('Rate of stroke', fontsize=20)
# plt.xlabel('Stroke')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(7, 7))
# sns.countplot(x=dataset['gender'])
# plt.title('Rate of gender', fontsize=20)
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.show()

temp_data = dataset.dropna()
temp_data = temp_data.drop(['id'], axis=1)

x = pd.get_dummies(temp_data.drop(['stroke'], axis=1))
y = temp_data['stroke']

bestfeature = SelectKBest(f_classif, k='all')
fit = bestfeature.fit(x, y)

dfcolumns = pd.DataFrame(x.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['feature', 'Score']

print("####FeatureScore####")
print(featureScores.nlargest(60, 'Score'))

# # correlation hour-per-week with other feature
# plt.figure(figsize=(12, 8))
# sns.heatmap(temp_data.corr(), linecolor='white', linewidths=1, annot=True)
# plt.show()

# find outlier
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[0].boxplot(dataset['age'])
ax[0].set_title("age")
ax[1].boxplot(dataset['avg_glucose_level'])
ax[1].set_title("avg_glucose_level")
plt.show()

# fig = plt.figure(figsize=(7,7))
# graph = sns.scatterplot(data=dataset, x='age', y='bmi', hue='gender')
# graph.axhline(y=25, linewidth=4, color='r', linestyle='--')

print("####Before drop null####")
print(dataset.isnull().sum())
dataset.dropna(inplace=True)
print("")
print("####After drop null####")
print(dataset.isnull().sum())

dataset = dataset.drop(['id'], axis=1)
print("")
print("####After drop unnecessary columns####")
print(dataset.head())

indexNames = dataset[(dataset['gender'] != 'Male')
               & (dataset['gender'] != 'Female')].index
dataset.drop(indexNames, inplace=True)
print("####After drop invalid gender####")
print(dataset)

X = dataset.drop(['stroke'], axis=1)
y = dataset.pop('stroke')

# drop id column
df.drop(['id'], axis=1, inplace=True)
# fill NaN value in bmi column
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
# # drop row if smoking_status is unknown
# df.dropna(axis=0, how='any', inplace=True)
# fill unknown data in smoking_status
df['smoking_status'].ffill(inplace=True)
df.set_index(np.arange(len(df)), inplace=True)
# fill wrong data in gender column
df['gender'].ffill(inplace=True)
# split data
x2=df.drop(['stroke'], axis=1)
y2=df['stroke']

# # show object column
print(x2.info())
# print('\n', x[['gender']].groupby(['gender'], as_index=False).size())
# print('\n', x[['ever_married']].groupby(['ever_married'], as_index=False).size())
# print('\n', x[['work_type']].groupby(['work_type'], as_index=False).size())
# print('\n', x[['Residence_type']].groupby(['Residence_type'], as_index=False).size())
# print('\n', x[['smoking_status']].groupby(['smoking_status'], as_index=False).size())

# function label encoding
# input target column list and dataframe
def lblEncoding(listObj, x):
    lbl = preprocessing.LabelEncoder()

    for i in range(len(listObj)):
        x[listObj[i]] = lbl.fit_transform(x[listObj[i]])
    # output encoded dataframe
    return x

# function ordinal encoding
# input target column list and dataframe
def ordEncoding(listObj, x):
    ord=preprocessing.OrdinalEncoder()

    for i in range(len(listObj)):
        tempColumn=x[listObj[i]].to_numpy().reshape(-1, 1)
        tempColumn=ord.fit_transform(tempColumn)
        tempColumn=tempColumn.reshape(1, -1)[0]
        x[listObj[i]].replace(x[listObj[i]].tolist(), tempColumn, inplace=True)
    # output encoded dataframe
    return x

# function ohehot encoding
# input dataframe
def ohEncoding(x):
    # output encoded dataframe
    return pd.get_dummies(x)

def dtClassifier(trainSetX, trainSetY, testSetX, testSetY):
    dt=DecisionTreeClassifier()
    dt.fit(trainSetX, trainSetY)
    print(dt.score(testSetX, testSetY))

def rfClassifier(trainSetX, trainSetY, testSetX, testSetY):
    rf=RandomForestClassifier()
    rf.fit(trainSetX, trainSetY)
    print(rf.score(testSetX, testSetY))

def knnClassifier(trainSetX, trainSetY, testSetX, testSetY):
    knn=KNeighborsClassifier()
    knn.fit(trainSetX, trainSetY)
    print(knn.score(testSetX, testSetY))

# function make preprocessing combination
# input data and target
def makeCombination(x, y):
    listObj=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    encoder = [lblEncoding(listObj, x), ordEncoding(listObj, x), ohEncoding(x)]
    nameEnc=['Label encoder', 'Ordinal encoder', 'OneHot encoder']
    scaler = [preprocessing.StandardScaler(), preprocessing.RobustScaler(), preprocessing.MaxAbsScaler(), preprocessing.MinMaxScaler()]
    nameSc=['Standard scaler', 'Robust scaler', 'MaxAbs scaler', 'MinMax scaler']
    listDf=[]
    listBestDf=[]
    listClassifier=[DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]

    # make 12 dataframe of each combination and store in listDf
    for i in range(len(encoder)):
        tempX = encoder[i]
        col=tempX.columns.values
        for j in range(len(scaler)):
            sc=scaler[j]
            tempX=sc.fit_transform(tempX)
            tempX=pd.DataFrame(tempX, columns=col)
            listDf.append(tempX)

    # search best encoder and scaler for each classifier
    for i in range(len(listClassifier)):
        classifer=listClassifier[i]
        scoreMax=0
        indexMax=0
        encBest=''
        scBest=''
        print(classifer)
        for j in range(len(listDf)):
            trainSetX, testSetX, trainSetY, testSetY = train_test_split(listDf[j], y, test_size=0.2)
            classifer.fit(trainSetX, trainSetY)
            score=classifer.score(testSetX, testSetY)
            print(score)
            if(scoreMax<=score):
                scoreMax=score
                indexMax=j
        listBestDf.append(listDf[indexMax])
        print('####Function result####')
        print('Best accuracy :', scoreMax)
        encBest=nameEnc[(int)(indexMax/4)]
        scBest=nameSc[indexMax%4]
        print('Best combination : Encoding -> ', encBest, '  Scaling -> ', scBest, '\n')

    # output list of best dataframe
    return listBestDf

# evaluation each model
def evaluation(x, y, classifier):
    # 교차검증 일반적으로 분류에는 StratifiedKFold 사용
    # https://homeproject.tistory.com/entry/%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D-cross-validation
    trainSetX, testSetX, trainSetY, testSetY = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    score=cross_val_score(classifier, trainSetX, trainSetY, cv=skf)
    print(classifier, '\nCross validation score :', score)
    print('Mean score :', score.mean())
    classifier.fit(trainSetX, trainSetY)
    print('Accuracy on test set :', classifier.score(testSetX, testSetY))
    matrix=confusion_matrix(testSetY, classifier.predict(testSetX))
    print('Confusion metrics\n', matrix)
    precision=matrix[0][0]/(matrix[0][0]+matrix[1][0])
    recall=matrix[0][0]/(matrix[0][0]+matrix[0][1])
    print('Precision :', precision)
    print('Recall :', recall)
    print('F1 score :', 2*precision*recall/(precision+recall), '\n')
    print('Classification report\n', classification_report(testSetY, classifier.predict(testSetX)))

listClassifier=[DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
# 여기서 데이터셋 바꿔야함
# listBestDf index 0=DecisionTree, 1=RandomForest, 2=KNeighbors
listBestDf=makeCombination(X, y)
for i in range(len(listBestDf)):
    evaluation(listBestDf[i], y, listClassifier[i])

# grid dt
trainSetX, testSetX, trainSetY, testSetY = train_test_split(listBestDf[0], y2, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'max_features': np.arange(1, len(testSetX.columns)), 'max_depth': np.arange(1, 20)}]
dt_gscv = GridSearchCV(listClassifier[0], param_grid, cv=2)
dt_gscv.fit(trainSetX, trainSetY)
print(dt_gscv.best_params_)
print('Best score :', dt_gscv.best_score_)

# grid rf
trainSetX, testSetX, trainSetY, testSetY = train_test_split(listBestDf[1], y2, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'max_features': np.arange(1, len(testSetX.columns)), 'max_depth': np.arange(1, 10)}]
rf_gscv = GridSearchCV(listClassifier[1], param_grid, cv=2, n_jobs=2)
rf_gscv.fit(trainSetX, trainSetY)
print(rf_gscv.best_params_)
print('Best score :', rf_gscv.best_score_)

# grid knn
trainSetX, testSetX, trainSetY, testSetY = train_test_split(listBestDf[2], y2, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'n_neighbors': np.arange(1, 10)}]
knn_gscv = GridSearchCV(listClassifier[2], param_grid, cv=2, n_jobs=2)
knn_gscv.fit(trainSetX, trainSetY)
print(knn_gscv.best_params_)
print('Best score :', knn_gscv.best_score_)

print('\n---------After GridSearchCV---------\n')
dt = DecisionTreeClassifier(max_depth=3, max_features=8)
rf = RandomForestClassifier(max_depth=9, max_features=8)
knn = KNeighborsClassifier(n_neighbors=6)
evaluation(listBestDf[0], y, dt)
evaluation(listBestDf[1], y, rf)
evaluation(listBestDf[2], y, knn)
