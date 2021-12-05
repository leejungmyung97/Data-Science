import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from pygments.lexers import graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn import tree, metrics

plt.style.use('seaborn-whitegrid')
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.tree import export_graphviz
import plotly.graph_objects as go

# Read dataset
print("------Data Set-----\n")
dataset = pd.read_csv('C:/Users/myung/Desktop/3-1/데이터과학/팀플/healthcare-dataset-stroke-data.csv')

# 2. Data Inspection
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

px.pie(Stroke_plot, values='count', names='stroke', template='plotly', title='Stroke')

plt.figure(figsize=(7, 7))
sns.countplot(x=dataset['stroke'])
plt.title('Rate of stroke', fontsize=20)
plt.xlabel('Stroke')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(7, 7))
sns.countplot(x=dataset['gender'])
plt.title('Rate of gender', fontsize=20)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# print(temp_data)
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

# correlation hour-per-week with other feature
plt.figure(figsize=(12, 8))
sns.heatmap(temp_data.corr(), linecolor='white', linewidths=1, annot=True)
plt.show()

# find outlier
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[0].boxplot(dataset['age'])
ax[0].set_title("age")
ax[1].boxplot(dataset['avg_glucose_level'])
ax[1].set_title("avg_glucose_level")
plt.show()

fig = plt.figure(figsize=(7, 7))
graph = sns.scatterplot(data=dataset, x='age', y='bmi', hue='gender')
graph.axhline(y=25, linewidth=4, color='r', linestyle='--')
fig = plt.figure(figsize=(7, 7))
graph = sns.scatterplot(data=dataset, x='age', y='bmi', hue='gender')
graph.axhline(y=25, linewidth=4, color='r', linestyle='--')

# 3. Data Pre-processing
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
print('!!!!!!\n', dataset[['gender']].groupby(['gender'], as_index=False).size())


# Function to label column within the data frame
def labelEncoding(data_frame, column):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(data_frame.loc[:, column].values)


# Label encoding for Categorical values (Sex: 'M'= 1, 'F'= 0)
dataset['gender'] = labelEncoding(dataset, 'gender')

# Label encoding for Categorical values ('Unknow' : 0, 'formerly smoked' : 1, 'never smoke' : 2, 'smoke' : 3)
dataset['smoking_status'] = labelEncoding(dataset, 'smoking_status')

# Label encoding for Categorical values
dataset['ever_married'] = labelEncoding(dataset, 'ever_married')

# Label encoding for Categorical values
dataset['work_type'] = labelEncoding(dataset, 'work_type')

# Label encoding for Categorical values
dataset['Residence_type'] = labelEncoding(dataset, 'Residence_type')

print("####After encoding")
print(dataset)
print('!!!!!!\n', dataset[['gender']].groupby(['gender'], as_index=False).size())

# 4. Data Analysis
X = dataset.drop(['stroke'], axis=1)
y = dataset.pop('stroke')

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30, random_state=1)
print("####Check train, test data shape.####")
print('Number transations x_train df', X_train.shape)
print('Number transations x_test df', X_test.shape)
print('Number transations y_train df', y_train.shape)
print('Number transations y_test df', y_test.shape)

## Scaling data

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# DecisionTree
kfold = KFold(n_splits=10)
print("")
print("------------------------------------------------------")
print("Decision Tree")
print("")

DT_model = DecisionTreeClassifier()

# fit the model
DT_model.fit(X_train, y_train)

# model score
predict_train_DT = DT_model.predict(X_train)
predict_test_DT = DT_model.predict(X_test)

# accuracy score
DT_train_score = DT_model.score(X_train, y_train)
DT_test_score = DT_model.score(X_test, y_test)

print('Accuracy on Train set', DT_train_score)
print('Accuracy on Test set', DT_test_score)
print('\n')
print(metrics.classification_report(y_test, predict_test_DT))
'''
# RandomForest

print("")
print("------------------------------------------------------")
print("Random forest")
print("")

RF_model = RandomForestClassifier()

# fit the model
RF_model.fit(X_train, y_train)

# model score
predict_train_RF = RF_model.predict(X_train)
predict_test_RF = RF_model.predict(X_test)

# accuracy score
RF_train_score = RF_model.score(X_train, y_train)
RF_test_score = RF_model.score(X_test, y_test)

print('Accuracy on Train set', RF_train_score)
print('Accuracy on Test set', RF_test_score)
print(metrics.classification_report(y_test, predict_test_RF))
'''
# KNN

print("")
print("------------------------------------------------------")
print("KNN")
print("")

KNN_model = KNeighborsClassifier()

# fit the model
KNN_model.fit(X_train, y_train)

# model score
predict_train_KNN = KNN_model.predict(X_train)
predict_test_KNN = KNN_model.predict(X_test)

# model score
predict_train_KNN = KNN_model.predict(X_train)
predict_test_KNN = KNN_model.predict(X_test)

# accuracy score
KNN_train_score = KNN_model.score(X_train, y_train)
KNN_test_score = KNN_model.score(X_test, y_test)

print('Accuracy on Train set', KNN_train_score)
print('Accuracy on Test set', KNN_test_score)
print(metrics.classification_report(y_test, predict_test_KNN))
print("-------------------------------------------")

# List Hyperparameters that we want to tune.
n_neighbors = list(range(1, 30))

# Convert to dictionary
hyperparameters = {'n_neighbors': n_neighbors}

# Use GridSearch
clf = GridSearchCV(KNN_model, hyperparameters, cv=5)

# Fit the model
best_model = clf.fit(X_train, y_train)

# Print The value of best Hyperparameters
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

y_pred = clf.predict(X_test)
print("Train set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print("cross validation score: {}".format(cross_val_score(clf, X, y, cv=kfold)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("--------------------------------------")
print("")

print("KNN(k-fold cross-validation)")
print("")
print(cross_val_score(KNN_model, X=X_train, y=y_train, cv=5))
print("Mean Score: ", cross_val_score(KNN_model, X=X_train, y=y_train, cv=5).mean(, cv=))
print("")
print("---------------------------------------------")

# Stratified k-fold cross-validation
# 데이터가 몰려있을 경우 k-cross validation사용해도 성능 평가 잘 안될수있음.
print("KNN(Stratified k-fold cross-validation)")
print("")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
score = cross_val_score(KNN_model, X=X_train, y=y_train, cv=skf)
print(score)
print("Mean Score: ", score.mean())
print("")
