import pandas as pd
import numpy as np
import os

# read data from csv file.
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

runs = pd.read_csv("C:/Users/82102/Documents/DataScience/runs.csv")
races = pd.read_csv("C:/Users/82102/Documents/DataScience/races.csv")

# PreProcessing both of runs and races data sets.
def pre_processing(A, B):
    '''--pre-processing--'''
    # Set unuse attributes and drop from the datasets.
    unused_run = ['horse_no', 'result', 'lengths_behind', 'position_sec1', 'position_sec2', 'position_sec3',
                  'position_sec4', 'position_sec5', 'position_sec6',
                  'behind_sec1', 'behind_sec2', 'behind_sec3', 'behind_sec4', 'behind_sec5', 'behind_sec6', 'time1',
                  'time2', 'time3', 'time4', 'time5', 'time6', 'finish_time', 'trainer_id', 'jockey_id']

    unused_race = ['date', 'race_no', 'horse_ratings', 'prize', 'sec_time1', 'sec_time2', 'sec_time3', 'sec_time4',
                   'sec_time5', 'sec_time6', 'sec_time7', 'time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7',
                   'place_combination1', 'place_combination2', 'place_combination3',
                   'place_combination4', 'place_dividend1', 'place_dividend2', 'place_dividend3', 'place_dividend4',
                   'win_combination1', 'win_combination2', 'win_dividend1', 'win_dividend2']

    A.drop(unused_run, axis=1, inplace=True)
    B.drop(unused_race, axis=1, inplace=True)

    # Merge two data sets to use.
    df = pd.merge(A, B, how='outer', on='race_id')

    # Fill Na values. We use 'ffill' method for flat datas.
    df['horse_age'].fillna(axis=0, method='ffill', inplace=True)
    df['horse_type'].fillna(axis=0, method='ffill', inplace=True)
    df['horse_country'].fillna(axis=0, method='ffill', inplace=True)
    df['place_odds'].fillna(axis=0, method='ffill', inplace=True)


    # Encoding Categorical Datas
    # impute gear feature
    def horse_gear_impute(cols):
        if cols == '--':
            return 0
        else:
            return 1

    df.horse_gear = df.horse_gear.apply(horse_gear_impute)
    # /impute gear feature

    # impute country feature
    def horse_country_change(cols):
        if cols == 'AUS':
            return 1
        else:
            return 0

    df.horse_country = df.horse_country.apply(horse_country_change)
    # /impute country feature

    # one hot encoding categorical data
    df = pd.get_dummies(df, drop_first=True)
    # /one hot encoding categorical data

    df2 = df.loc[df.race_id == 2001]
    df = df.loc[df.race_id != 2001]
    df.drop(['race_id'], axis=1, inplace=True)
    df.drop(['horse_id'], axis=1, inplace=True)
    df2.drop(['race_id'], axis=1, inplace=True)
    df2.drop(['horse_id'], axis=1, inplace=True)
    '''--end pre-processing--'''

    return df, df2


df, df2 = pre_processing(runs, races)
print(df)
print(df2)


# Standard Scaling Module
def StandardScaling(data):
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_df)
    return scaled_df


# MinMax Scaling Module
def MinMaxScaling(data):
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_df)
    return scaled_df


# Robust Scaling Module
def RobustScaling(data):
    scaler = preprocessing.RobustScaler()
    scaled_df = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_df)
    return scaled_df

# Method to Split Train and Test dataset.
def SplitData(X, Y, testsize):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize, shuffle=False, random_state=0)
    return X_train, X_test, y_train, y_test

# Make Scaled Dataframes to one array.
def Scaled_list(dataframe):
    scaled_data_list = []
    scaled_data_list.append(StandardScaling(dataframe))
    scaled_data_list.append(MinMaxScaling(dataframe))
    scaled_data_list.append(RobustScaling(dataframe))
    return scaled_data_list

# KNN model
def KNN(X, y):
    print(":Start KNN model")
    global columns_X
    global knn_global
    result_print = ["Standard scaled", "Minmax scaled", "Robust scaled"]
    index = 0
    best = 0
    Accuracy_score_list = []

    # scaling
    scaled_data_list = Scaled_list(X)
    # restore columns name
    for scaled_list in scaled_data_list:
        scaled_list.columns = columns_X
    # print(scaled_data_list)

    # calculate KNN accuracy each scaled DF
    for X in scaled_data_list:
        # split
        X_train, X_test, y_train, y_test = SplitData(X, y, 0.2)

        # Fitting by using knn model.
        knn = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
        knn.fit(X_train, y_train)

        # Get predicted values.
        y_pred = knn.predict(X_train)
        mismatch = 0
        for i in range(0, len(y_pred) - 1):
            if y_pred[i] == y_train.iloc[i]:
                continue;
            else:
                mismatch = mismatch + 1;
        accuracy = (1 - (mismatch / len(y_pred))) * 100;
        if accuracy > best:
            best = accuracy
            knn_global.fit(X_train, y_train)

        Accuracy_score_list.append(accuracy)
        print("------------ {} Result ------------".format(result_print[index]))
        print('The accuracy of the knn classifier for k = %d is %f%%' % (9, accuracy))
        index += 1
    print(":finish KNN model")
    return Accuracy_score_list

# Logistic Model
def Logistic(X, y):
    print(":Start Logistic model")
    global columns_X
    global reg_global
    result_print = ["Standard scaled", "Minmax scaled", "Robust scaled"]
    index = 0
    best = 0
    Accuracy_score_list = []

    # scaling
    scaled_data_list = Scaled_list(X)
    # restore columns name
    for scaled_list in scaled_data_list:
        scaled_list.columns = columns_X
    # print(scaled_data_list)

    # Make LogisticRegression model

    # Calculate accuracy for each scaled data
    result_print = ["Standard scaled", "Minmax scaled", "Robust scaled"]
    reg = LogisticRegression(max_iter=3000)
    for X in scaled_data_list:
        X_train, X_test, y_train, y_test = SplitData(X, y, 0.2)
        reg.fit(X_train, y_train)  # Regression fit in train set
        Accuracy_score = reg.score(X_test, y_test)
        if Accuracy_score > best:
            best = Accuracy_score
            reg_global.fit(X_train, y_train)
        Accuracy_score_list.append(Accuracy_score)
        print("------------ {} Result ------------".format(result_print[index]))
        print("------------ Accuracy = {}% -----------".format(Accuracy_score))
        index += 1
    print(":finish logistic model")
    return Accuracy_score_list

# Random Forest Model
def Random_forest(X, y):
    print(":Start Random forest model")
    global columns_X
    global clf_global
    result_print = ["Standard scaled", "Minmax scaled", "Robust scaled"]
    index = 0
    best = 0
    Accuracy_score_list = []

    # scaling
    scaled_data_list = Scaled_list(X)
    # restore columns name
    for scaled_list in scaled_data_list:
        scaled_list.columns = columns_X
    # print(scaled_data_list)

    # Calculate accuracy for each scaled data
    result_print = ["Standard scaled", "Minmax scaled", "Robust scaled"]

    clf = RandomForestClassifier(n_estimators=20, max_depth=20, random_state=0)
    for X in scaled_data_list:
        X_train, X_test, y_train, y_test = SplitData(X, y, 0.2)
        clf.fit(X_train, y_train)
        # Using RandomizedSearch, get fitting model.
        # pred2 = clf.predict(X_test)
        # print("Y = ",list(X_test).count(1))
        # print("P = ",list(pred2).count(1))
        # if index == 0:
        #     model = RandomForestClassifier(class_weight='balanced')

        #     search_space = {'n_estimators': [int(x) for x in np.linspace(50, 1000, num=20)],
        #                     'max_depth': [int(x) for x in np.linspace(10, 100, num=10)] + [None],
        #                     'min_samples_split': [2, 5, 10],
        #                     'min_samples_leaf': [1, 2, 4],
        #                     'criterion': ['gini', 'entropy'],
        #                     'bootstrap': [True, False]
        #                     }

        #     cv_model = RandomizedSearchCV(model,
        #                                 scoring='f1_macro',
        #                                 param_distributions=search_space,
        #                                 n_jobs=-1,
        #                                 cv=2,
        #                                 n_iter=30,
        #                                 verbose=1)

        #     cv_model.fit(X_train, y_train)

        #     print(cv_model.fit(X_train, y_train))

        Accuracy_score = clf.score(X_test, y_test)
        if Accuracy_score > best:
            best = Accuracy_score
            clf_global.fit(X_train, y_train)
        Accuracy_score_list.append(Accuracy_score)
        print("------------ {} Result ------------".format(result_print[index]))
        print("------------ Accuracy = {}% -----------".format(Accuracy_score))
        index += 1
    print(":finish Random forest model")
    return Accuracy_score_list

# Set Target Column
target_col = 'won'

# Using Pre-Porcessed data, set target and attributes array.
X = df.drop(target_col, axis=1)
y = df[target_col]
columns_X = X.columns

X2 = df2.drop(target_col, axis=1)
y2 = df2[target_col]

# To save the best accuracy model.x
knn_global = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
reg_global = LogisticRegression(max_iter=3000)
clf_global = RandomForestClassifier(n_estimators=800, max_depth=30, random_state=0)

# Make KNN model accuracy list for 3 different scaled datas.
KNN_accuracy_list = KNN(X, y)
print(KNN_accuracy_list)
print("\n")

# Make Logistic model accuracy list for 3 different scaled datas.
Logistic_accuracy_list = Logistic(X, y)
print(Logistic_accuracy_list)
print("\n")

# Make RandomForest model accuracy list for 3 different scaled datas.
Randomforest_accuracy_list = Random_forest(X, y)
print(Randomforest_accuracy_list)
print("\n")

print(X2)

# Predict won by each best model.
pred_knn = knn_global.predict(X2)
pred_reg = reg_global.predict(X2)
pred_clf = clf_global.predict(X2)

print("original Won = \n", y2)
print("Knn Model Won = ", list(pred_knn))
print("logistic Model Won = ", list(pred_reg))
print("random forest Model Won = ", list(pred_clf))

 # extract best F
# Using 5-fold Cross Validation method and KNN model, get optimal k of this dataset.
cv_scores = [];
count = 2;
max_score = 0;
neighbors = list(filter(lambda x: x % 2 != 0, range(1,100)))
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    print('selected k is %d' % (k));
    scores = cross_val_score(knn,train_img_scale, y_train, cv=4, scoring='accuracy')
    print('\bmean score is %f' % (scores.mean()))
    cv_scores.append(scores.mean())
    if max_score<scores.mean():
        max_score = scores.mean();
        count = 2;
    else:
        count = count - 1;
    if count == 0:
        break;

error = [1-x for x in cv_scores]
optimal_k = neighbors[error.index(min(error))]
print('Optimal K is', optimal_k)

# Fitting by using knn model.
knn = KNeighborsClassifier(n_neighbors=optimal_k,n_jobs=-1)
knn.fit(train_img_scale,y_train)

# Get predicted values.
y_pred = knn.predict(test_img_scale)
mismatch = 0
for i in range(0,len(y_pred)-1):
    if y_pred[i] == y_train.iloc[i]:
        continue;
    else:
        mismatch = mismatch + 1;
accuracy = (1-(mismatch/len(y_pred)))*100;
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, accuracy))


# GridSearch
KNN = KNeighborsClassifier()

# Set parameters
parameters = {'n_neighbors' : [1,2,3,4,5,6,7,8,9,10]}

# Re-educate by best parameter using GridSearch.
grid_KNN = GridSearchCV(KNN, param_grid=parameters, cv=3, refit=True)

# Using Train data, educate hyper parameters in turns.
grid_KNN.fit(train_img_scale, y_train)

# Result of Grid Search
scores_df = pd.DataFrame(grid_KNN.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('GridSearchCV Best Parameter:', grid_KNN.best_params_)
print('GridSearchCV Best Accuracy: {0:.4f}'.format(grid_KNN.best_score_))
