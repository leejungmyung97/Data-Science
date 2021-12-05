import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('seaborn-whitegrid')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Curation
# Read dataset
dataset = pd.read_csv('C:/Users/myung/Desktop/3-1/데이터과학/팀플/healthcare-dataset-stroke-data.csv')
#
# 2. Data Inspection
print("####dataset#### ")
print(dataset.head())

print("####dataset describe####")
print(dataset.describe())

print("####dataset info####")
print(dataset.info())

print("####null data####")
print(dataset.isnull().sum())

dataset['is_Stroke'] = ' '
for i in range(len(dataset)):
    if dataset['stroke'][i] == 1:
        dataset['is_Stroke'][i] = 'Yes'
    else:
        dataset['is_Stroke'][i] = 'No'

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['is_Stroke'])
plt.title('Rate of stroke', fontsize=20)
plt.xlabel('Stroke')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['gender'])
plt.title('Gender', fontsize=20)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['heart_disease'])
plt.title('Heart Disease', fontsize=20)
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['hypertension'])
plt.title('Hyper Tension', fontsize=20)
plt.xlabel('Hyper Tension')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['ever_married'])
plt.title('Married or Not', fontsize=20)
plt.xlabel('Married')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['work_type'])
plt.title('Type of Work', fontsize=20)
plt.xlabel('Work')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['Residence_type'])
plt.title('Area of Residence', fontsize=20)
plt.xlabel('Area')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 7))
sns.countplot(x=dataset['smoking_status'])
plt.title('Smoking Status', fontsize=20)
plt.xlabel('Somking')
plt.ylabel('Count')
plt.show()

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

# Function to label column within the data frame
def labelEncoding(data_frame, column):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(data_frame.loc[:, column].values)

# Label encoding for Categorical values (Sex: 'M'= 1, 'F'= 0)
dataset['gender'] = labelEncoding(dataset, 'gender')

# Label encoding for Categorical values ('Unknow' : 0, 'formerly smoked' : 1, 'never smoke' : 2, 'smoke' : 3)
dataset['smoking_status'] = labelEncoding(dataset, 'smoking_status')

print("####After encoding")
print(dataset)


# 4. Data Analysis

# 5. Data Evaluation