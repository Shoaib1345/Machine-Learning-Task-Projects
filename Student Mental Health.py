import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,accuracy_score,f1_score

df = pd.read_csv('/kaggle/input/student-mental-health/Student Mental health.csv') # Reading data from the Student Mental health.csv file
df.head(2)

df.shape # rows = 101, col = 11
df.dtypes # 1 - float , 10 - object
df.isnull().sum() # age has a null
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'].isnull().sum()
df.columns # WE will rename these columns

df.columns = ['Date_Time', 'Gender', 'Age', 'Course', 'Year', 'CGPA', 'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
df.columns


df.head(1) # Renamed columns
df['Year'].unique() # year 1 and Year 1 are the same.We will only keep the year

df['Year'] = df['Year'].apply(lambda x : int(x.split(' ')[-1]))
df['Year'].unique()

df.head(3)

df['CGPA'].unique() #  '3.50 - 4.00', '3.50 - 4.00 ' are same . We have to remove the trailling spaces

df['CGPA'] = df['CGPA'].apply(lambda x : x.strip())
df['CGPA'].unique()

# To see the no of courses students are enrolled in

Course_List = df['Course'].unique().tolist()
print(len(Course_List))

df['Course'].unique() # There are multiple courses with the same name

course_dic = {'engin': 'Engineering' , 'Engine':'Engineering' , 'Islamic education':'Islamic Education' ,
              'Pendidikan islam':'Pendidikan Islam' , 'BIT':'IT', 'psychology':'Psychology', 'koe': 'Koe',
              'Kirkhs': 'Irkhs', 'KIRKHS': 'Irkhs', 'Benl': 'BENL', 'Fiqh fatwa ': 'Fiqh', 'Laws': 'Law'}

len(df['Course'].unique().tolist()) # So in actual there are 37 unique courses
df.sample(5)

df['Course'].value_counts() # No of students enrolled in each course
df.columns


def plot_student_cnt(dataframe):
    for yr in dataframe['Year'].unique().tolist():
        plt.figure(figsize=(15, 7))
        pl_1 = sns.countplot(x='Course', data=dataframe[dataframe['Year'] == yr])
        pl_1.set_xticklabels(pl_1.get_xticklabels(), rotation=45)
        pl_1.set_title(f'Student Count for  Year={yr}')


plot_student_cnt(df)

df['Anxiety'].value_counts() # Out of 101 students 34 are suffering from anxiety

plt.figure(figsize=(15,12))
plot_1 = sns.stripplot(x='Anxiety',y='Course',data=df,hue='Gender')
plot_1.set_title("Course vs Anxiety")

print(" No Anxiety \n")
print(df[df['Anxiety']=='No']['Course'].value_counts().nlargest(5))
print("\n Anxiety \n")
print(df[df['Anxiety']=='Yes']['Course'].value_counts().nlargest(5))

df[['Anxiety','Gender']].groupby('Gender')['Anxiety'].value_counts()

# To find the courses enrolled in where students faces no anxiety

Anx = df[df['Anxiety']=='Yes']['Course'].unique().tolist()
No_Anx = df[df['Anxiety']=='No']['Course'].unique().tolist()


set1 = set(No_Anx)
set2 = set(Anx)

print(list(sorted(set1 - set2)))

df['Depression'].value_counts()

plt.figure(figsize=(15,12))
plot_2 = sns.stripplot(x='Depression',y='Course',data=df,hue='Gender')
plot_2.set_title("Course vs Depression")
print(df[df['Depression']=='Yes']['Course'].value_counts().nlargest(5))

df.columns
df[['Gender','Depression']].value_counts()

plt.figure(figsize=(15,12))
plot_2 = sns.stripplot(x='Panic_Attack',y='Course',data=df,hue='Gender')
plot_2.set_title("Course vs Panic_Attack")
df[df['Panic_Attack']=='Yes']['Course'].value_counts().nlargest(5)

df[['Gender','Panic_Attack']].value_counts()


def problem(dataframe):
    list = ['Depression', 'Anxiety', 'Panic_Attack']
    for i in list:
        print(df[['Year', i]].groupby('Year')[i].value_counts())
        print('\n')


problem(df)

sns.countplot(x='Age',data=df)

df["Age"].value_counts() # 20.53 age has 1 record which is to be removed
row_idx = df[df['Age']==20.53].index
df.iloc[row_idx] # This row have 20.53 age
df_1 = df.drop(row_idx)
sns.countplot(x='Age',data=df_1)

print(df_1.shape)  # Removed the fractional age
df.iloc[row_idx]
from scipy.stats import shapiro
print(df[['Age','Year']].skew())

print("For age :", shapiro(df[['Age']]))

print("For Year :", shapiro(df[['Year']]))

sns.kdeplot(df['Age']) # Non Normal Distribution
sns.kdeplot(df['Year']) # Non Normal Distribution
df.columns

plot_2 = sns.countplot(x='CGPA',data=df_1,hue='Depression')
plot_2.set_title('Count of CGPA by Depression')


