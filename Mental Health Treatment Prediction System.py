import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

df= pd.read_csv("survey.csv")

print(df.columns)
print(df.head())

df["gender"] = df["gender"].str.lower()
df["gender"] = df["gender"].replace({
    "male-ish": "male", "maile": "male", "mal": "male", "cis male": "male", "man": "male", "msle": "male",
    "female": "female", "femail": "female", "cis female": "female", "woman": "female"
})
df["gender"] = df["gender"].apply(lambda x: "male" if "male" in x else "female" if "female" in x else "other")
df = df[(df["age"] >= 18) & (df["age"] <= 65)]

import numpy as np

df["age"] = df["age"].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["age"])


plt.figure(figsize=(8, 4))
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution (18-65)")
plt.show()

categorical_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
data_encoding = df.copy()
for col in categorical_cols:
    data_encoding[col] = le.fit_transform(data_encoding[col].astype(str))


top_countries = df["country"].value_counts().nlargest(10)
plt.figure(figsize=(10, 4))
sns.barplot(x=top_countries.index, y=top_countries.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Countries")
plt.show()