import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("HTRU_2.csv", header=None)
df.columns = [
    'Mean_IP', 'SD_IP', 'EK_IP', 'Skew_IP',
    'Mean_DM', 'SD_DM', 'EK_DM', 'Skew_DM', 'target'
]

print(df.head())
print(df.info())
print(df['target'].value_counts())

sns.pairplot(df.sample(300), hue="target", palette="husl")
plt.suptitle("Pairplot для HTRU2", y=1.02)
plt.show()

# Разделение данных
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Классификация — случайный лес
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Точность:", accuracy_score(y_test, y_pred))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))
print("\nОтчет:\n", classification_report(y_test, y_pred))