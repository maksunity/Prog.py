import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(path):
    df = pd.read_csv(path)
    df = df[['Humidity', 'Wind Speed (km/h)', 'Apparent Temperature (C)']]
    df = df.dropna()
    df = df[(df['Humidity'] >= 0) & (df['Humidity'] <= 1)]
    return df

#1: температура по влажности
def model_humidity(df):
    X = df[['Humidity']]
    y = df['Apparent Temperature (C)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Model 1 — по влажности:")
    print("  Coefficients:", model.coef_)
    print("  Intercept:", model.intercept_)
    print("  RMSE:", mean_squared_error(y_test, y_pred, squared=False))

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_test['Humidity'], y=y_test, label='Истинные значения')
    sns.lineplot(x=X_test['Humidity'], y=y_pred, color='red', label='Регрессия')
    plt.title('Ощущаемая температура от влажности')
    plt.xlabel('Влажность')
    plt.ylabel('Ощущаемая температура (C)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

#2: температура по влажности и скорости ветра
def model_humidity_wind(df):
    X = df[['Humidity', 'Wind Speed (km/h)']]
    y = df['Apparent Temperature (C)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel 2 — по влажности и скорости ветра:")
    print("  Coefficients:", model.coef_)
    print("  Intercept:", model.intercept_)
    print("  RMSE:", mean_squared_error(y_test, y_pred, squared=False))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_test['Humidity'], y=y_test, hue=X_test['Wind Speed (km/h)'], palette='cool', legend=False)
    sns.scatterplot(x=X_test['Humidity'], y=y_pred, color='red', label='Предсказания')
    plt.title('Ощущаемая температура от влажности и ветра')
    plt.xlabel('Влажность')
    plt.ylabel('Ощущаемая температура (C)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return model

def console_test(model1, model2):
    print("\nКонсольный тестовый стенд")
    humidity = float(input("Введите влажность (0.0–1.0): "))
    wind_speed = float(input("Введите скорость ветра (км/ч): "))

    pred1 = model1.predict([[humidity]])[0]
    pred2 = model2.predict([[humidity, wind_speed]])[0]

    print(f"\nМодель 1 (только влажность): {pred1:.2f} °C")
    print(f"Модель 2 (влажность + ветер): {pred2:.2f} °C")

if __name__ == "__main__":
    df = load_data("weatherHistory.csv")
    model1 = model_humidity(df)
    model2 = model_humidity_wind(df)
    console_test(model1, model2)
