import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df = df[['Humidity', 'Wind Speed (km/h)', 'Apparent Temperature (C)']]
    df = df.dropna()

    # Проверка типов данных
    if not all(df[col].dtype in [np.float64, np.int64] for col in df.columns):
        raise ValueError("Некорректные типы данных. Ожидались числовые значения.")

    df = df[(df['Humidity'] >= 0) & (df['Humidity'] <= 1)]
    df = df[(df['Wind Speed (km/h)'] >= 0) & (df['Wind Speed (km/h)'] <= 100)]
    df = df[(df['Apparent Temperature (C)'] >= -50) & (df['Apparent Temperature (C)'] <= 50)]
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
    print(f"  Коэффициент (наклон): {model.coef_[0]:.2f}")
    print(f"  Пересечение (intercept): {model.intercept_:.2f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  RMSE: {rmse:.2f}")

    X_test_sorted = X_test.sort_values(by='Humidity')
    y_pred_sorted = model.predict(X_test_sorted)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_test['Humidity'], y=y_test, alpha=0.5, label='Истинные значения')
    sns.lineplot(x=X_test_sorted['Humidity'], y=y_pred_sorted, color='red', linewidth=2, label='Регрессия')
    plt.title('Ощущаемая температура в зависимости от влажности')
    plt.xlabel('Влажность (0-1)')
    plt.ylabel('Ощущаемая температура (°C)')
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
    print(f"  Коэффициенты: Влажность = {model.coef_[0]:.2f}, Ветер = {model.coef_[1]:.2f}")
    print(f"  Пересечение (intercept): {model.intercept_:.2f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  RMSE: {rmse:.2f}")

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

    while True:
        try:
            humidity = float(input("Введите влажность (0.0–1.0): "))
            if not (0 <= humidity <= 1):
                raise ValueError("Влажность должна быть от 0 до 1.")

            wind_speed = float(input("Введите скорость ветра (км/ч, >=0): "))
            if wind_speed < 0:
                raise ValueError("Скорость ветра не может быть отрицательной.")

            break
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")

    # Исправленные строки:
    pred1 = model1.predict(pd.DataFrame([[humidity]], columns=['Humidity']))[0]
    pred2 = model2.predict(pd.DataFrame([[humidity, wind_speed]], columns=['Humidity', 'Wind Speed (km/h)']))[0]

    print(f"\nМодель 1 (только влажность): {pred1:.2f} °C")
    print(f"Модель 2 (влажность + ветер): {pred2:.2f} °C")


    # pred1 = model1.predict([[humidity]])[0]
    # pred2 = model2.predict([[humidity, wind_speed]])[0]


if __name__ == "__main__":
    df = load_data("weatherHistory.csv")
    model1 = model_humidity(df)
    model2 = model_humidity_wind(df)
    console_test(model1, model2)
