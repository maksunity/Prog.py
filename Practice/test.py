import simpy
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------
# Конфигурация
# -------------------
T_max_rolls = 400
delta_speed = 10

# -------------------
# SimPy процессы
# -------------------
class RollingMill:
    def __init__(self, env, furnace_temp, roll_speed, winder_speed):
        self.env = env
        self.furnace_temp = furnace_temp
        self.roll_speed = roll_speed
        self.winder_speed = winder_speed

        self.ingot_temp = furnace_temp
        self.rolls_temp = 100
        self.state = "IDLE"
        self.history = {"time": [], "furnace": [], "ingot": [], "rolls": []}

        env.process(self.run())

    def run(self):
        while True:
            # Запись метрик
            self.history["time"].append(self.env.now)
            self.history["furnace"].append(self.furnace_temp)
            self.history["ingot"].append(self.ingot_temp)
            self.history["rolls"].append(self.rolls_temp)

            # Логика
            if self.state == "IDLE":
                self.state = "HEATING"
            elif self.state == "HEATING":
                self.ingot_temp = self.furnace_temp - np.random.uniform(20, 50)
                self.state = "ROLLING"
            elif self.state == "ROLLING":
                self.rolls_temp = 100 + 0.1 * self.roll_speed + 0.05 * self.ingot_temp
                if self.rolls_temp > T_max_rolls:
                    self.state = "FAULT"
            elif self.state == "FAULT":
                break

            yield self.env.timeout(1)  # шаг моделирования = 1 сек

# -------------------
# Streamlit UI
# -------------------
st.title("Тестовый стенд горячего прокатного стана (SimPy)")

T_furnace = st.slider("Температура печи (°C)", 800, 1300, 1000, step=50)
speed_rolls = st.slider("Скорость роликов (м/с)", 0, 30, 10)
speed_winder = st.slider("Скорость намотки (м/с)", 0, 30, 10)

# -------------------
# Запуск симуляции
# -------------------
env = simpy.Environment()
mill = RollingMill(env, T_furnace, speed_rolls, speed_winder)
env.run(until=20)

st.write(f"Текущее состояние стана: **{mill.state}**")

# -------------------
# Визуализация
# -------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

ax[0].plot(mill.history["time"], mill.history["furnace"], color="orange")
ax[0].set_title("Температура печи")

ax[1].plot(mill.history["time"], mill.history["ingot"], color="red")
ax[1].set_title("Температура заготовки")

ax[2].plot(mill.history["time"], mill.history["rolls"], color="blue")
ax[2].set_title("Температура роликов")

st.pyplot(fig)

# -------------------
# Графики скоростей
# -------------------
time_axis = np.arange(0, 10, 1)
roll_speeds = np.ones_like(time_axis) * speed_rolls
winder_speeds = np.ones_like(time_axis) * speed_winder

plt.figure(figsize=(8, 3))
plt.plot(time_axis, roll_speeds, label="Скорость роликов")
plt.plot(time_axis, winder_speeds, label="Скорость намотки")
plt.xlabel("Время (с)")
plt.ylabel("Скорость (м/с)")
plt.legend()
st.pyplot(plt)
