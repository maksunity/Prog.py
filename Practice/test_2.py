import simpy
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time

# -------------------
# Конфигурация
# -------------------
T_max_rolls = 400
delta_speed = 10
SIMULATION_STEPS = 20  # количество шагов симуляции

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

    def step(self):
        # Запись метрик
        self.history["time"].append(self.env.now)
        self.history["furnace"].append(self.furnace_temp)
        self.history["ingot"].append(self.ingot_temp)
        self.history["rolls"].append(self.rolls_temp)

        # Логика состояний
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
            pass

        self.env.step()  # имитация шага времени

# -------------------
# Streamlit UI
# -------------------
st.title("Тестовый стенд горячего прокатного стана (SimPy)")

T_furnace = st.slider("Температура печи (°C)", 800, 1300, 1000, step=50)
speed_rolls = st.slider("Скорость роликов (м/с)", 0, 30, 10)
speed_winder = st.slider("Скорость намотки (м/с)", 0, 30, 10)

# -------------------
# Инициализация SimPy
# -------------------
env = simpy.Environment()
mill = RollingMill(env, T_furnace, speed_rolls, speed_winder)

# Контейнеры для динамической визуализации
placeholder_temp = st.empty()
placeholder_speed = st.empty()
status_text = st.empty()

# -------------------
# Симуляция во времени
# -------------------
for _ in range(SIMULATION_STEPS):
    mill.step()

    # Вывод текущего состояния
    status_text.write(f"Текущее состояние стана: **{mill.state}**")

    # Визуализация температур
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].plot(mill.history["time"], mill.history["furnace"], color="orange")
    ax[0].set_title("Температура печи")
    ax[0].set_ylim(0, 1400)

    ax[1].plot(mill.history["time"], mill.history["ingot"], color="red")
    ax[1].set_title("Температура заготовки")
    ax[1].set_ylim(0, 1400)

    ax[2].plot(mill.history["time"], mill.history["rolls"], color="blue")
    ax[2].set_title("Температура роликов")
    ax[2].set_ylim(0, 500)

    placeholder_temp.pyplot(fig)

    # Визуализация скоростей
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(mill.history["time"], [mill.roll_speed]*len(mill.history["time"]), label="Скорость роликов")
    ax2.plot(mill.history["time"], [mill.winder_speed]*len(mill.history["time"]), label="Скорость намотки")
    ax2.set_xlabel("Время (с)")
    ax2.set_ylabel("Скорость (м/с)")
    ax2.legend()
    placeholder_speed.pyplot(fig2)

    # Пауза, чтобы визуально видеть динамику
    time.sleep(0.3)
