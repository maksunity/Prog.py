import time
import os
import json
from simulation_model import RollingMillStand, Furnace

def print_system_state(mill):
    os.system('cls' if os.name == 'nt' else 'clear')
    state = mill.get_state()
    print("--- Симуляция прокатного стана ---")
    print(json.dumps(state, indent=2))
    print("\n--- Оповещения ---")
    if mill.status != "OK":
        print(f"Прокатный стан '{mill.name}': {mill.status}")
    
    for roll_name in ["roll_1_state", "roll_2_state"]:
        roll = state[roll_name]
        if roll['status'] != "OK":
            print(f"- {roll['name']}: {roll['status']} -> {roll['alerts'][0]}")
        
        bearing = roll['bearing_state']
        if bearing['status'] != "OK":
            print(f"  - {bearing['name']}: {bearing['status']} -> {bearing['alerts'][0]}")
    print("\n-----------------------------")


def run_simulation():
    mill = RollingMillStand("Mill-Stand-1")
    furnace = Furnace()
    furnace.set_temperature(1150)

    try:
        print_system_state(mill)
        time.sleep(2)

        print("СЦЕНАРИЙ: Запуск стана на 200 об/мин...")
        mill.set_speed(200)
        for _ in range(10):
            mill.update()
            print_system_state(mill)
            time.sleep(1)

        print("СЦЕНАРИЙ: Начинается прокатка горячего слитка (1150°C)...")
        time.sleep(2)
        mill.is_rolling = True
        for _ in range(15):
            ingot_temp = furnace.get_ingot()
            mill.update(ingot_temperature=ingot_temp)
            print_system_state(mill)
            time.sleep(1)

        print("СЦЕНАРИЙ: Увеличение скорости до 450 об/мин...")
        time.sleep(2)
        mill.set_speed(450)
        for _ in range(20):
            ingot_temp = furnace.get_ingot()
            mill.update(ingot_temperature=ingot_temp)
            print_system_state(mill)
            if mill.status == "CRITICAL":
                print("!!! ОБНАРУЖЕН КРИТИЧЕСКИЙ СБОЙ. ОСТАНОВКА. !!!")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nСимуляция остановлена пользователем.")
    finally:
        mill.set_speed(0)
        mill.is_rolling = False
        print("СЦЕНАРИЙ: Стан остановлен.")
        print_system_state(mill)


if __name__ == "__main__":
    run_simulation()
