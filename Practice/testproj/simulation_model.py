import time

# Константы окружающей среды
AMBIENT_TEMPERATURE = 25.0  # Температура окружающей среды, °C
TIME_STEP = 1  # Шаг симуляции в секундах

# Пороговые значения для оборудования
BEARING_TEMP_WARNING = 90.0 
BEARING_TEMP_CRITICAL = 120.0

ROLL_TEMP_WARNING = 400.0
ROLL_TEMP_CRITICAL = 500.0


class Equipment: #Класс для всего оборудования
    def __init__(self, name, max_history=200):
        self.name = name
        self.status = "OK"
        self.alerts = []
        self.history = []
        self.max_history = max_history

    def update(self, **kwargs):
        raise NotImplementedError

    def check_alerts(self):
        pass

    def get_state(self):
        return {
            "name": self.name,
            "status": self.status,
            "alerts": self.alerts
        }


class Bearing(Equipment): #Класс для подшипника
    def __init__(self, name, friction_coeff=0.001, cooling_coeff=0.005, roll_heat_coeff=0.01):
        super().__init__(name)
        self.temperature = AMBIENT_TEMPERATURE
        self.friction_coeff = friction_coeff
        self.base_cooling_coeff = cooling_coeff
        self.roll_heat_coeff = roll_heat_coeff

    def update(self, speed=0.0, roll_temperature=AMBIENT_TEMPERATURE, cooling_intensity=1.0, **kwargs):
        """Обновляет температуру на основе скорости, охлаждения и тепла от вала."""
        # Нагрев от трения
        heat_gain_friction = self.friction_coeff * speed
        # Потеря тепла в окружающую среду, скорректированная по интенсивности
        heat_loss_cooling = self.base_cooling_coeff * cooling_intensity * (self.temperature - AMBIENT_TEMPERATURE)
        # Теплообмен с валом
        heat_from_roll = self.roll_heat_coeff * (roll_temperature - self.temperature)

        self.temperature += (heat_gain_friction - heat_loss_cooling + heat_from_roll) * TIME_STEP
        
        self.history.append(round(self.temperature, 2))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self.check_alerts()

    def check_alerts(self):
        self.alerts.clear()
        if self.temperature > BEARING_TEMP_CRITICAL:
            self.status = "CRITICAL"
            self.alerts.append(f"Критическая температура: {self.temperature:.1f}°C")
        elif self.temperature > BEARING_TEMP_WARNING:
            self.status = "WARNING"
            self.alerts.append(f"Высокая температура: {self.temperature:.1f}°C")
        else:
            self.status = "OK"

    def get_state(self):
        state = super().get_state()
        state['temperature'] = round(self.temperature, 2)
        return state


class Roll(Equipment): #Класс для вала
    def __init__(self, name, bearing_heat_coeff=0.02, ingot_heat_coeff=0.05, cooling_coeff=0.01):
        super().__init__(name)
        self.temperature = AMBIENT_TEMPERATURE
        self.bearing = Bearing(f"{name}-Подшипник")
        self.bearing_heat_coeff = bearing_heat_coeff
        self.ingot_heat_coeff = ingot_heat_coeff
        self.base_cooling_coeff = cooling_coeff

    def update(self, speed=0.0, ingot_temperature=None, cooling_intensity=1.0, **kwargs):
        # Подшипник обновляется с использованием температуры вала из *предыдущего* шага.
        self.bearing.update(speed=speed, roll_temperature=self.temperature, cooling_intensity=cooling_intensity)

        # Вал обновляется с использованием новой температуры подшипника.
        heat_from_bearing = self.bearing_heat_coeff * (self.bearing.temperature - self.temperature)
        
        # Логика для горячей подачи
        heat_from_ingot = 0
        if ingot_temperature:
            heat_from_ingot = self.ingot_heat_coeff * (ingot_temperature - self.temperature)
        
        # Потеря тепла от системы охлаждения вала
        heat_loss_cooling = self.base_cooling_coeff * cooling_intensity * (self.temperature - AMBIENT_TEMPERATURE)

        self.temperature += (heat_from_bearing + heat_from_ingot - heat_loss_cooling) * TIME_STEP
        
        self.history.append(round(self.temperature, 2))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self.check_alerts()

    def check_alerts(self):
        self.alerts.clear()
        if self.temperature > ROLL_TEMP_CRITICAL:
            self.status = "CRITICAL"
            self.alerts.append(f"Критическая температура: {self.temperature:.1f}°C")
        elif self.temperature > ROLL_TEMP_WARNING:
            self.status = "WARNING"
            self.alerts.append(f"Высокая температура: {self.temperature:.1f}°C")
        else:
            self.status = "OK"

    def get_state(self):
        state = super().get_state()
        state['temperature'] = round(self.temperature, 2)
        state['bearing_state'] = self.bearing.get_state()
        return state


class RollingMillStand(Equipment):
    def __init__(self, name):
        super().__init__(name)
        self.roll_1 = Roll("Вал-1")
        self.roll_2 = Roll("Вал-2")
        self.speed = 0.0
        self.is_rolling = False
        self.cooling_intensity = 1.0

    def set_speed(self, new_speed):
        self.speed = max(0, new_speed)

    def set_cooling_intensity(self, intensity):
        self.cooling_intensity = max(0, intensity)

    def update(self, ingot_temperature=None, **kwargs):
        current_ingot_temp = ingot_temperature if self.is_rolling else None
        self.roll_1.update(speed=self.speed, ingot_temperature=current_ingot_temp, cooling_intensity=self.cooling_intensity)
        self.roll_2.update(speed=self.speed, ingot_temperature=current_ingot_temp, cooling_intensity=self.cooling_intensity)
        self.check_alerts()

    def check_alerts(self):
        if self.roll_1.status == "CRITICAL" or self.roll_2.status == "CRITICAL":
            self.status = "CRITICAL"
        elif self.roll_1.status == "WARNING" or self.roll_2.status == "WARNING":
            self.status = "WARNING"
        else:
            self.status = "OK"

    def get_state(self):
        state = super().get_state()
        state['speed'] = self.speed
        state['is_rolling'] = self.is_rolling
        state['cooling_intensity'] = self.cooling_intensity
        state['roll_1_state'] = self.roll_1.get_state()
        state['roll_2_state'] = self.roll_2.get_state()
        return state

class Furnace:
    def __init__(self):
        self.target_temperature = 1200.0
    
    def set_temperature(self, temp):
        self.target_temperature = temp

    def get_ingot(self):
        return self.target_temperature
