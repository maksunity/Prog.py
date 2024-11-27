class Car:
    def __init__(self, model: str, brand: str, engine_volume: float, horsepower: int):
        self.model = model
        self.brand = brand
        self.engine_volume = engine_volume  # литры
        self.horsepower = horsepower  # л.с.

    def __str__(self):
        return (f"Car is {self.model} {self.brand}. "
                f"Engine volume: {self.engine_volume} L, Power: {self.horsepower} HP")

    def __repr__(self):
        return f"Car('{self.model}', '{self.brand}', {self.engine_volume}, {self.horsepower})"

    def __eq__(self, other):
        if isinstance(other, Car):
            return self.horsepower == other.horsepower
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, Car):
            return self.horsepower < other.horsepower
        return NotImplemented

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

class RacingCar(Car):
    def __init__(self, model: str, brand: str, engine_volume: float, horsepower: int, race_cage: bool, octane: int):
        super().__init__(model, brand, engine_volume, horsepower)
        self.race_cage = race_cage  # Наличие гоночного каркаса
        self.octane = octane  # Октановое число топлива

    def __str__(self):
        return (f"RacingCar is {self.model} {self.brand}. "
                f"Engine volume: {self.engine_volume} L, Power: {self.horsepower} HP, "
                f"Race cage: {'Yes' if self.race_cage else 'No'}, Octane: {self.octane}")

    def __repr__(self):
        return (f"RacingCar('{self.model}', '{self.brand}', {self.engine_volume}, "
                f"{self.horsepower}, {self.race_cage}, {self.octane})")


class DriftCar(RacingCar):
    def __init__(self, model: str, brand: str, engine_volume: float, horsepower: int,
                 race_cage: bool, octane: int, camber_angle: float, tire_width: int):
        super().__init__(model, brand, engine_volume, horsepower, race_cage, octane)
        self.camber_angle = camber_angle  # Угол развала колес
        self.tire_width = tire_width  # Ширина покрышки (мм)

    def __str__(self):
        return (f"DriftCar is {self.model} {self.brand}. "
                f"Engine volume: {self.engine_volume} L, Power: {self.horsepower} HP, "
                f"Race cage: {'Yes' if self.race_cage else 'No'}, Octane: {self.octane}, "
                f"Camber angle: {self.camber_angle}°, Tire width: {self.tire_width} mm")

    def __repr__(self):
        return (f"DriftCar('{self.model}', '{self.brand}', {self.engine_volume}, {self.horsepower}, "
                f"{self.race_cage}, {self.octane}, {self.camber_angle}, {self.tire_width})")


# Пример использования
car1 = Car("Nissan", "350z", 2.0, 150)
car2 = RacingCar("Honda", "NSX", 4.2, 660, False, 98)
car3 = DriftCar("Nissan", "GTR-34", 3.5, 1100, True, 100, -3.5, 280)
#
# print(f'{car1} \n')  # Вывод через __str__
# print(repr(car1))  # Вывод через __repr__
#
# print(car2)  # Вывод через __str__
# print(repr(car2))  # Вывод через __repr__
#
# print(car3)  # Вывод через __str__
# print(repr(car3))  # Вывод через __repr__
#
#
print(f'{car1 < car2} {car1.horsepower}, {car2.horsepower}') # True, сравнение по horsepower
print(f'{car3 > car1} {car3.horsepower}, {car2.horsepower}')  # True
print(f'{car2 == car3} {car2.horsepower}, {car3.horsepower}')  # False
print(f'{car1 != car3} {car1.horsepower}, {car3.horsepower}')  # True

print(f'{car1} \n')
print(f'{repr(car1)} \n')

print(f'{car2} \n')
print(f'{repr(car2)} \n')

print(f'{car3} \n')
print(f'{repr(car3)} \n')

