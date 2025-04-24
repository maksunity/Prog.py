class DriftCar:
    def __init__(self, model: str, brand: str, engine_volume: float, horsepower: int,
                 race_cage: bool, octane: int, camber_angle: float, tire_width: int):
        self.model = model
        self.brand = brand
        self.engine_volume = engine_volume
        self.horsepower = horsepower
        self.race_cage = race_cage
        self.octane = octane
        self.camber_angle = camber_angle
        self.tire_width = tire_width

    def __str__(self):
        return (f"DriftCar is {self.model} {self.brand}. "
                f"Engine: {self.engine_volume} L, {self.horsepower} HP, "
                f"Race cage: {'Yes' if self.race_cage else 'No'}, Octane: {self.octane}, "
                f"Camber angle: {self.camber_angle}°, Tire width: {self.tire_width} mm")

    def __repr__(self):
        return (f"DriftCar('{self.model}', '{self.brand}', {self.engine_volume}, {self.horsepower}, "
                f"{self.race_cage}, {self.octane}, {self.camber_angle}, {self.tire_width})")

    def __add__(self, angle: float):
        if isinstance(angle, (int, float)):
            return DriftCar(self.model, self.brand, self.engine_volume, self.horsepower,
                            self.race_cage, self.octane, self.camber_angle + angle, self.tire_width)
        return NotImplemented

    def __sub__(self, angle: float):
        if isinstance(angle, (int, float)):
            return DriftCar(self.model, self.brand, self.engine_volume, self.horsepower,
                            self.race_cage, self.octane, self.camber_angle - angle, self.tire_width)
        return NotImplemented



car = DriftCar("Nissan", "GTR-34", 3.0, 400, True, 100, -3.5, 265)

print(f'{car} \n')

car1 = car + 2.0
print(f'After adding camber angle: {car1} \n')

car2 = car - 1.0
print(f'After reducing camber angle: {car2} \n')

