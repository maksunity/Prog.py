import random

# Функция для создания начальной популяции точек (x, y)
def initialize_population(size: int, lower_bound: float) -> list[tuple[float, float]]:
    # Генерируем список из 'size' точек с координатами в диапазоне [lower_bound, lower_bound + 1.5]
    return [
        (random.uniform(lower_bound, lower_bound + 1.5),
         random.uniform(lower_bound, lower_bound + 1.5))
        for _ in range(size)
    ]

# Функция для мутации точки с добавлением случайного шума
def mutate_point(point: tuple[float, float], step: float, lower_bound: float) -> tuple[float, float]:
    x, y = point
    # Добавляем случайный шум в диапазоне [-step, step] к каждой координате
    x_new = max(lower_bound, x + random.uniform(-step, step))
    y_new = max(lower_bound, y + random.uniform(-step, step))

    return x_new, y_new

# Функция для оценки популяции
def evaluate_population(pop: list[tuple[float, float]]) -> list[tuple[float, float, float, float]]:
    evaluated = []
    for x, y in pop:
        # Вычисляем значение функции k = 3x^2 + 2y^2
        k_val = 3 * x**2 + 2 * y**2
        # Вычисляем отклонение от точки
        deviation = abs(x - 0.5) + abs(y - 0.5)
        evaluated.append((x, y, k_val, deviation))

    return evaluated

# Функция для выбора лучших точек из оценённой популяции
def select_best(evaluated: list[tuple[float, float, float, float]], keep_fraction: float) -> list[tuple[float, float]]:
    evaluated.sort(key=lambda item: (item[3], item[2]))
    cutoff = int(len(evaluated) * keep_fraction)

    return [(x, y) for x, y, _, _ in evaluated[:cutoff]]

# Основная функция для выполнения эволюционного поиска
def evolutionary_search(
    pop_size: int = 100,
    mutation_step: float = 0.05,
    lower_bound: float = 0.5,
    epsilon: float = 0.001,
    keep_fraction: float = 0.5
) -> tuple[float, float, int]:
    # Инициализируем начальную популяцию
    population = initialize_population(pop_size, lower_bound)

    generation = 0

    while True:
        generation += 1

        # Создаём потомков путём мутации случайных точек из текущей популяции
        offspring = [mutate_point(random.choice(population), mutation_step, lower_bound)
                     for _ in range(pop_size)]

        # Объединяем текущую популяцию с потомками
        combined = population + offspring
        evaluated = evaluate_population(combined)

        # Проверяем, достигнута ли цель (отклонение меньше epsilon)
        for x, y, k_val, deviation in evaluated:
            if deviation < epsilon:

                return x, y, generation

        # Если цель не достигнута, выбираем лучших особей для следующего поколения
        population = select_best(evaluated, keep_fraction)

if __name__ == "__main__":
    best_x, best_y, total_gens = evolutionary_search()
    print(f"Найдено оптимальное решение: x={best_x:.4f}, y={best_y:.4f} за {total_gens} поколений.")
