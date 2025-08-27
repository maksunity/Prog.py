import random


def manual_input(count):
    """Функция для ручного ввода словаря с ключами-строками и значениями-числами."""
    dictionary = {}
    for i in range(1, count + 1):
        key = input(f'Введите ключ {i} (строка): ')
        while True:
            try:
                value = input(f'Введите значение для ключа "{key}" (целое или вещественное число): ')
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
                dictionary[key] = value
                break
            except ValueError:
                print("Ошибка: Введите допустимое число (целое или с плавающей точкой).")
    return dictionary


def random_input(count):
    """Функция для генерации случайного словаря с ключами-строками и значениями-числами."""
    dictionary = {}
    for i in range(count):
        key = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        value = random.choice([random.randint(1, 100), round(random.uniform(1, 100), 2)])
        dictionary[key] = value
    return dictionary


def main():
    while True:
        try:
            variant_vvoda = int(input("Введите 1 для ручного ввода или 2 для случайного ввода: "))
            count = int(input("Введите количество элементов в каждом словаре: "))
            if variant_vvoda == 1:
                dict1 = manual_input(count)
                dict2 = manual_input(count)
                break
            elif variant_vvoda == 2:
                dict1 = random_input(count)
                dict2 = random_input(count)
                break
            else:
                print("Ошибка: Введите 1 или 2.")
        except ValueError as e:
            print(f"Ошибка: {e}. Пожалуйста, введите целое число.")

    # Вывод исходных словарей
    print("Словарь 1:", dict1)
    print("Словарь 2:", dict2)

    # Найти пересечения значений словарей
    values_intersection = set(dict1.values()).intersection(set(dict2.values()))

    # Создать новый словарь с парами ключ-значение, где значения входят в пересечение
    result_dict = {k: v for k, v in dict1.items() if v in values_intersection}
    result_dict.update({k: v for k, v in dict2.items() if v in values_intersection and k not in result_dict})

    # Вывод словаря с пересечениями
    print("Новый словарь с пересечениями значений:", result_dict)


if __name__ == "__main__":
    main()
