import random


def manual_input(count):
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
            count = int(input("Введите количество элементов в словаре: "))
            if variant_vvoda == 1:
                dictionary = manual_input(count)
                break
            elif variant_vvoda == 2:
                dictionary = random_input(count)
                break
            else:
                print("Ошибка: Введите 1 или 2.")
        except ValueError as e:
            print(f"Ошибка: {e}. Пожалуйста, введите целое число.")

    # Вывод исходного словаря
    print("Исходный словарь:", dictionary)

    # Создание списка кортежей для уникальных значений
    unique_values = {}
    for key, value in dictionary.items():
        if value not in unique_values:
            unique_values[value] = []
        unique_values[value].append(key)

    # Формирование списка кортежей
    result = [(value, keys) for value, keys in unique_values.items()]

    # Вывод списка кортежей
    print("Список кортежей (значение, [связанные ключи]):", result)


if __name__ == "__main__":
    main()
