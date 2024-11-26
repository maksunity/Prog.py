import random


def manual_input(count):
    elements = []
    for i in range(1, count + 1):
        element = input(f'Введите {i}-й элемент (может быть числом или строкой): ')
        try:
            if '.' in element:
                element = float(element)
            else:
                element = int(element)
        except ValueError:
            pass
        elements.append(element)
    return elements


def random_input(count):
    elements = []
    for _ in range(count):
        choice = random.choice(['int', 'float', 'str'])
        if choice == 'int':
            elements.append(random.randint(1, 100))
        elif choice == 'float':
            elements.append(round(random.uniform(1, 100), 2))
        elif choice == 'str':
            elements.append(''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5)))
    return elements


def main():
    while True:
        try:
            variant_vvoda = int(input("Введите 1 для ручного ввода или 2 для случайного ввода: "))
            kolvo = int(input("Введите количество элементов: "))
            if variant_vvoda == 1:
                elements = manual_input(kolvo)
                break
            elif variant_vvoda == 2:
                elements = random_input(kolvo)
                break
            else:
                print("Ошибка: Введите 1 или 2.")
        except ValueError as e:
            print(f"Ошибка: {e}. Пожалуйста, введите целое число.")

    # Вывод исходного списка
    print("Исходный список:", elements)

    # Удаление дубликатов
    unique_elements = list(set(elements))

    # Вывод списка без дубликатов
    print("Список без дубликатов:", unique_elements)


if __name__ == "__main__":
    main()
