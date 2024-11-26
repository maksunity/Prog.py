import random


def manual_input(count):
    numbers = []
    for i in range(1, count + 1):
        while True:
            try:
                num = int(input(f'Введите {i}-е число: '))
                numbers.append(num)
                break
            except ValueError:
                print("Ошибка: Введите целое число.")
    return numbers


def random_input(count):
    return [random.randint(1, 10000) for _ in range(count)]


def main():
    while True:
        try:
            variant_vvoda = int(input("Введите 1 для ручного ввода или 2 для случайного ввода: "))
            if variant_vvoda == 1:
                kolvo = int(input("Введите количество чисел для ввода: "))
                list1 = manual_input(kolvo)
                list2 = manual_input(kolvo)
                break
            elif variant_vvoda == 2:
                kolvo = int(input("Введите количество чисел для генерации: "))
                list1 = random_input(kolvo)
                list2 = random_input(kolvo)
                break
            else:
                print("Ошибка: Введите 1 или 2.")
        except ValueError as e:
            print(f"Ошибка: {e}. Пожалуйста, введите целое число.")

    print("Список 1:", list1)
    print("Список 2:", list2)
    list3 = [list1[i] for i in range(1, len(list1), 2)] + [list2[i] for i in range(0, len(list2), 2)]

    print("Список 3:", list3)


if __name__ == "__main__":
    main()
