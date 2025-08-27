import random
nums = []

while True:
    try:
        variant_vvoda = int(input("Введите 1 для ручного ввода или 2 для случайного ввода: "))
        if variant_vvoda == 1:
            kolvo = int(input("Введите количество чисел для ввода: "))
            for i in range(1, kolvo + 1):
                num = int(input(f'Введите {i}-е число: '))
                nums.append(num)
                break
        elif variant_vvoda == 2:
            kolvo = int(input("Введите количество чисел для генерации: "))
            nums = [random.randint(1, 10000) for i in range(kolvo)]
            break
        else:
            raise ValueError("Неверный выбор! Введите 1 или 2.")
    except ValueError as e:
        print(f"Ошибка: {e}")


print("Правильный порядок: ", nums)
print("Инвертированный: ", list(reversed(nums)))