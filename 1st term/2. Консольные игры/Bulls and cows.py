import random


def generate_secret_number():
    """Генерирует случайное 4-значное число с уникальными цифрами."""
    digits = random.sample(range(10), 4)
    return ''.join(map(str, digits))


def get_bulls_and_cows(secret, guess):
    """
    Определяет количество быков и коров между секретным числом и догадкой.

    Параметры:
        secret (str): Секретное число, загаданное компьютером.
        guess (str): Число, предложенное игроком.

    Возвращает:
        tuple: Количество быков и коров (bulls, cows).
    """
    cows = sum(s == g for s, g in zip(secret, guess))
    bulls = sum(min(secret.count(d), guess.count(d)) for d in set(guess)) - cows
    return bulls, cows


def main():
    """Основная функция игры."""
    print("Добро пожаловать в игру 'Быки и коровы'!")
    print("Компьютер загадал 4-значное число с уникальными цифрами. Попробуйте угадать его.")

    secret_number = generate_secret_number()
    attempts = 0
    print(secret_number)

    while True:
        guess = input("Введите ваше предположение (4 уникальных цифры): ")

        # Проверка корректности ввода
        if len(guess) != 4 or not guess.isdigit() or len(set(guess)) != 4:
            print("Ошибка: введите 4-значное число с уникальными цифрами.")
            continue

        attempts += 1
        bulls, cows = get_bulls_and_cows(secret_number, guess)

        print(f"Быки: {bulls}, Коровы: {cows}")

        if cows == 4:
            print(f"Поздравляем! Вы угадали число {secret_number} за {attempts} попыток.")
            break


if __name__ == "__main__":
    main()
