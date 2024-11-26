import random


def choose_secret_word():
    """Возвращает случайное 5-буквенное слово из списка возможных слов."""
    word_list = ["apple", "grape", "lemon", "peach", "melon"]  # Можно расширить список
    return random.choice(word_list)


def evaluate_guess(secret, guess):
    """
    Оценивает догадку и возвращает строку с обозначениями:
    - [буква] если буква и позиция правильные,
    - (буква) если буква правильная, но позиция неверная,
    - просто буква, если буква не присутствует в секретном слове.

    Параметры:
        secret (str): Секретное слово.
        guess (str): Догадка игрока.

    Возвращает:
        str: Оценка догадки.
    """
    result = []
    used_positions = set()  # Для отслеживания правильных позиций

    # Шаг 1: Отмечаем все правильные буквы на правильных позициях (быки)
    for i, char in enumerate(guess):
        if char == secret[i]:
            result.append(f"[{char}]")
            used_positions.add(i)
        else:
            result.append(None)  # Заполняем пробелом для дальнейших шагов

    # Шаг 2: Отмечаем правильные буквы на неправильных позициях (коровы)
    for i, char in enumerate(guess):
        if result[i] is None:  # Если еще не отмечено
            if char in secret and secret.index(char) not in used_positions:
                result[i] = f"({char})"
                used_positions.add(secret.index(char))
            else:
                result[i] = char  # Обычная буква, если она отсутствует в секретном слове

    return ''.join(result)


def main():
    """Основная функция игры."""
    print("Добро пожаловать в игру 'Wordle'!")
    print(
        "Попробуйте угадать 5-буквенное слово. Правильная буква на правильной позиции обозначается [буквой], правильная буква на неправильной позиции обозначается (буквой).")

    secret_word = choose_secret_word()
    attempts = 0

    while True:
        guess = input("Введите вашу догадку (5 букв): ").lower()

        # Проверка корректности ввода
        if len(guess) != 5 or not guess.isalpha():
            print("Ошибка: введите слово из 5 букв.")
            continue

        attempts += 1
        result = evaluate_guess(secret_word, guess)
        print("Результат:", result)

        if guess == secret_word:
            print(f"Поздравляем! Вы угадали слово '{secret_word}' за {attempts} попыток.")
            break


if __name__ == "__main__":
    main()
