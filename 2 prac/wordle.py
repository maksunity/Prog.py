import random


def choose_secret_word():
    word_list = ["apple", "grape", "lemon", "peach", "melon"]  # Можно расширить список
    return random.choice(word_list)


def evaluate_guess(secret, guess):
    result = []
    used_positions = set()

    # Быки
    for i, char in enumerate(guess):
        if char == secret[i]:
            result.append(f"[{char}]")
            used_positions.add(i)
        else:
            result.append(None)

    # Коровы
    for i, char in enumerate(guess):
        if result[i] is None:
            if char in secret and secret.index(char) not in used_positions:
                result[i] = f"({char})"
                used_positions.add(secret.index(char))
            else:
                result[i] = char  # Обычная буква, если она отсутствует в секретном слове

    return ''.join(result)


def main():
    print("Добро пожаловать в игру 'Wordle'!")
    print(
        "Попробуйте угадать 5-буквенное слово. Правильная буква на правильной позиции обозначается [], правильная буква на неправильной позиции обозначается ().")

    secret_word = choose_secret_word()
    attempts = 0

    while True:
        guess = input("Введите вашу догадку (5 букв): ").lower()

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
