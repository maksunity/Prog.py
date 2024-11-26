import random


def get_computer_choice():
    """Возвращает случайный выбор компьютера: 'камень', 'ножницы' или 'бумага'."""
    choices = ["камень", "ножницы", "бумага"]
    return random.choice(choices)

def determine_winner(user_choice, computer_choice):
    """
    Определяет победителя игры на основе выбора пользователя и компьютера.

    Параметры:
        user_choice (str): Выбор пользователя.
        computer_choice (str): Выбор компьютера.

    Возвращает:
        str: Результат игры ("Вы выиграли!", "Компьютер выиграл!" или "Ничья!").
    """
    if user_choice == computer_choice:
        return "Ничья!"

    # Правила игры: камень > ножницы, ножницы > бумага, бумага > камень
    winning_combinations = {
        "камень": "ножницы",
        "ножницы": "бумага",
        "бумага": "камень"
    }

    if winning_combinations[user_choice] == computer_choice:
        return "Вы выиграли!"
    else:
        return "Компьютер выиграл!"


def main():
    """Основная функция игры."""
    print("Добро пожаловать в игру 'Камень-ножницы-бумага'!")

    while True:
        user_choice = input("Сделайте выбор (камень, ножницы, бумага) или 'выход' для завершения: ").lower()

        # Проверка на выход из игры
        if user_choice == 'выход':
            print("Спасибо за игру!")
            break

        # Проверка корректности ввода
        if user_choice not in ["камень", "ножницы", "бумага"]:
            print("Ошибка: введите 'камень', 'ножницы' или 'бумага'.")
            continue

        # Получаем выбор компьютера и определяем победителя
        computer_choice = get_computer_choice()
        print(f"Компьютер выбрал: {computer_choice}")

        result = determine_winner(user_choice, computer_choice)
        print(result)


if __name__ == "__main__":
    main()
