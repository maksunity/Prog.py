import random


def get_computer_choice():
    choices = ["камень", "ножницы", "бумага"]
    return random.choice(choices)

def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Ничья!"

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
    print("Добро пожаловать в игру 'Камень-ножницы-бумага'!")

    while True:
        user_choice = input("Сделайте выбор (камень, ножницы, бумага) или 'выход' для завершения: ").lower()

        if user_choice == 'выход':
            print("Спасибо за игру!")
            break

        if user_choice not in ["камень", "ножницы", "бумага"]:
            print("Ошибка: введите 'камень', 'ножницы' или 'бумага'.")
            continue

        computer_choice = get_computer_choice()
        print(f"Компьютер выбрал: {computer_choice}")

        result = determine_winner(user_choice, computer_choice)
        print(result)


if __name__ == "__main__":
    main()
