def find_divisors_and_check_prime(number: int) -> dict:
    """
    Возвращает список делителей числа и определяет, является ли число простым.

    Параметры:
    number (int): Целое число больше 0, до 100 000.

    Возвращает:
    dict: Словарь с ключами 'divisors' (список делителей) и 'is_prime' (логическое значение).
    """
    if number <= 0 or number > 100_000:
        raise ValueError("Число должно быть больше 0 и не превышать 100 000.")

    divisors = [i for i in range(1, number + 1) if number % i == 0]
    is_prime = len(divisors) == 2

    return {'divisors': divisors, 'is_prime': is_prime}


if __name__ == "__main__":
    try:
        user_input = int(input("Введите целое число больше 0 и до 100 000: "))
        result = find_divisors_and_check_prime(user_input)
        print(f"Делители числа {user_input}: {result['divisors']}")
        if result['is_prime']:
            print(f"{user_input} является простым числом.")
        else:
            print(f"{user_input} не является простым числом.")
    except ValueError as e:
        print(f"Ошибка: {e}")
