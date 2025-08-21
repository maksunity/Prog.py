import json


class Transaction:
    def __init__(self, description, amount, category):
        self.description = description
        self.amount = amount
        self.category = category

    def to_dict(self):
        return {
            "description": self.description,
            "amount": self.amount,
            "category": self.category,
        }

    @staticmethod
    def from_dict(data):
        return Transaction(data["description"], data["amount"], data["category"])


class BudgetTracker:
    def __init__(self, filename="budget.json"):
        self.transactions = []
        self.limits = {}
        self.filename = filename
        self.load_data()

    def add_transaction(self, description, amount, category):
        self.transactions.append(Transaction(description, amount, category))
        print("Транзакция добавлена.")

    def set_limit(self, category, limit):
        self.limits[category] = limit
        print(f"Лимит для категории '{category}' установлен: {limit}.")

    def get_category_total(self, category):
        return sum(
            transaction.amount
            for transaction in self.transactions
            if transaction.category == category
        )

    def check_limits(self):
        for category, limit in self.limits.items():
            total = self.get_category_total(category)
            if total > limit:
                print(
                    f"Внимание: категория '{category}' превышает лимит на {total - limit}."
                )

    def list_transactions(self):
        if not self.transactions:
            print("Транзакции отсутствуют.")
            return
        for transaction in self.transactions:
            print(
                f"{transaction.description} | {transaction.amount} | {transaction.category}"
            )

    def save_data(self):
        data = {
            "transactions": [t.to_dict() for t in self.transactions],
            "limits": self.limits,
        }
        with open(self.filename, "w") as file:
            json.dump(data, file, indent=4)
        print("Данные сохранены.")

    def load_data(self):
        try:
            with open(self.filename, "r") as file:
                data = json.load(file)
                self.transactions = [
                    Transaction.from_dict(t) for t in data.get("transactions", [])
                ]
                self.limits = data.get("limits", {})
                print("Данные загружены.")
        except FileNotFoundError:
            print("Файл данных не найден, начинаем с пустой базы.")


def main():
    tracker = BudgetTracker()

    while True:
        print("\nМеню трекера бюджета:")
        print("1. Добавить транзакцию")
        print("2. Установить лимит для категории")
        print("3. Показать все транзакции")
        print("4. Проверить лимиты по категориям")
        print("5. Выйти")
        choice = input("Введите номер действия: ")

        if choice == "1":
            description = input("Введите описание транзакции: ")
            amount = float(
                input("Введите сумму (положительное для дохода, отрицательное для расхода): ")
            )
            category = input("Введите категорию: ")
            tracker.add_transaction(description, amount, category)
        elif choice == "2":
            category = input("Введите категорию для установки лимита: ")
            limit = float(input("Введите лимит для категории: "))
            tracker.set_limit(category, limit)
        elif choice == "3":
            tracker.list_transactions()
        elif choice == "4":
            tracker.check_limits()
        elif choice == "5":
            tracker.save_data()
            print("До свидания!")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
