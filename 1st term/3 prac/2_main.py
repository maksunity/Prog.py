class Stack:
    def __init__(self):
        self.stack = list()

    def add (self, elem):
        self.stack.append(elem)

    def delete(self):
        if len(self.stack) == 0:
            return None
        else:
            removed_self = self.stack.pop()
            return removed_self

    def print (self):
        return self.stack


def main():
    Slist = Stack()
    while True:
        try:
            choose = int(input("Выберите действие со стеком: \n "
                               "1 - Добавление элемента; \n "
                               "2 - Удаление элемента \n "
                               "3 - Вывод очереди \n "
                               "4 - завершение программы \n "
                               "Ваш выбор: "))
            if choose == 1:
                var = input("Введите элемент для добавления: ")
                Slist.add(var)
            elif choose == 2:
                removed = Slist.delete()
                if removed is None:
                    print("Стек пуст, удалять нечего!")
                else:
                    print(f"Удалён первый элемент: {removed}")
            elif choose == 3:
                print(f"Ваш стек: {Slist.print()}")
            elif choose == 4:
                break
            else:
                print("Ошибка, повторите корректный ввод!")
        except ValueError as e:
            print(f"Ошибка: {e}. Пожалуйста, введите целое число.")



if __name__ == "__main__":
    main()