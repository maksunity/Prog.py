class Queue:
    def __init__(self):
        self.queue = list()

    def add (self, elem):
        self.queue.append(elem)

    def delete(self):
        if len(self.queue) == 0:
            return None
        else:
            removed_self = self.queue.pop(0)
            return removed_self

    def print (self):
        return self.queue


def main():
    Qlist = Queue()
    while True:
        try:
            choose = int(input("Выберите действие с очередью: \n "
                               "1 - Добавление элемента; \n "
                               "2 - Удаление элемента \n "
                               "3 - Вывод очереди \n "
                               "4 - завершение программы \n "
                               "Ваш выбор: "))
            if choose == 1:
                var = input("Введите элемент для добавления: ")
                Qlist.add(var)
            elif choose == 2:
                removed = Qlist.delete()
                if removed is None:
                    print("Очередь пуста, удалять нечего!")
                else:
                    print(f"Удалён первый элемент: {removed}")
            elif choose == 3:
                print(f"Ваша очередь: {Qlist.print()}")
            elif choose == 4:
                break
            else:
                print("Ошибка, повторите корректный ввод!")
        except ValueError as e:
            print(f"Ошибка: {e}. Пожалуйста, введите целое число.")



if __name__ == "__main__":
    main()