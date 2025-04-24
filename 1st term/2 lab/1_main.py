import json

class Task:
    def __init__(self, description, category):
        self.description = description
        self.category = category
        self.completed = False

    def mark_completed(self):
        self.completed = True

    def to_dict(self):
        return {
            "description": self.description,
            "category": self.category,
            "completed": self.completed,
        }

    @staticmethod
    def from_dict(data):
        task = Task(data["description"], data["category"])
        task.completed = data["completed"]
        return task


class TaskTracker:
    def __init__(self, filename="tasks.json"):
        self.tasks = []
        self.filename = filename
        self.load_tasks()

    def add_task(self, description, category):
        self.tasks.append(Task(description, category))

    def mark_task_completed(self, task_index):
        if 0 <= task_index < len(self.tasks):
            self.tasks[task_index].mark_completed()
            print(f"Задача {task_index} отмечена как выполненная.")
        else:
            print("Неверный индекс задачи.")

    def find_tasks(self, keyword):
        return [task for task in self.tasks if keyword.lower() in task.description.lower()]

    def list_tasks_by_category(self, category):
        return [task for task in self.tasks if task.category.lower() == category.lower()]

    def list_tasks(self):
        for index, task in enumerate(self.tasks):
            status = "✓" if task.completed else "✗"
            print(f"{index}: [{status}] {task.description} (Категория: {task.category})")

    def save_tasks(self):
        with open(self.filename, "w") as file:
            json.dump([task.to_dict() for task in self.tasks], file, indent=4)

    def load_tasks(self):
        try:
            with open(self.filename, "r") as file:
                tasks_data = json.load(file)
                self.tasks = [Task.from_dict(data) for data in tasks_data]
        except FileNotFoundError:
            pass


def main():
    tracker = TaskTracker()

    while True:
        print("\nМеню:")
        print("1. Добавить задачу")
        print("2. Отметить задачу как выполненную")
        print("3. Вывести все задачи")
        print("4. Найти задачу по имени")
        print("5. Вывести все задачи из категории")
        print("6. Выход и сохранение файла")
        choice = input("Введите команду: ")

        if choice == "1":
            description = input("Введите название задачи: ")
            category = input("Введите категорию задачи: ")
            tracker.add_task(description, category)
        elif choice == "2":
            task_index = int(input("Введите индекс задачи для отметки её как выполненной: "))
            tracker.mark_task_completed(task_index)
        elif choice == "3":
            tracker.list_tasks()
        elif choice == "4":
            keyword = input("Введите название задачи: ")
            found_tasks = tracker.find_tasks(keyword)
            for task in found_tasks:
                status = "✓" if task.completed else "✗"
                print(f"[{status}] {task.description} (Категория: {task.category})")
        elif choice == "5":
            category = input("Введите категорию для поиска: ")
            category_tasks = tracker.list_tasks_by_category(category)
            for task in category_tasks:
                status = "✓" if task.completed else "✗"
                print(f"[{status}] {task.description} (Категория: {task.category})")
        elif choice == "6":
            tracker.save_tasks()
            print("Файл успешно сохранен! До свидиния!")
            break
        else:
            print("Некорректный ввод. Повторите попытку!")


if __name__ == "__main__":
    main()
