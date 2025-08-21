import neat
import numpy as np
import os
import pickle
import visualize
from torchvision import datasets
from PIL import Image, ImageDraw
from neat.parallel import ParallelEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Параметры ---
CONFIG_PATH = 'config-feedforward.txt'
# IMAGE_SIZE = (8, 8)
IMAGE_SIZE = (16, 16)
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_GENERATIONS = 60
# batch_size = 1000
OUTPUT_DIR = 'neat_output_mnist_parallel'


class MNISTDataHolder:
    def __init__(self):
        self.train_images, self.train_labels = self._load_data(train=True)
        self.test_images, self.test_labels = self._load_data(train=False)

    def _load_data(self, train=True):
        dataset = datasets.MNIST(
            root='./mnist_data',
            train=train,
            download=True,
            transform=None
        )

        images = []
        labels = []
        for img, label in dataset:
            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_np = np.array(img_resized) / 255.0
            images.append(img_np.flatten())
            labels.append(label)

        return np.array(images), np.array(labels)


# Инициализация данных в главном процессе
data_holder = MNISTDataHolder()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# def eval_single_genome_mnist(genome, config):
#     try:
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         correct = 0
#         total = len(data_holder.train_images)
#
#         for i in range(total):
#             outputs = net.activate(data_holder.train_images[i])
#             if np.argmax(outputs) == data_holder.train_labels[i]:
#                 correct += 1
#
#         return correct / total
#     except Exception as e:
#         print(f"Ошибка оценки генома: {e}")
#         return 0.0

def eval_single_genome_mnist(genome, config):
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Используем 70% данных для ускорения оценки
        subset_size = int(len(data_holder.train_images) * 0.5)
        # subset_size = int(len(data_holder.train_images))
        indices = np.random.choice(len(data_holder.train_images), subset_size)
        # indices = np.random.choice(len(data_holder.train_images), batch_size)

        correct = sum(
            np.argmax(net.activate(data_holder.train_images[i])) == data_holder.train_labels[i]
            for i in indices
        )
        return correct / subset_size
    except Exception as e:
        print(f"Ошибка оценки генома: {e}")
        return 0.0


def run_neat_mnist_parallel(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Проверка параметров конфигурации
    if config.genome_config.num_inputs != NUM_INPUTS:
        raise ValueError(f"Несоответствие входов: {config.genome_config.num_inputs} vs {NUM_INPUTS}")

    # Создание популяции
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Настройка параллельной оценки
    num_workers = min(os.cpu_count(), 8)  # Ограничение для Windows
    pe = ParallelEvaluator(num_workers, eval_single_genome_mnist)

    print(f"\nЗапуск NEAT для MNIST с {NUM_GENERATIONS} поколениями ({num_workers} ядер)...")
    winner = p.run(pe.evaluate, NUM_GENERATIONS)

    # Сохранение результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'winner_genome.pkl'), 'wb') as f:
        pickle.dump(winner, f)

    return winner, config, stats


def visualize_predictions(net, data_holder, num_examples=5):
    plt.switch_backend('TkAgg')
    indices = np.random.choice(len(data_holder.test_images), num_examples)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        img = data_holder.test_images[idx].reshape(IMAGE_SIZE)
        true_label = data_holder.test_labels[idx]
        output = net.activate(data_holder.test_images[idx])
        predicted_label = np.argmax(output)
        confidence = output[predicted_label]

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(net, data_holder):
    plt.switch_backend('Agg')
    predictions = [np.argmax(net.activate(img)) for img in data_holder.test_images]
    cm = confusion_matrix(data_holder.test_labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def analyze_errors(net, data_holder, num_errors=5):
    predictions = [np.argmax(net.activate(img)) for img in data_holder.test_images]
    errors = np.where(np.array(predictions) != data_holder.test_labels)[0]

    print(f"Всего ошибок: {len(errors)} из {len(data_holder.test_images)}")
    print("Примеры ошибок:")

    for i in errors[:num_errors]:
        img = data_holder.test_images[i].reshape(IMAGE_SIZE)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {data_holder.test_labels[i]}, Pred: {predictions[i]}")
        plt.axis('off')
        plt.show()


class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.label = tk.Label(self.root, text="Нарисуйте цифру", font=('Arial', 14))
        self.label.pack()

        self.result_label = tk.Label(self.root, text="", font=('Arial', 16))
        self.result_label.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Распознать", command=self.recognize).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Очистить", command=self.clear).pack(side=tk.LEFT, padx=5)

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_point:
            self.canvas.create_line(self.last_point[0], self.last_point[1], x, y,
                                   width=15, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_point, (x, y)], fill=0, width=15)
        self.last_point = (x, y)

    def reset(self, event):
        self.last_point = None

    def recognize(self):
        try:
            # Подготовка изображения
            img = self.image.resize(IMAGE_SIZE)
            img_array = (np.array(img) / 255.0)
            inputs = img_array.flatten()

            # Получение предсказания
            output = self.net.activate(inputs)
            probabilities = softmax(output)  # Применяем softmax
            predicted = np.argmax(probabilities)
            confidence = probabilities[predicted]

            # Вывод результатов
            self.result_label.config(
                text=f"Цифра: {predicted} \nУверенность: {confidence:.2%}",
                fg="green" if confidence > 0.5 else "red"
            )

            # Визуализация вероятностей
            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(6, 3))
            plt.bar(range(10), probabilities)
            plt.ylim(0, 1)
            plt.title('Вероятности распознавания')
            plt.xlabel('Цифра')
            plt.ylabel('Вероятность')
            plt.xticks(range(10))

            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except Exception as e:
            self.result_label.config(text=f"Ошибка: {str(e)}", fg="red")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")
        self.last_point = None  # Сброс последней точки

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    # Проверка данных
    if len(data_holder.train_images) == 0:
        raise ValueError("Обучающие данные не загружены!")

    # обучение !!!

    # # Запуск NEAT
    # best_genome, config, stats = run_neat_mnist_parallel(CONFIG_PATH)
    #
    # # Оценка на тестовых данных
    # correct = 0
    # net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    # for i in range(len(data_holder.test_images)):
    #     outputs = net.activate(data_holder.test_images[i])
    #     if np.argmax(outputs) == data_holder.test_labels[i]:
    #         correct += 1
    #
    # accuracy = correct / len(data_holder.test_images)
    # print(f"\nТочность на тестовом наборе: {accuracy:.2%}")

    # использование!

    # Загрузка или обучение модели
    model_path = os.path.join(OUTPUT_DIR, 'winner_genome.pkl')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            best_genome = pickle.load(f)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_PATH)
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        print("\nЗагружена сохраненная модель")
    else:
        best_genome, config, stats = run_neat_mnist_parallel(CONFIG_PATH)
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # Оценка на тестовых данных
    correct = sum(np.argmax(net.activate(img)) == label
                for img, label in zip(data_holder.test_images, data_holder.test_labels))
    accuracy = correct / len(data_holder.test_images)
    print(f"\nТочность на тестовом наборе: {accuracy:.2%}")

    # Визуализация
    visualize_predictions(net, data_holder)
    plot_confusion_matrix(net, data_holder)
    analyze_errors(net, data_holder)

    # Запуск интерфейса
    plt.switch_backend('TkAgg')
    drawer = DigitDrawer(net)
    drawer.run()