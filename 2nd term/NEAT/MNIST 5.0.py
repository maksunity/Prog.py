import neat
import numpy as np
import os
import pickle
import math
import visualize  # OLD: визуализация структур сетей из NEAT
from torchvision import datasets
from PIL import Image, ImageDraw
from neat.parallel import ParallelEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import csv

warnings.filterwarnings("ignore", category=UserWarning)

# --- Параметры ---
CONFIG_PATH = 'config-feedforward.txt'
IMAGE_SIZE = (16, 16)  # размер входа
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_GENERATIONS = 75
OUTPUT_DIR = 'neat_output_mnist_parallel_5.0'
LOG_CSV = os.path.join(OUTPUT_DIR, 'training_log.csv')  # лог CSV
CHECKPOINT_EVERY = 5  # чекпоинты каждые N поколений
SUBSET_FRAC = 0.6  # константная подвыборка

class MNISTDataHolder:
    def __init__(self):
        self.train_images, self.train_labels = self._load_data(train=True)
        self.test_images, self.test_labels = self._load_data(train=False)

    def _load_data(self, train=True):
        dataset = datasets.MNIST(
            root='./mnist_data', train=train, download=True, transform=None
        )
        images, labels = [], []
        for img, label in dataset:
            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_np = np.array(img_resized) / 255.0  # нормализация
            images.append(img_np.flatten())
            labels.append(label)
        return np.array(images), np.array(labels)

# Инициализация данных

data_holder = MNISTDataHolder()

# Оценка генома по случайной подвыборке SUBSET_FRAC

def eval_single_genome_mnist(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    try:
        total = len(data_holder.train_images)
        subset_size = int(total * SUBSET_FRAC)
        indices = np.random.choice(total, subset_size, replace=False)
        correct = sum(
            np.argmax(net.activate(data_holder.train_images[i])) == data_holder.train_labels[i]
            for i in indices
        )
        return correct / subset_size
    except Exception as e:
        print(f"Ошибка оценки генома: {e}")
        return 0.0

# Функция запуска NEAT с ParallelEvaluator, чекпоинтами и логированием

def run_neat_mnist_parallel(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    if config.genome_config.num_inputs != NUM_INPUTS:
        raise ValueError(f"Несоответствие входов: {config.genome_config.num_inputs} vs {NUM_INPUTS}")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=CHECKPOINT_EVERY,
                                     filename_prefix=os.path.join(OUTPUT_DIR, 'chkpt-')))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_file = open(LOG_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['generation', 'best_fitness', 'avg_fitness'])

    num_workers = min(os.cpu_count(), 10)
    pe = ParallelEvaluator(num_workers, eval_single_genome_mnist)

    gen_counter = {'gen': 0}


    # Обёртка логирования
    def fitness_wrapper(genomes, config_inner):
        gen = gen_counter['gen']
        # Распараллеливание оценки
        result = pe.evaluate(genomes, config_inner)

        # Логируем
        fits = [g.fitness for _, g in genomes]
        csv_writer.writerow([gen, max(fits), float(np.mean(fits))])
        csv_file.flush()

        gen_counter['gen'] += 1
        return result

    winner = p.run(fitness_wrapper, NUM_GENERATIONS)
    csv_file.close()
    with open(os.path.join(OUTPUT_DIR, 'winner_genome.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    return winner, config, stats

# Постобработка: графики и анализ


def plot_training_curves():
    data = np.loadtxt(LOG_CSV, delimiter=',', skiprows=1)
    gens, best, avg = data[:,0], data[:,1], data[:,2]
    plt.figure()
    plt.plot(gens, best, label='Best')
    plt.plot(gens, avg, label='Average')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Training Curves')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()


# def post_training_analysis(net, data_holder):
#     preds = [np.argmax(net.activate(img)) for img in data_holder.test_images]
#     cm = confusion_matrix(data_holder.test_labels, preds)
#     plt.figure(figsize=(8,6))
#     sns.heatmap(cm, annot=True, fmt='d')
#     plt.title('Confusion Matrix')
#     plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
#     plt.close()
#
#     probs = np.array([net.activate(img) for img in data_holder.test_images])
#     y_true = (data_holder.test_labels == 0).astype(int)
#     y_score = probs[:, 0]
#     fpr, tpr, _ = roc_curve(y_true, y_score)
#     roc_auc = auc(fpr, tpr)
#     plt.figure()
#     plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
#     plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve for Digit 0')
#     plt.legend(); plt.savefig(os.path.join(OUTPUT_DIR, 'roc_digit0.png')); plt.close()
#
#     tsne = TSNE(n_components=2)
#     features = tsne.fit_transform(probs)
#     plt.figure(figsize=(6,6))
#     for digit in range(10):
#         idxs = data_holder.test_labels == digit
#         plt.scatter(features[idxs,0], features[idxs,1], label=str(digit), s=5)
#     plt.legend(); plt.title('t-SNE of Output Vectors')
#     plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_outputs.png')); plt.close()

def post_training_analysis(net, data_holder):
    print("\n--- Анализ после обучения на тестовых данных ---")
    try:
        # Получаем предсказания для всего тестового набора
        print("Получение предсказаний на тестовом наборе...")
        test_inputs = data_holder.test_images
        test_labels = data_holder.test_labels
        # Используем list comprehension для предсказаний (может быть быстрее для большого набора)
        preds = [np.argmax(net.activate(img)) for img in test_inputs]
        print(f"Предсказания получены для {len(preds)} тестовых образцов.")

        # 1. Расчет и вывод точности (Accuracy)
        accuracy = accuracy_score(test_labels, preds)
        print(f"Точность на тестовом наборе: {accuracy:.4f} ({int(accuracy * len(test_labels))}/{len(test_labels)})")

        # 2. Матрица ошибок (Confusion Matrix)
        print("Построение матрицы ошибок...")
        cm = confusion_matrix(test_labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # Добавлен cmap
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label') # Добавлено
        plt.xlabel('Predicted Label') # Добавлено
        plt.tight_layout() # Чтобы метки не обрезались
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        plt.close()
        print("Матрица ошибок сохранена.")

        # 3. ROC-кривая для цифры 0 (опционально, можно для любой)
        print("Построение ROC-кривой для цифры 0...")
        # Получаем сырые выходы или вероятности (если используете Softmax)
        # Здесь используем сырые выходы, предполагая, что более высокий выход = больше уверенности
        raw_outputs = np.array([net.activate(img) for img in test_inputs])
        y_true_roc = (test_labels == 0).astype(int) # 1 если цифра 0, иначе 0
        # Используем выход нейрона для цифры 0 как оценку уверенности
        y_score_roc = raw_outputs[:, 0]
        fpr, tpr, _ = roc_curve(y_true_roc, y_score_roc)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})') # Улучшен стиль
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Линия случайного угадывания
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Digit 0 (Test Set)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_digit0.png'))
        plt.close()
        print("ROC-кривая сохранена.")

        # 4. t-SNE визуализация выходов сети (требует времени)
        print("Построение t-SNE визуализации (может занять время)...")
        # Используем сырые выходы сети как признаки для t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30.0) # Добавлен random_state и perplexity
        features = tsne.fit_transform(raw_outputs)
        plt.figure(figsize=(8, 8)) # Увеличен размер
        scatter = plt.scatter(features[:, 0], features[:, 1], c=test_labels, cmap='viridis', s=10) # s=5 слишком мелко
        plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title="Digits") # Улучшена легенда
        plt.title('t-SNE Visualization of Network Outputs (Test Set)')
        plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_outputs.png'))
        plt.close()
        print("t-SNE визуализация сохранена.")

    except Exception as e:
        print(f"Ошибка во время анализа после обучения: {e}")
        import traceback
        print(traceback.format_exc())

def display_sample_predictions(net, data_holder, total_images=10, images_per_plot=5):
    """
    Отображает случайные изображения из тестового набора порциями,
    сохраняя каждую порцию (график) в отдельный файл.

    Args:
        net: Обученная нейронная сеть NEAT.
        data_holder: Объект с тестовыми данными (images, labels).
        total_images (int): Общее количество случайных изображений для показа.
        images_per_plot (int): Количество изображений на одном графике/файле.
    """
    print(f"\n--- Отображение {total_images} случайных предсказаний ({images_per_plot} на файл) ---")
    try:
        test_images = data_holder.test_images
        test_labels = data_holder.test_labels
        num_available = len(test_images)

        if num_available == 0:
            print("Тестовый набор пуст, невозможно отобразить примеры.")
            return
        if total_images <= 0 or images_per_plot <= 0:
             print("Количество изображений (total_images и images_per_plot) должно быть положительным.")
             return

        # Убедимся, что не просим больше изображений, чем есть
        actual_total_images = min(total_images, num_available)
        if actual_total_images < total_images:
             print(f"Внимание: В тестовом наборе доступно только {actual_total_images} изображений.")

        # Выбираем все необходимые случайные индексы один раз
        all_indices = np.random.choice(num_available, actual_total_images, replace=False)

        # Определяем количество графиков/файлов
        # Используем деление с округлением вверх
        num_plots = math.ceil(actual_total_images / images_per_plot)

        indices_offset = 0 # Смещение в массиве all_indices
        for plot_idx in range(num_plots):
            # Выбираем индексы для текущего графика
            start_idx = indices_offset
            # Конечный индекс не должен превышать общее количество выбранных индексов
            end_idx = min(indices_offset + images_per_plot, actual_total_images)
            current_indices = all_indices[start_idx:end_idx]
            num_in_this_plot = len(current_indices)

            if num_in_this_plot == 0: # Проверка на всякий случай
                continue

            # Создаем фигуру для текущей порции изображений
            # 1 строка, num_in_this_plot колонок
            fig, axes = plt.subplots(1, num_in_this_plot, figsize=(num_in_this_plot * 3, 4))
            # Если только один образец, axes не будет массивом, делаем его списком
            if num_in_this_plot == 1:
                 axes = [axes]

            fig.suptitle(f'Примеры предсказаний (Часть {plot_idx + 1}/{num_plots})')

            # Отображаем изображения на текущем графике
            for i, ax in enumerate(axes):
                idx = current_indices[i]
                flat_image = test_images[idx]
                true_label = test_labels[idx]

                # Преобразуем плоское изображение обратно в 2D
                image_2d = flat_image.reshape(IMAGE_SIZE)

                # Получаем предсказание сети
                output = net.activate(flat_image)
                prediction = np.argmax(output)

                # Отображаем
                ax.imshow(image_2d, cmap='gray')
                ax.set_title(f"Предск: {prediction}\nИстина: {true_label}",
                             color=("green" if prediction == true_label else "red"))
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Оставляем место для suptitle

            # Сохраняем текущий график в отдельный файл
            filename = os.path.join(OUTPUT_DIR, f'sample_predictions_{plot_idx + 1}.png')
            plt.savefig(filename)
            print(f"График сохранен в {filename}")

            # Закрываем текущую фигуру, чтобы она не отображалась через plt.show()
            # и чтобы освободить память перед созданием следующей фигуры
            plt.close(fig)

            # Сдвигаем смещение для следующей итерации
            indices_offset += num_in_this_plot

    except Exception as e:
        print(f"Ошибка при отображении примеров предсказаний: {e}")
        import traceback
        print(traceback.format_exc())

# GUI: рисование и распознавание

class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.threshold = 0.5  # порог уверенности
        self.root = tk.Tk()
        self.root.title("Digit Recognizer with Controls")

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.label = tk.Label(self.root, text="Нарисуйте цифру", font=('Arial', 14))
        self.label.pack()

        self.result_label = tk.Label(self.root, text="", font=('Arial', 16))
        self.result_label.pack()

        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Label(frame, text='Threshold').pack(side=tk.LEFT)
        self.thresh_slider = Scale(frame, from_=0, to=1, resolution=0.01,
                                   orient=HORIZONTAL, command=self.update_threshold)
        self.thresh_slider.set(self.threshold)
        self.thresh_slider.pack(side=tk.LEFT)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Распознать", command=self.recognize).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Очистить", command=self.clear).pack(side=tk.LEFT, padx=5)

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def update_threshold(self, val):
        self.threshold = float(val)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_point:
            self.canvas.create_line(self.last_point[0], self.last_point[1], x, y,
                                   width=15, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_point, (x, y)], fill=0, width=15)
        self.last_point = (x, y)

    def reset(self, event):
        self.last_point = None

    # def recognize(self):
    #     try:
    #         img = self.image.resize(IMAGE_SIZE)
    #         # img_array = np.array(img) / 255.0
    #         img_array = 1.0 - (np.array(img) / 255.0)
    #         inputs = img_array.flatten()
    #         output = self.net.activate(inputs)
    #         exp = np.exp(output - np.max(output))
    #         probs = exp / exp.sum()
    #         pred = np.argmax(probs)
    #         conf = probs[pred]
    #         color = 'green' if conf >= self.threshold else 'red'
    #         self.result_label.config(text=f"Цифра: {pred} Уверенность: {conf:.1%}", fg=color)
    #
    #         plt.switch_backend('Agg')
    #         fig = plt.figure(figsize=(5,2))
    #         plt.bar(range(10), probs)
    #         plt.ylim(0,1); plt.title('Probabilities'); plt.xticks(range(10))
    #         canvas = FigureCanvasTkAgg(fig, master=self.root)
    #         canvas.draw(); canvas.get_tk_widget().pack()
    #     except Exception as e:
    #         self.result_label.config(text=f"Ошибка: {e}", fg='red')
    def recognize(self):
        try:
            # 1. Использовать тот же метод ресайза, что и при обучении
            img = self.image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            # 2. Нормализация
            img_array = np.array(img) / 255.0

            # 3. ИНВЕРТИРОВАНИЕ! (Белый фон -> 0, Черная цифра -> 1)
            img_array_inverted = 1.0 - img_array

            # 4. Выравнивание
            inputs = img_array_inverted.flatten()

            # 5. Получение сырых выходов сети
            output = self.net.activate(inputs)

            # --- Вариант А: Использовать Softmax (как у вас было) ---
            # Стабилизация Softmax вычитанием максимума
            output_stable = output - np.max(output)
            exp_outputs = np.exp(output_stable)
            probs = exp_outputs / np.sum(exp_outputs)
            pred = np.argmax(probs)
            conf = probs[pred]
            # --- Конец Варианта А ---

            # --- Вариант Б: Использовать argmax на сырых выходах (часто для NEAT) ---
            # pred = np.argmax(output)
            # conf = output[pred] # В этом случае conf - это сырая активация, а не вероятность
            # Можно попытаться нормировать сырые выходы иначе, если нужно подобие уверенности,
            # но для простоты можно просто показать предсказание.
            # print(f"Raw outputs: {output}") # Для отладки
            # --- Конец Варианта Б ---


            # Используем предсказание и уверенность из выбранного варианта (А или Б)
            # В примере оставляем вариант А (Softmax)

            color = 'green' if conf >= self.threshold else 'red'
            self.result_label.config(text=f"Цифра: {pred} Уверенность: {conf:.1%}", fg=color)

            # Удаление старого графика перед отрисовкой нового
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Canvas) and hasattr(widget, 'figure'):
                     widget.destroy()
                elif isinstance(widget, FigureCanvasTkAgg): # Более надежный способ найти виджет графика
                     widget.get_tk_widget().destroy()


            # Отображение гистограммы вероятностей (из Варианта А)
            plt.switch_backend('Agg') # Переключаем бэкенд matplotlib перед созданием фигуры
            fig_probs = plt.figure(figsize=(5, 2)) # Используем новое имя переменной
            plt.bar(range(10), probs)
            plt.ylim(0, 1)
            plt.title('Probabilities')
            plt.xticks(range(10))
            plt.tight_layout() # Чтобы заголовок не наезжал

            # Создание нового виджета канваса для графика
            canvas_probs = FigureCanvasTkAgg(fig_probs, master=self.root) # Используем новое имя переменной
            canvas_probs.draw()
            # Упаковываем виджет графика ВНИЗУ окна
            canvas_probs.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            # Сохраняем ссылку на фигуру, чтобы ее можно было удалить (не обязательно, т.к. ищем по типу)
            # canvas_probs.figure = fig_probs # Сохраняем ссылку (опционально)

        except Exception as e:
            self.result_label.config(text=f"Ошибка: {e}", fg='red')
            import traceback
            print(f"Ошибка в recognize: {traceback.format_exc()}") # Печать полной ошибки в консоль

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.result_label.config(text="")

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, 'winner_genome.pkl')
    if os.path.exists(model_path):
        print(f"Загрузка обученного генома из {model_path}")
        with open(model_path, 'rb') as f:
            winner = pickle.load(f)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             CONFIG_PATH)
    else:
        winner, config, stats = run_neat_mnist_parallel(CONFIG_PATH)
    print("Создание сети из лучшего генома...")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    print("\nТестовая точность:")
    test_acc = sum(np.argmax(net.activate(img)) == label
                   for img, label in zip(data_holder.test_images, data_holder.test_labels)) / len(
        data_holder.test_images)
    print(f"Accuracy: {test_acc:.2%}")

    plot_training_curves()
    post_training_analysis(net, data_holder)
    display_sample_predictions(net, data_holder, total_images=100, images_per_plot=10)

    print("Запуск GUI для рисования...")
    drawer = DigitDrawer(net)
    drawer.run()
