import neat
import numpy as np
import os
import pickle
import math
from torchvision import datasets
from PIL import Image, ImageDraw
import numba
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
CONFIG_PATH = 'config-hyperneat.txt' # <--- ИЗМЕНЕНО: Путь к конфигу HyperNEAT
IMAGE_SIZE = (16, 16)  # Размер входного изображения для СУБСТРАТА
SUBSTRATE_INPUT_NODES = IMAGE_SIZE[0] * IMAGE_SIZE[1] # Количество входов субстрата
SUBSTRATE_OUTPUT_NODES = 10 # Количество выходов субстрата (10 цифр)

# Параметры CPPN (должны совпадать с config-hyperneat.txt)
CPPN_INPUTS = 5 # (x_src, y_src, x_target, y_target, bias)
CPPN_OUTPUTS = 1 # (weight)

NUM_GENERATIONS = 50 # Может потребоваться больше для HyperNEAT
OUTPUT_DIR = 'hyperneat_output_mnist' # <--- ИЗМЕНЕНО: Папка вывода
LOG_CSV = os.path.join(OUTPUT_DIR, 'training_log.csv')
CHECKPOINT_EVERY = 5
SUBSET_FRAC = 0.5

# --- Параметры для генерации субстрата ---
CPPN_WEIGHT_THRESHOLD = 0.2 # Порог для формирования связи в субстрате


# --- Класс для хранения данных MNIST (без изменений) ---
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
            img_np = np.array(img_resized) / 255.0
            images.append(img_np.flatten()) # Данные остаются плоскими для подачи в субстрат
            labels.append(label)
        return np.array(images), np.array(labels)

data_holder = MNISTDataHolder()


# --- Координаты нейронов субстрата ---
# Нормализуем координаты в диапазон [-1, 1] для стабильности CPPN
def get_substrate_coords():
    input_coords = []
    for r in range(IMAGE_SIZE[0]):
        y = (r / (IMAGE_SIZE[0] - 1)) * 2 - 1 if IMAGE_SIZE[0] > 1 else 0.0
        for c in range(IMAGE_SIZE[1]):
            x = (c / (IMAGE_SIZE[1] - 1)) * 2 - 1 if IMAGE_SIZE[1] > 1 else 0.0
            input_coords.append((x, y))

    output_coords = []
    for i in range(SUBSTRATE_OUTPUT_NODES):
        # Расположим выходные нейроны в линию
        x = (i / (SUBSTRATE_OUTPUT_NODES - 1)) * 2 - 1 if SUBSTRATE_OUTPUT_NODES > 1 else 0.0
        y = 1.0 # Немного выше входного слоя для визуального разделения (если бы мы рисовали)
        output_coords.append((x, y))
    return input_coords, output_coords

SUBSTRATE_INPUT_COORDS, SUBSTRATE_OUTPUT_COORDS = get_substrate_coords()


# --- Функции для работы с субстратом ---
def create_substrate_network(cppn, input_coords, output_coords):
    """
    Создает веса для субстрата (input -> output), используя CPPN.
    Возвращает матрицу весов.
    """
    num_inputs = len(input_coords)
    num_outputs = len(output_coords)
    weights = np.zeros((num_inputs, num_outputs))

    for i, (ix, iy) in enumerate(input_coords):
        for j, (ox, oy) in enumerate(output_coords):
            # Входы для CPPN: (x_src, y_src, x_target, y_target, bias)
            cppn_input = [ix, iy, ox, oy, 1.0] # Добавляем bias = 1.0
            weight = cppn.activate(cppn_input)[0] # CPPN выдает один вес

            if abs(weight) > CPPN_WEIGHT_THRESHOLD:
                weights[i, j] = weight
    return weights

def activate_substrate(substrate_weights, inputs_to_substrate):
    """
    Активирует субстрат. Простая матричная операция для полносвязного слоя.
    inputs_to_substrate: плоский массив пикселей изображения.
    """
    # inputs_to_substrate (1, num_substrate_inputs)
    # substrate_weights (num_substrate_inputs, num_substrate_outputs)
    # result = inputs @ weights
    # Используем np.tanh как функцию активации для выходного слоя субстрата (или другую)
    return np.tanh(np.dot(inputs_to_substrate, substrate_weights))


# --- Оценка генома CPPN ---
def eval_single_genome_hyperneat(genome, config): # <--- ИЗМЕНЕНО Имя
    # 1. Создаем CPPN из генома
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)

    # 2. Генерируем веса субстрата с помощью CPPN
    substrate_weights = create_substrate_network(cppn, SUBSTRATE_INPUT_COORDS, SUBSTRATE_OUTPUT_COORDS)

    try:
        total = len(data_holder.train_images)
        subset_size = int(total * SUBSET_FRAC)
        indices = np.random.choice(total, subset_size, replace=False)





        # correct = 0
        # for i in indices:
        #     image_pixels = data_holder.train_images[i] # Это уже плоский массив
        #     true_label = data_holder.train_labels[i]
        #
        #     # 3. Активируем субстрат
        #     substrate_output = activate_substrate(substrate_weights, image_pixels)
        #     prediction = np.argmax(substrate_output)
        #
        #     if prediction == true_label:
        #         correct += 1


        # Собираем батч изображений
        image_batch = data_holder.train_images[indices]  # (subset_size, num_substrate_inputs)
        true_labels_batch = data_holder.train_labels[indices]

        # Активируем субстрат на всем батче
        # activate_substrate должна быть готова принять батч:
        # substrate_weights: (num_substrate_inputs, num_substrate_outputs)
        # image_batch: (subset_size, num_substrate_inputs)
        # result: (subset_size, num_substrate_outputs)
        substrate_outputs_batch = activate_substrate(substrate_weights, image_batch)

        predictions_batch = np.argmax(substrate_outputs_batch, axis=1)
        correct = np.sum(predictions_batch == true_labels_batch)


        return correct / subset_size
    except Exception as e:
        print(f"Ошибка оценки генома (HyperNEAT): {e}")
        import traceback
        print(traceback.format_exc())
        return 0.0


# --- Функция запуска HyperNEAT ---
def run_hyperneat_parallel(config_file): # <--- ИЗМЕНЕНО Имя
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Проверяем, что конфигурация CPPN соответствует ожиданиям
    if config.genome_config.num_inputs != CPPN_INPUTS:
        raise ValueError(f"Несоответствие входов CPPN в конфиге: {config.genome_config.num_inputs} vs {CPPN_INPUTS}")
    if config.genome_config.num_outputs != CPPN_OUTPUTS:
        raise ValueError(f"Несоответствие выходов CPPN в конфиге: {config.genome_config.num_outputs} vs {CPPN_OUTPUTS}")

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

    num_workers = min(os.cpu_count() or 1, 6) # os.cpu_count() может вернуть None
    pe = ParallelEvaluator(num_workers, eval_single_genome_hyperneat) # <--- Используем новую функцию оценки

    gen_counter = {'gen': 0}

    # Обёртка для оценки и логирования (остается похожей)
    def fitness_wrapper(genomes_tuples, config_inner): # genomes_tuples это [(gid, genome), ...]
        gen = gen_counter['gen']

        # Передаем только геномы в pe.evaluate, если он этого ожидает,
        # или кортежи, если он сам разбирает.
        # ParallelEvaluator ожидает функцию, принимающую (genome, config)
        # Он сам передаст кортежи (genome_id, genome) в eval_single_genome_hyperneat
        # и тот должен принять genome, config.
        # Однако, pe.evaluate(genomes_tuples, config_inner) ожидает список кортежей
        # и функцию, которая будет вызвана для каждого genome из кортежа.
        # Здесь нужно быть внимательным с тем, как ParallelEvaluator вызывает eval_single_genome_hyperneat
        # eval_single_genome_hyperneat должна вызываться с (genome_object, config_object)
        # А fitness_wrapper получает genomes_tuples = [(genome_id, genome_object), ...]

        # Для ParallelEvaluator, которому передается eval_single_genome_hyperneat(genome, config),
        # нужно, чтобы genomes_tuples были переданы в pe.evaluate.
        # Он сам организует вызов eval_single_genome_hyperneat(genome, config) для каждого.
        # После этого мы соберем фитнесы из обновленных геномов.
        pe.evaluate(genomes_tuples, config_inner) # Эта строка обновляет genome.fitness для каждого генома

        fits = [g.fitness for _, g in genomes_tuples] # Собираем фитнесы
        if not fits: # Если список пуст (маловероятно, но для безопасности)
            best_fit, avg_fit = 0.0, 0.0
        else:
            best_fit = max(fits) if fits else 0.0
            avg_fit = float(np.mean(fits)) if fits else 0.0

        csv_writer.writerow([gen, best_fit, avg_fit])
        csv_file.flush()
        gen_counter['gen'] += 1
        # p.run ожидает, что fitness_wrapper не будет ничего возвращать, т.к. фитнес присваивается геномам внутри pe.evaluate

    winner_cppn_genome = p.run(fitness_wrapper, NUM_GENERATIONS) # <--- winner будет CPPN геномом
    csv_file.close()
    with open(os.path.join(OUTPUT_DIR, 'winner_cppn_genome.pkl'), 'wb') as f: # <--- Сохраняем CPPN
        pickle.dump(winner_cppn_genome, f)
    print(f"Лучший CPPN геном сохранен в {os.path.join(OUTPUT_DIR, 'winner_cppn_genome.pkl')}")
    return winner_cppn_genome, config, stats


# --- Постобработка (графики и анализ) ---
# plot_training_curves остается без изменений

def post_training_analysis(cppn_genome, config, data_holder): # <--- Принимает CPPN геном
    print("\n--- Анализ после обучения на тестовых данных (HyperNEAT) ---")
    try:
        # 1. Создаем CPPN и субстрат из лучшего CPPN генома
        cppn = neat.nn.FeedForwardNetwork.create(cppn_genome, config)
        substrate_weights = create_substrate_network(cppn, SUBSTRATE_INPUT_COORDS, SUBSTRATE_OUTPUT_COORDS)

        print("Получение предсказаний на тестовом наборе...")
        test_inputs = data_holder.test_images
        test_labels = data_holder.test_labels
        preds = []
        raw_outputs_list = [] # Для ROC и t-SNE

        for img_pixels in test_inputs:
            substrate_output = activate_substrate(substrate_weights, img_pixels)
            preds.append(np.argmax(substrate_output))
            raw_outputs_list.append(substrate_output)
        print(f"Предсказания получены для {len(preds)} тестовых образцов.")

        raw_outputs_np = np.array(raw_outputs_list)

        accuracy = accuracy_score(test_labels, preds)
        print(f"Точность на тестовом наборе: {accuracy:.4f} ({int(accuracy * len(test_labels))}/{len(test_labels)})")

        # Остальная часть post_training_analysis (матрица ошибок, ROC, t-SNE) остается
        # такой же, только использует preds и raw_outputs_np, полученные от субстрата.
        # ... (скопируйте сюда код для Confusion Matrix, ROC, t-SNE из вашего предыдущего кода) ...
        # ... Убедитесь, что используете raw_outputs_np для ROC и t-SNE ...
        print("Построение матрицы ошибок...")
        cm = confusion_matrix(test_labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test Set - HyperNEAT)')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_hyperneat.png'))
        plt.close(); print("Матрица ошибок сохранена.")

        print("Построение ROC-кривой для цифры 0...")
        y_true_roc = (test_labels == 0).astype(int)
        y_score_roc = raw_outputs_np[:, 0] # Используем выход для цифры 0
        fpr, tpr, _ = roc_curve(y_true_roc, y_score_roc)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Digit 0 (Test Set - HyperNEAT)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_digit0_hyperneat.png'))
        plt.close(); print("ROC-кривая сохранена.")

        print("Построение t-SNE визуализации...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30.0, len(raw_outputs_np)-1) ) # perplexity < n_samples
        features = tsne.fit_transform(raw_outputs_np)
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(features[:, 0], features[:, 1], c=test_labels, cmap='viridis', s=10)
        plt.legend(handles=scatter.legend_elements(num=10)[0] if len(np.unique(test_labels)) >=10 else scatter.legend_elements()[0],
                   labels=[str(i) for i in np.unique(test_labels)], title="Digits")
        plt.title('t-SNE Visualization of Substrate Outputs (Test Set - HyperNEAT)')
        plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_outputs_hyperneat.png'))
        plt.close(); print("t-SNE визуализация сохранена.")

    except Exception as e:
        print(f"Ошибка во время анализа после обучения (HyperNEAT): {e}")
        import traceback
        print(traceback.format_exc())


def display_sample_predictions(cppn_genome, config, data_holder, total_images=10, images_per_plot=5): # <--- Принимает CPPN
    print(f"\n--- Отображение {total_images} случайных предсказаний (HyperNEAT) ---")
    try:
        cppn = neat.nn.FeedForwardNetwork.create(cppn_genome, config)
        substrate_weights = create_substrate_network(cppn, SUBSTRATE_INPUT_COORDS, SUBSTRATE_OUTPUT_COORDS)

        # ... (остальной код display_sample_predictions почти без изменений,
        # только вместо net.activate(flat_image) будет activate_substrate(substrate_weights, flat_image))
        test_images = data_holder.test_images
        test_labels = data_holder.test_labels
        num_available = len(test_images)

        if num_available == 0: print("Тестовый набор пуст."); return
        actual_total_images = min(total_images, num_available)
        all_indices = np.random.choice(num_available, actual_total_images, replace=False)
        num_plots = math.ceil(actual_total_images / images_per_plot)
        indices_offset = 0

        for plot_idx in range(num_plots):
            start_idx = indices_offset
            end_idx = min(indices_offset + images_per_plot, actual_total_images)
            current_indices = all_indices[start_idx:end_idx]
            num_in_this_plot = len(current_indices)
            if num_in_this_plot == 0: continue

            fig, axes = plt.subplots(1, num_in_this_plot, figsize=(num_in_this_plot * 3, 4))
            if num_in_this_plot == 1: axes = [axes]
            fig.suptitle(f'Примеры предсказаний HyperNEAT (Часть {plot_idx + 1})')

            for i, ax in enumerate(axes):
                idx = current_indices[i]
                flat_image = test_images[idx] # Уже плоский
                true_label = test_labels[idx]
                image_2d = flat_image.reshape(IMAGE_SIZE)

                substrate_output = activate_substrate(substrate_weights, flat_image) # <--- Используем субстрат
                prediction = np.argmax(substrate_output)

                ax.imshow(image_2d, cmap='gray')
                ax.set_title(f"Предск: {prediction}\nИстина: {true_label}",
                             color=("green" if prediction == true_label else "red"))
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = os.path.join(OUTPUT_DIR, f'sample_predictions_hyperneat_{plot_idx + 1}.png')
            plt.savefig(filename); print(f"График сохранен в {filename}")
            plt.close(fig)
            indices_offset += num_in_this_plot
    except Exception as e:
        print(f"Ошибка отображения (HyperNEAT): {e}"); import traceback; print(traceback.format_exc())


# --- GUI: рисование и распознавание ---
class DigitDrawerHyperNEAT: # <--- Новое имя класса
    def __init__(self, cppn_genome, config): # <--- Принимает CPPN геном
        self.cppn_genome = cppn_genome
        self.config = config
        # Создаем CPPN и субстрат один раз при инициализации GUI
        self.cppn = neat.nn.FeedForwardNetwork.create(self.cppn_genome, self.config)
        self.substrate_weights = create_substrate_network(self.cppn, SUBSTRATE_INPUT_COORDS, SUBSTRATE_OUTPUT_COORDS)

        self.threshold = 0.3
        self.root = tk.Tk()
        self.root.title("Digit Recognizer (HyperNEAT)")
        # ... (остальная часть __init__ без изменений: canvas, label, result_label, etc.)
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()
        self.label = tk.Label(self.root, text="Нарисуйте цифру (HyperNEAT)", font=('Arial', 14))
        self.label.pack()
        self.result_label = tk.Label(self.root, text="", font=('Arial', 16))
        self.result_label.pack()
        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Label(frame, text='Threshold').pack(side=tk.LEFT)
        self.thresh_slider = Scale(frame, from_=0, to=1, resolution=0.01,
                                   orient=HORIZONTAL, command=self.update_threshold)
        self.thresh_slider.set(self.threshold); self.thresh_slider.pack(side=tk.LEFT)
        btn_frame = tk.Frame(self.root); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Распознать", command=self.recognize).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Очистить", command=self.clear).pack(side=tk.LEFT, padx=5)
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.canvas.bind("<B1-Motion>", self.paint); self.canvas.bind("<ButtonRelease-1>", self.reset)


    def update_threshold(self, val): self.threshold = float(val)
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_point:
            self.canvas.create_line(self.last_point[0], self.last_point[1], x, y,
                                   width=15, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_point, (x, y)], fill=0, width=15)
        self.last_point = (x, y)
    def reset(self, event): self.last_point = None
    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.result_label.config(text="")

    def recognize(self):
        try:
            img = self.image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0
            img_array_inverted = 1.0 - img_array # Инверсия
            inputs_to_substrate = img_array_inverted.flatten()

            # Используем уже созданный субстрат
            substrate_output = activate_substrate(self.substrate_weights, inputs_to_substrate)

            # Softmax для получения вероятностей
            output_stable = substrate_output - np.max(substrate_output)
            exp_outputs = np.exp(output_stable)
            probs = exp_outputs / np.sum(exp_outputs)
            pred = np.argmax(probs)
            conf = probs[pred]

            color = 'green' if conf >= self.threshold else 'red'
            self.result_label.config(text=f"Цифра: {pred} Уверенность: {conf:.1%}", fg=color)

            # ... (код для обновления графика вероятностей в GUI без изменений) ...
            for widget in self.root.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg): widget.get_tk_widget().destroy()
            plt.switch_backend('Agg')
            fig_probs = plt.figure(figsize=(5, 2))
            plt.bar(range(10), probs); plt.ylim(0, 1); plt.title('Probabilities')
            plt.xticks(range(10)); plt.tight_layout()
            canvas_probs = FigureCanvasTkAgg(fig_probs, master=self.root)
            canvas_probs.draw(); canvas_probs.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        except Exception as e:
            self.result_label.config(text=f"Ошибка: {e}", fg='red')
            import traceback
            print(f"Ошибка в recognize (HyperNEAT): {traceback.format_exc()}")

    def run(self):
        self.root.mainloop()


# --- Основной блок ---
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # Загрузка данных MNIST один раз
    print("Инициализация данных MNIST...")
    data_holder = MNISTDataHolder() # Уже было глобально, но для ясности
    print(f"Загружено {len(data_holder.train_images)} обучающих и {len(data_holder.test_images)} тестовых изображений.")


    model_path = os.path.join(OUTPUT_DIR, 'winner_cppn_genome.pkl') # <--- ИЗМЕНЕНО
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH) # Загружаем конфиг HyperNEAT

    if os.path.exists(model_path):
        print(f"Загрузка обученного CPPN генома из {model_path}")
        with open(model_path, 'rb') as f:
            winner_cppn_genome = pickle.load(f)
    else:
        print("Обученный CPPN геном не найден, запуск обучения HyperNEAT...")
        # Передаем data_holder (или делаем его глобальным для eval_single_genome_hyperneat)
        # В текущей реализации eval_single_genome_hyperneat использует глобальный data_holder
        winner_cppn_genome, _, stats = run_hyperneat_parallel(CONFIG_PATH) # config уже загружен
        # plot_training_curves() # Можно вызвать здесь, если stats возвращается и используется

    print("Анализ лучшего CPPN генома...")
    # plot_training_curves() # Вызываем здесь, если лог файл уже полностью записан
    post_training_analysis(winner_cppn_genome, config, data_holder)
    display_sample_predictions(winner_cppn_genome, config, data_holder, total_images=10, images_per_plot=5)

    print("Запуск GUI для рисования (HyperNEAT)...")
    drawer = DigitDrawerHyperNEAT(winner_cppn_genome, config) # <--- Используем новый класс GUI
    drawer.run()