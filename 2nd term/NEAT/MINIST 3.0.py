import neat
import numpy as np
import os
import pickle
import visualize  # OLD: визуализация топ-сетей из neat
from torchvision import datasets
from PIL import Image, ImageDraw
from neat.parallel import ParallelEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import seaborn as sns
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import csv

warnings.filterwarnings("ignore", category=UserWarning)

# --- Параметры ---
CONFIG_PATH = 'config-feedforward-extended.txt'
IMAGE_SIZE = (16, 16)  # OLD: ранее 16x16
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_GENERATIONS = 60
OUTPUT_DIR = 'neat_output_mnist_parallel'
LOG_CSV = os.path.join(OUTPUT_DIR, 'training_log.csv')  # NEW FEATURE: логирование в CSV
CHECKPOINT_EVERY = 10  # NEW FEATURE: сохранять популяцию каждые 10 поколений

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
            img_np = np.array(img_resized) / 255.0  # OLD: нормализация, без инверсии
            images.append(img_np.flatten())
            labels.append(label)
        return np.array(images), np.array(labels)

# Инициализация данных
data_holder = MNISTDataHolder()

# NEW FEATURE: адаптивная доля обучающей выборки
def get_subset_size(gen, total):
    # начиная с 30% до 100% за первые 50% поколений
    min_frac, max_frac = 0.3, 1.0
    frac = min_frac + (max_frac - min_frac) * min(gen / (NUM_GENERATIONS * 0.5), 1.0)
    return int(total * frac)

# OLD: стохастическая оценка 50%
def eval_single_genome_mnist(genome, config, gen=0):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    try:
        subset_size = get_subset_size(gen, len(data_holder.train_images))  # NEW
        indices = np.random.choice(len(data_holder.train_images), subset_size)
        correct = sum(
            np.argmax(net.activate(data_holder.train_images[i])) == data_holder.train_labels[i]
            for i in indices
        )
        return correct / subset_size
    except Exception as e:
        print(f"Ошибка оценки генома: {e}")
        return 0.0

# NEW FEATURE: коллективный проверщик с учётом номера поколения
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
    # NEW: CSV логгер
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_file = open(LOG_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['generation', 'best_fitness', 'avg_fitness'])

    # параллельная оценка
    num_workers = min(os.cpu_count(), 8)
    pe = ParallelEvaluator(num_workers,
                            lambda genome, config: eval_single_genome_mnist(genome, config, stats.generation))

    print(f"Запуск NEAT ({NUM_GENERATIONS} ген.)... cores={num_workers}")
    # Run with callback для логирования
    def fitness_wrapper(genomes, config):
        # запускает обычный evaluate, а потом логирует
        result = pe.evaluate(genomes, config)
        # после оценки поколение увеличилось в stats
        gen = stats.generation
        best = max(g.fitness for _, g in genomes)
        avg = np.mean([g.fitness for _, g in genomes])
        csv_writer.writerow([gen, best, avg])
        csv_file.flush()
        return result

    winner = p.run(fitness_wrapper, NUM_GENERATIONS)
    csv_file.close()
    # сохранение победителя
    with open(os.path.join(OUTPUT_DIR, 'winner_genome.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    return winner, config, stats

# Визуализация обучения
# NEW FEATURE: после тренировки строим графики фитнеса из CSV

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

# OLD: визуализация предсказаний, матрицы ошибок
# + NEW: ROC-AUC и t-SNE анализ после тренировки

def post_training_analysis(net, data_holder):
    # Confusion matrix
    preds = [np.argmax(net.activate(img)) for img in data_holder.test_images]
    cm = confusion_matrix(data_holder.test_labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Conf Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    # ROC-AUC
    probs = np.array([net.activate(img) for img in data_holder.test_images])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(10):
        y_true = (data_holder.test_labels == i).astype(int)
        y_score = probs[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot ROC for digit '0' as example
    plt.figure()
    plt.plot(fpr[0], tpr[0], label=f'Digit 0 (AUC={roc_auc[0]:.2f})')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend(); plt.savefig(os.path.join(OUTPUT_DIR, 'roc_digit0.png')); plt.close()
    # t-SNE
    tsne = TSNE(n_components=2)
    features = tsne.fit_transform(probs)
    plt.figure(figsize=(6,6))
    for digit in range(10):
        idxs = data_holder.test_labels == digit
        plt.scatter(features[idxs,0], features[idxs,1], label=str(digit), s=5)
    plt.legend(); plt.title('t-SNE of output vectors')
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_outputs.png')); plt.close()

# GUI с дополнительными контролами
class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.threshold = 0.5  # NEW: порог уверенности из слайда
        self.subset_frac = 0.5
        self.root = tk.Tk()
        self.root.title("Digit Recognizer with Controls")
        # NEW FEATURE: слайдер для порога уверенности
        tk.Label(self.root, text='Confidence Threshold').pack()
        self.thresh_slider = Scale(self.root, from_=0, to=1, resolution=0.01,
                                   orient=HORIZONTAL, command=self.update_threshold)
        self.thresh_slider.set(self.threshold)
        self.thresh_slider.pack()
        # Canvas и остальной код, как раньше...
        # ...
    def update_threshold(self, val):
        self.threshold = float(val)
    # OLD: paint, reset, recognize, clear, run методы

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    # Загрузка или обучение
    model_path = os.path.join(OUTPUT_DIR, 'winner_genome.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            winner = pickle.load(f)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             CONFIG_PATH)
    else:
        winner, config, stats = run_neat_mnist_parallel(CONFIG_PATH)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Анализ
    plot_training_curves()  # NEW
    post_training_analysis(net, data_holder)  # NEW

    # Запуск GUI
    drawer = DigitDrawer(net)
    drawer.run()