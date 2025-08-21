import neat
from neat import ctrnn, nn, population, parallel, config
import numpy as np
import os
import pickle
import math
from torchvision import datasets
from PIL import Image, ImageDraw
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

# Конфигурационные параметры
CONFIG_PATH = 'config-hyperneat.txt'
IMAGE_SIZE = (16, 16)
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_OUTPUTS = 10
OUTPUT_DIR = 'hyperneat_output'
CHECKPOINT_EVERY = 5
NUM_GENERATIONS = 200
SUBSET_FRAC = 0.5



class MNISTDataHolder:
    def __init__(self):
        self.train_images, self.train_labels = self._load_data(train=True)
        self.test_images, self.test_labels = self._load_data(train=False)

    def _load_data(self, train=True):
        dataset = datasets.MNIST(
            root='./mnist_data', train=train, download=True, transform=None)
        images, labels = [], []
        for img, label in dataset:
            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_np = 1.0 - (np.array(img_resized) / 255.0)
            images.append(img_np.flatten())
            labels.append(label)
        return np.array(images), np.array(labels)


class HyperNetwork:
    def __init__(self, cppn, substrate):
        self.cppn = cppn
        self.substrate = substrate

    def activate(self, inputs):
        self.substrate.set_inputs(inputs.reshape(IMAGE_SIZE))
        return self.substrate.activate(self.cppn)


def create_substrate():
    return neat.Substrate(
        inputs=[(x, y) for x in np.linspace(-1, 1, IMAGE_SIZE[0])
                for y in np.linspace(-1, 1, IMAGE_SIZE[1])],
        outputs=[(0, 0, i) for i in range(NUM_OUTPUTS)],
        hidden=[(0.5, 0.5)]
    )


def eval_genome(genome, config):
    try:
        substrate = create_substrate()
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        hyper_net = HyperNetwork(cppn, substrate)

        indices = np.random.choice(len(data_holder.train_images),
                                   int(len(data_holder.train_images) * SUBSET_FRAC))
        correct = 0
        for i in indices:
            output = hyper_net.activate(data_holder.train_images[i])
            if np.argmax(output) == data_holder.train_labels[i]:
                correct += 1
        return correct / len(indices)
    except Exception as e:
        print(f"Ошибка оценки генома: {e}")
        return 0.0


def run_hyperneat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(CHECKPOINT_EVERY,
                                       filename_prefix=os.path.join(OUTPUT_DIR, 'hyperneat-chkpt-')))

    pe = parallel.ParallelEvaluator(max(2, os.cpu_count() // 2), eval_genome)
    winner = pop.run(pe.evaluate, NUM_GENERATIONS)

    with open(os.path.join(OUTPUT_DIR, 'hyperneat_winner.pkl'), 'wb') as f:
        pickle.dump(winner, f)

    return winner, config, stats


def plot_training_curves(log_file):
    data = np.loadtxt(log_file, delimiter=',', skiprows=1)
    gens, best, avg = data[:, 0], data[:, 1], data[:, 2]

    plt.figure(figsize=(10, 6))
    plt.plot(gens, best, label='Лучшая приспособленность')
    plt.plot(gens, avg, label='Средняя приспособленность')
    plt.title('Динамика обучения HyperNEAT')
    plt.xlabel('Поколение')
    plt.ylabel('Приспособленность')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperneat_training.png'))
    plt.close()


def plot_confusion_matrix(net, data_holder):
    preds = []
    for img in data_holder.test_images:
        output = net.activate(img)
        preds.append(np.argmax(output))

    cm = confusion_matrix(data_holder.test_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок HyperNEAT')
    plt.xlabel('Предсказанные')
    plt.ylabel('Истинные')
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperneat_confusion.png'))
    plt.close()


def plot_tsne(net, data_holder):
    outputs = []
    for img in data_holder.test_images[:1000]:
        outputs.append(net.activate(img))

    tsne = TSNE(n_components=2, random_state=42)
    features = tsne.fit_transform(outputs)

    plt.figure(figsize=(10, 8))
    plt.scatter(features[:, 0], features[:, 1], c=data_holder.test_labels[:1000],
                cmap='tab10', alpha=0.6)
    plt.colorbar()
    plt.title('t-SNE визуализация выходов HyperNEAT')
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperneat_tsne.png'))
    plt.close()


class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.threshold = 0.3
        self.root = tk.Tk()
        self.root.title("HyperNEAT Digit Recognizer")

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.label = tk.Label(self.root, text="Нарисуйте цифру", font=('Arial', 14))
        self.label.pack()

        self.result_label = tk.Label(self.root, text="", font=('Arial', 16))
        self.result_label.pack()

        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Label(frame, text='Порог').pack(side=tk.LEFT)
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

    def recognize(self):
        try:
            img = self.image.resize(IMAGE_SIZE)
            img_array = 1.0 - (np.array(img) / 255.0)
            output = self.net.activate(img_array.flatten())
            pred = np.argmax(output)
            conf = output[pred]

            color = 'green' if conf >= self.threshold else 'red'
            self.result_label.config(text=f"Цифра: {pred} Уверенность: {conf:.1%}", fg=color)

            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(6, 3))
            plt.bar(range(10), output)
            plt.title('Выходы сети')
            plt.xticks(range(10))
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except Exception as e:
            self.result_label.config(text=f"Ошибка: {str(e)}", fg='red')

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.result_label.config(text="")

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_holder = MNISTDataHolder()

    if not os.path.exists(os.path.join(OUTPUT_DIR, 'hyperneat_winner.pkl')):
        winner, config, stats = run_hyperneat(CONFIG_PATH)
    else:
        with open(os.path.join(OUTPUT_DIR, 'hyperneat_winner.pkl'), 'rb') as f:
            winner = pickle.load(f)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             CONFIG_PATH)

    substrate = create_substrate()
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    hyper_net = HyperNetwork(cppn, substrate)

    # Визуализация
    plot_training_curves(os.path.join(OUTPUT_DIR, 'hyperneat_training_log.csv'))
    plot_confusion_matrix(hyper_net, data_holder)
    plot_tsne(hyper_net, data_holder)

    # GUI
    print("\nЗапуск интерактивного интерфейса...")
    drawer = DigitDrawer(hyper_net)
    drawer.run()