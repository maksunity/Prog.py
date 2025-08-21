import neat
import numpy as np
import os
import pickle
import visualize
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image # Для изменения размера изображений MNIST

# --- Параметры ---
CONFIG_PATH = 'config-feedforward.txt'
IMAGE_SIZE = (8, 8) # Уменьшаем MNIST до 8x8 для скорости NEAT
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_GENERATIONS = 50 # MNIST обычно проще, может хватить меньше поколений
OUTPUT_DIR = 'neat_output_mnist' # Папка для результатов MNIST

# Убедитесь, что в config-feedforward.txt num_inputs = IMAGE_SIZE[0] * IMAGE_SIZE[1]

# --- 1. Загрузка и предобработка MNIST ---
def load_and_preprocess_mnist(image_size, train=True):
    print(f"Загрузка {'обучающего' if train else 'тестового'} MNIST...")
    # Используем torchvision для загрузки
    mnist_dataset = datasets.MNIST(
        root='./mnist_data', # Папка для скачивания/хранения MNIST
        train=train,
        download=True,
        transform=None # Загружаем как PIL изображения сначала
    )

    images = []
    labels = []
    target_size = image_size

    print(f"Предобработка изображений до размера {target_size}...")
    for img, label in mnist_dataset:
        # Изменение размера с использованием Pillow (PIL)
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS) # Используем качественный метод ресайза

        # Преобразование в numpy массив и нормализация [0, 1]
        img_np = np.array(img_resized) / 255.0

        # Выравнивание в 1D вектор
        flattened_img = img_np.flatten()

        images.append(flattened_img)
        labels.append(label)

    print(f"Загружено и обработано {len(images)} изображений.")
    return np.array(images), np.array(labels)

# Загрузка данных MNIST
train_images_mnist, train_labels_mnist = load_and_preprocess_mnist(IMAGE_SIZE, train=True)
test_images_mnist, test_labels_mnist = load_and_preprocess_mnist(IMAGE_SIZE, train=False)

# Проверка, что данные загружены
if train_images_mnist.size == 0 or train_labels_mnist.size == 0:
    raise ValueError("Обучающий датасет MNIST пуст или не удалось загрузить.")

# --- 2. Функция оценки приспособленности (Fitness Function) ---
# Используем ту же функцию eval_genomes, но она будет работать с данными MNIST
def eval_genomes_mnist(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        correct_predictions = 0
        total_samples = len(train_images_mnist)

        if total_samples == 0:
            genome.fitness = 0.0
            continue

        for i in range(total_samples):
            inputs = train_images_mnist[i]
            true_label = train_labels_mnist[i]
            outputs = net.activate(inputs)
            predicted_label = np.argmax(outputs)
            if predicted_label == true_label:
                correct_predictions += 1

        genome.fitness = correct_predictions / total_samples

# --- 3. Запуск NEAT (Функция run_neat похожа, но использует eval_genomes_mnist) ---
def run_neat_mnist(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Проверка num_inputs и num_outputs
    if config.genome_config.num_inputs != NUM_INPUTS:
         raise ValueError(f"Ошибка: num_inputs в конфиге ({config.genome_config.num_inputs}) "
                         f"не совпадает с размером входных данных ({NUM_INPUTS}). "
                         f"Обновите config-feedforward.txt.")
    if config.genome_config.num_outputs != 10:
         raise ValueError(f"Ошибка: num_outputs в конфиге ({config.genome_config.num_outputs}) "
                         f"должно быть 10. Обновите config-feedforward.txt.")


    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print(f"\nЗапуск NEAT для MNIST с {NUM_GENERATIONS} поколениями...")
    winner = p.run(eval_genomes_mnist, NUM_GENERATIONS)

    print('\nЛучший геном MNIST:\n{!s}'.format(winner))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    winner_path = os.path.join(OUTPUT_DIR, 'winner_genome_mnist.pkl')
    with open(winner_path, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Лучший геном MNIST сохранен в {winner_path}")

    node_names = {-i-1: str(i) for i in range(10)}
    try:
        visualize.draw_net(config, winner, True, node_names=node_names, directory=OUTPUT_DIR, filename='net_mnist')
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(OUTPUT_DIR, 'avg_fitness_mnist.svg'))
        visualize.plot_species(stats, view=False, filename=os.path.join(OUTPUT_DIR, 'speciation_mnist.svg'))
        print(f"Графики и структура сети MNIST сохранены в папку: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Ошибка визуализации MNIST (возможно, не установлен graphviz): {e}")

    return winner, config, stats

# --- Основной блок ---
if __name__ == '__main__':
    # Создаем директорию для вывода MNIST, если ее нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Проверка наличия данных MNIST перед запуском
    if train_images_mnist.size > 0:
        best_genome_mnist, neat_config_mnist, statistics_mnist = run_neat_mnist(CONFIG_PATH)

        # Оценка лучшего генома на тестовом наборе MNIST
        print("\nОценка лучшего генома MNIST на тестовом наборе:")
        if test_images_mnist.size > 0:
            winner_net_mnist = neat.nn.FeedForwardNetwork.create(best_genome_mnist, neat_config_mnist)
            correct = 0
            for i in range(len(test_images_mnist)):
                output = winner_net_mnist.activate(test_images_mnist[i])
                prediction = np.argmax(output)
                if prediction == test_labels_mnist[i]:
                    correct += 1
            accuracy = correct / len(test_images_mnist)
            print(f"Точность на тестовом наборе MNIST: {accuracy:.4f}")
        else:
            print("Тестовый набор MNIST пуст или не загружен.")
    else:
         print("Обучение MNIST не может быть запущено, так как обучающий набор данных пуст.")