import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import neat
import matplotlib.pyplot as plt


def load_and_preprocess_data(base_path):
    """
    Загружает данные из CSV и изображения, предобрабатывает их
    Возвращает X (признаки) и y (метки)
    """
    X = []
    y = []

    for dataset_type in ['train', 'test', 'valid']:
        csv_path = os.path.join(base_path, dataset_type, '_annotations.csv')
        img_dir = os.path.join(base_path, dataset_type)

        # Читаем CSV с аннотациями
        annotations = pd.read_csv(csv_path)

        # Группируем по имени файла для эффективной обработки
        grouped = annotations.groupby('filename')

        for filename, group in grouped:
            # Загружаем изображение
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Не удалось загрузить изображение {img_path}")
                continue

            # Обрабатываем каждый bounding box
            for _, row in group.iterrows():
                # Извлекаем bounding box
                x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                digit_class = row['class']

                # Обрезаем цифру
                digit_img = img[y1:y2, x1:x2]

                # Пропускаем слишком маленькие изображения
                if digit_img.size < 100:
                    continue

                # Ресайз и нормализация
                digit_img = cv2.resize(digit_img, (28, 28))
                digit_img = digit_img.flatten() / 255.0

                X.append(digit_img)
                y.append(digit_class)

    return np.array(X), np.array(y)


def create_config(config_path):
    """Создает конфигурационный файл NEAT"""
    config = """
    [NEAT]
    fitness_criterion = max
    fitness_threshold = 3.9
    pop_size = 100
    reset_on_extinction = False

    [DefaultGenome]
    num_inputs = 784
    num_outputs = 10
    num_hidden = 0
    initial_connection = partial_direct 0.5
    feed_forward = True
    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient = 0.5

    [DefaultSpeciesSet]
    compatibility_threshold = 3.0

    [DefaultStagnation]
    species_fitness_func = max
    max_stagnation = 20
    """
    with open(config_path, 'w') as f:
        f.write(config)


def eval_genomes(genomes, config):
    """Функция оценки для NEAT"""
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0
        for xi, yo in zip(X_train, y_train):
            output = net.activate(xi)
            if np.argmax(output) == yo:
                genome.fitness += 1.0
        genome.fitness /= len(X_train)


def visualize_training(stats):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(stats.generation_values, stats.get_fitness_mean(), label='Средняя')
    plt.plot(stats.generation_values, stats.get_fitness_max(), label='Максимальная')
    plt.xlabel('Поколение')
    plt.ylabel('Приспособленность')
    plt.legend()
    plt.title('Прогресс обучения')

    plt.subplot(1, 2, 2)
    plt.plot(stats.generation_values, stats.get_fitness_min(), label='Минимальная')
    plt.xlabel('Поколение')
    plt.ylabel('Приспособленность')
    plt.legend()
    plt.title('Минимальная приспособленность')

    plt.tight_layout()
    plt.show()


def main():
    global X_train, X_test, y_train, y_test

    # Путь к данным
    base_path = r"C:\Users\maksunity\PycharmProjects\Prog_Den\Prog.py\2nd term\NEAT\dataset"
    # base_path = r'2nd term\NEAT\dataset'
    # Загрузка и предобработка данных
    X, y = load_and_preprocess_data(base_path)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание конфига NEAT
    config_path = 'config-neat-digits.txt'
    if not os.path.exists(config_path):
        create_config(config_path)

    # Настройка NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Создание популяции
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Запуск обучения
    winner = population.run(eval_genomes, 50)

    # Визуализация
    visualize_training(stats)

    # Оценка лучшей сети
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Тестирование
    test_predictions = [np.argmax(winner_net.activate(x)) for x in X_test]
    accuracy = np.mean(test_predictions == y_test)
    print(f"\nТочность на тестовых данных: {accuracy:.2%}")


if __name__ == "__main__":
    main()