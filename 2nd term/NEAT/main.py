import neat
import numpy as np
import pandas as pd
import cv2  # OpenCV для обработки изображений
import os
import pickle  # Для сохранения лучшего генома
import visualize  # Модуль визуализации из neat-python
import matplotlib.pyplot as plt

# --- Параметры ---
DATASET_BASE_PATH = r'C:\Users\maksunity\PycharmProjects\Prog_Den\Prog.py\2nd term\NEAT\dataset'  # Укажите путь к вашему датасету
CONFIG_PATH = 'config-feedforward.txt'
IMAGE_SIZE = (8, 8)  # Размер, до которого будут сжиматься вырезанные цифры (меньше = быстрее NEAT)
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1] # Убедитесь, что в config-feedforward.txt num_inputs = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_GENERATIONS = 50  # Количество поколений для обучения
OUTPUT_DIR = 'neat_output_custom'  # Папка для сохранения результатов


# --- 1. Загрузка и предобработка вашего датасета ---
def load_and_preprocess_custom_data(dataset_path, image_size):
    images = []
    labels = []
    annotation_path = os.path.join(dataset_path, '_annotations.csv')
    if not os.path.exists(annotation_path):
        print(f"Файл аннотаций не найден: {annotation_path}")
        return np.array([]), np.array([])

    try:
        df = pd.read_csv(annotation_path)
    except Exception as e:
        print(f"Ошибка чтения CSV {annotation_path}: {e}")
        return np.array([]), np.array([])

    print(f"Загрузка данных из: {dataset_path}")
    processed_count = 0
    for index, row in df.iterrows():
        img_path = os.path.join(dataset_path, row['filename'])
        if not os.path.exists(img_path):
            print(f"Предупреждение: Файл изображения не найден: {img_path}")
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Предупреждение: Не удалось прочитать изображение: {img_path}")
                continue

            # Обрезка изображения по координатам
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # Проверка корректности координат
            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
                print(f"Предупреждение: Некорректные координаты для {row['filename']}: ({xmin},{ymin})-({xmax},{ymax})")
                continue

            digit_img = img[ymin:ymax, xmin:xmax]

            if digit_img.size == 0:
                print(f"Предупреждение: Получено пустое изображение после обрезки для {row['filename']}")
                continue

            # Преобразование в градации серого
            gray_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

            # Изменение размера
            resized_img = cv2.resize(gray_img, image_size, interpolation=cv2.INTER_AREA)

            # Нормализация [0, 1] и инвертирование (если фон темный, цифры светлые - как в MNIST)
            # Если у вас наоборот (фон светлый), закомментируйте инвертирование
            normalized_img = resized_img / 255.0
            # normalized_img = 1.0 - normalized_img # Раскомментируйте, если цифры темные на светлом фоне

            # Выравнивание в 1D вектор
            flattened_img = normalized_img.flatten()

            images.append(flattened_img)
            labels.append(int(row['class']))
            processed_count += 1

        except Exception as e:
            print(f"Ошибка обработки строки {index} ({row['filename']}): {e}")
            continue  # Пропускаем эту строку при ошибке

    print(f"Обработано {processed_count} изображений.")
    return np.array(images), np.array(labels)


# Загрузка данных
train_images, train_labels = load_and_preprocess_custom_data(os.path.join(DATASET_BASE_PATH, 'train'), IMAGE_SIZE)
# test_images, test_labels = load_and_preprocess_custom_data(os.path.join(DATASET_BASE_PATH, 'test'), IMAGE_SIZE)
# valid_images, valid_labels = load_and_preprocess_custom_data(os.path.join(DATASET_BASE_PATH, 'valid'), IMAGE_SIZE)

# Проверка, что данные загружены
if train_images.size == 0 or train_labels.size == 0:
    raise ValueError("Обучающий датасет пуст или не удалось загрузить. Проверьте пути и формат данных.")


# --- 2. Функция оценки приспособленности (Fitness Function) ---
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0  # Начинаем с нулевой приспособленности
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        correct_predictions = 0
        total_samples = len(train_images)

        if total_samples == 0:
            genome.fitness = 0.0  # Нечего оценивать
            continue

        for i in range(total_samples):
            inputs = train_images[i]
            true_label = train_labels[i]

            # Получаем выход сети (10 значений)
            outputs = net.activate(inputs)

            # Предсказанный класс - индекс нейрона с максимальной активацией
            predicted_label = np.argmax(outputs)

            if predicted_label == true_label:
                correct_predictions += 1

        # Приспособленность - это точность на обучающем наборе
        genome.fitness = correct_predictions / total_samples


# --- 3. Запуск NEAT ---
def run_neat(config_file):
    # Загрузка конфигурации NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Проверка соответствия num_inputs в конфиге
    if config.genome_config.num_inputs != NUM_INPUTS:
        raise ValueError(f"Ошибка: num_inputs в конфиге ({config.genome_config.num_inputs}) "
                         f"не совпадает с размером входных данных ({NUM_INPUTS}). "
                         f"Обновите config-feedforward.txt.")
    if config.genome_config.num_outputs != 10:
        raise ValueError(f"Ошибка: num_outputs в конфиге ({config.genome_config.num_outputs}) "
                         f"должно быть 10. Обновите config-feedforward.txt.")

    # Создание популяции
    p = neat.Population(config)

    # Добавление репортеров для вывода статистики и визуализации
    p.add_reporter(neat.StdOutReporter(True))  # Вывод в консоль
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5)) # Сохранение чекпоинтов каждые 5 поколений (опционально)

    # Запуск процесса эволюции
    winner = p.run(eval_genomes, NUM_GENERATIONS)

    # Показать статистику по лучшему геному
    print('\nЛучший геном:\n{!s}'.format(winner))

    # Сохранение лучшего генома
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    winner_path = os.path.join(OUTPUT_DIR, 'winner_genome_custom.pkl')
    with open(winner_path, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Лучший геном сохранен в {winner_path}")

    # Визуализация
    node_names = {-i - 1: str(i) for i in range(10)}  # Имена выходных узлов (0-9)
    # Остальные узлы будут иметь стандартные имена (входные - отриц., скрытые - полож.)

    try:
        visualize.draw_net(config, winner, True, node_names=node_names, directory=OUTPUT_DIR, filename='net_custom')
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(OUTPUT_DIR, 'avg_fitness_custom.svg'))
        visualize.plot_species(stats, view=False, filename=os.path.join(OUTPUT_DIR, 'speciation_custom.svg'))
        print(f"Графики и структура сети сохранены в папку: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Ошибка визуализации (возможно, не установлен graphviz): {e}")

    return winner, config, stats


# --- Основной блок ---
if __name__ == '__main__':
    # Убедимся, что директория для вывода существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Проверка наличия обучающих данных перед запуском
    if train_images.size > 0:
        print(f"\nЗапуск NEAT с {NUM_GENERATIONS} поколениями...")
        best_genome, neat_config, statistics = run_neat(CONFIG_PATH)

        # Опционально: Оценка лучшего генома на тестовом или валидационном наборе
        # Если вы загрузили test_images и test_labels:
        # print("\nОценка лучшего генома на тестовом наборе:")
        # test_images, test_labels = load_and_preprocess_custom_data(os.path.join(DATASET_BASE_PATH, 'test'), IMAGE_SIZE)
        # if test_images.size > 0:
        #     winner_net = neat.nn.FeedForwardNetwork.create(best_genome, neat_config)
        #     correct = 0
        #     for i in range(len(test_images)):
        #         output = winner_net.activate(test_images[i])
        #         prediction = np.argmax(output)
        #         if prediction == test_labels[i]:
        #             correct += 1
        #     accuracy = correct / len(test_images)
        #     print(f"Точность на тестовом наборе: {accuracy:.4f}")
        # else:
        #     print("Тестовый набор пуст или не загружен.")

    else:
        print("Обучение не может быть запущено, так как обучающий набор данных пуст.")