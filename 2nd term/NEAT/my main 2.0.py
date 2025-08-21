import neat
import numpy as np
import pandas as pd
import cv2
import os
import pickle
import visualize
import matplotlib.pyplot as plt
from neat.parallel import ParallelEvaluator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Параметры ---
DATASET_BASE_PATH = r'C:\Users\maksunity\PycharmProjects\Prog_Den\Prog.py\2nd term\NEAT\dataset'  # Укажите путь к вашему датасету
CONFIG_PATH = 'config-feedforward.txt'
IMAGE_SIZE = (16, 16)  # Размер, до которого будут сжиматься вырезанные цифры (меньше = быстрее NEAT)
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1] # Убедитесь, что в config-feedforward.txt num_inputs = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_GENERATIONS = 100  # Количество поколений для обучения
OUTPUT_DIR = 'neat_output_custom'  # Папка для сохранения результатов

# --- Глобальные переменные для данных (для доступа из дочерних процессов) ---
# ВАЖНО: Простой подход. Для очень больших данных могут потребоваться
# более эффективные методы разделения памяти (например, multiprocessing.Array)
train_images_global = None
train_labels_global = None

# --- 1. Загрузка и предобработка вашего датасета ---
# (Функция load_and_preprocess_custom_data остается без изменений)
def load_and_preprocess_custom_data(dataset_path, image_size):
    images = []
    labels = []
    # --- ИСПРАВЛЕНО: Имя файла аннотаций ---
    # Убедитесь, что имя файла верное (_annotation.csv или _annotations.csv)
    # В вашем исходном коде было '_annotations.csv', в первом примере '_annotation.csv'
    # Используем '_annotation.csv' как в первом примере, измените если у вас другое имя
    annotation_path = os.path.join(dataset_path, '_annotations.csv')
    if not os.path.exists(annotation_path):
        # Попробуем альтернативное имя на всякий случай
        annotation_path_alt = os.path.join(dataset_path, '_annotation.csv')
        if os.path.exists(annotation_path_alt):
            annotation_path = annotation_path_alt
        else:
             print(f"Файл аннотаций не найден ни как _annotation.csv, ни как _annotations.csv в: {dataset_path}")
             return np.array([]), np.array([])

    try:
        df = pd.read_csv(annotation_path)
    except Exception as e:
        print(f"Ошибка чтения CSV {annotation_path}: {e}")
        return np.array([]), np.array([])

    print(f"Загрузка данных из: {dataset_path}")
    processed_count = 0
    skipped_no_file = 0
    skipped_read_error = 0
    skipped_coords = 0
    skipped_crop = 0

    for index, row in df.iterrows():
        img_path = os.path.join(dataset_path, row['filename'])
        if not os.path.exists(img_path):
            # print(f"Предупреждение: Файл изображения не найден: {img_path}")
            skipped_no_file += 1
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                # print(f"Предупреждение: Не удалось прочитать изображение: {img_path}")
                skipped_read_error +=1
                continue

            # Обрезка изображения по координатам
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # Проверка корректности координат
            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
                 # print(f"Предупреждение: Некорректные координаты для {row['filename']}: ({xmin},{ymin})-({xmax},{ymax}), shape={img.shape}")
                 skipped_coords += 1
                 continue

            digit_img = img[ymin:ymax, xmin:xmax]

            if digit_img.size == 0:
                # print(f"Предупреждение: Получено пустое изображение после обрезки для {row['filename']}")
                skipped_crop += 1
                continue

            # Преобразование в градации серого
            gray_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

            # Изменение размера
            resized_img = cv2.resize(gray_img, image_size, interpolation=cv2.INTER_AREA)

            # Нормализация [0, 1]
            normalized_img = resized_img / 255.0
            # Инвертирование, если нужно (если цифры темные на светлом фоне)
            # normalized_img = 1.0 - normalized_img

            # Выравнивание в 1D вектор
            flattened_img = normalized_img.flatten()

            images.append(flattened_img)
            # Преобразуем метку в int
            labels.append(int(row['class']))
            processed_count += 1

        except Exception as e:
            print(f"Ошибка обработки строки {index} ({row.get('filename', 'N/A')}): {e}")
            continue  # Пропускаем эту строку при ошибке

    print(f"Обработано {processed_count} изображений.")
    if skipped_no_file > 0: print(f"Пропущено из-за отсутствия файла: {skipped_no_file}")
    if skipped_read_error > 0: print(f"Пропущено из-за ошибки чтения: {skipped_read_error}")
    if skipped_coords > 0: print(f"Пропущено из-за некорректных координат: {skipped_coords}")
    if skipped_crop > 0: print(f"Пропущено из-за пустого кропа: {skipped_crop}")

    # Убедимся, что возвращаем numpy массивы
    return np.array(images), np.array(labels)


# --- 2. Функция оценки ОДНОГО генома (для ParallelEvaluator) ---
def eval_single_genome(genome, config):
    """
    Оценивает приспособленность одного генома.
    Использует глобальные переменные train_images_global и train_labels_global.
    """
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    except Exception as e:
        print(f"Ошибка при создании сети из генома {genome.key}: {e}")
        return 0.0 # Возвращаем 0 фитнес при ошибке создания сети

    correct_predictions = 0
    # Используем глобальные данные
    total_samples = len(train_images_global) if train_images_global is not None else 0

    if total_samples == 0:
        # print("Предупреждение: Глобальные данные для обучения не найдены или пусты.")
        return 0.0 # Нечего оценивать

    for i in range(total_samples):
        inputs = train_images_global[i]
        true_label = train_labels_global[i]
        try:
            # Получаем выход сети (10 значений)
            outputs = net.activate(inputs)
            # Предсказанный класс - индекс нейрона с максимальной активацией
            predicted_label = np.argmax(outputs)

            if predicted_label == true_label:
                correct_predictions += 1
        except Exception as e:
            # Ошибка при активации сети может случиться с некоторыми геномами
            # print(f"Ошибка активации сети для генома {genome.key} на данных {i}: {e}")
            # Пропускаем эту оценку, но не прерываем весь процесс
            pass


    # Приспособленность - это точность на обучающем наборе
    fitness = correct_predictions / total_samples if total_samples > 0 else 0.0
    return fitness


# --- 3. Запуск NEAT с использованием ParallelEvaluator ---
def run_neat_parallel(config_file):
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

    # --- Создание ParallelEvaluator ---
    # Укажите количество процессов (можно использовать os.cpu_count())
    # Убедитесь, что функция eval_single_genome доступна
    num_workers = min(os.cpu_count(), 8)
    print(f"Использование {num_workers} процессов для параллельной оценки.")
    # Передаем функцию оценки ОДНОГО генома
    pe = ParallelEvaluator(num_workers, eval_single_genome)

    # --- Запуск процесса эволюции с ParallelEvaluator ---
    # Передаем метод evaluate от ParallelEvaluator
    winner = p.run(pe.evaluate, NUM_GENERATIONS)

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

    try:
        visualize.draw_net(config, winner, True, node_names=node_names, directory=OUTPUT_DIR, filename='net_custom')
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(OUTPUT_DIR, 'avg_fitness_custom.svg'))
        visualize.plot_species(stats, view=False, filename=os.path.join(OUTPUT_DIR, 'speciation_custom.svg'))
        print(f"Графики и структура сети сохранены в папку: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Ошибка визуализации (возможно, не установлен graphviz или его зависимости): {e}")

    return winner, config, stats


def plot_training_stats(stats, output_dir):
    """Визуализация статистики обучения"""
    plt.figure(figsize=(12, 8))

    # График средней и максимальной приспособленности
    plt.subplot(2, 2, 1)
    plt.plot(stats.generation_statistics[::10], stats.get_fitness_mean()[::10], label='Средняя')
    plt.plot(stats.generation_statistics[::10], stats.get_fitness_max()[::10], label='Максимальная')
    plt.xlabel('Поколение')
    plt.ylabel('Приспособленность')
    plt.title('Динамика приспособленности')
    plt.legend()

    # График размера популяции
    plt.subplot(2, 2, 2)
    plt.plot(stats.generation_statistics[::10], stats.num_species[::10])
    plt.xlabel('Поколение')
    plt.ylabel('Количество видов')
    plt.title('Эволюция видов')

    # Сохранение графиков
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_stats.png'))
    plt.close()


def analyze_results(net, test_images, test_labels, output_dir):
    """Анализ результатов и визуализация"""
    # Получение предсказаний
    predictions = []
    for img in test_images:
        output = net.activate(img)
        predictions.append(np.argmax(output))

    # Матрица ошибок
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Предсказанные')
    plt.ylabel('Истинные')
    plt.title('Матрица ошибок')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Отчет классификации
    print("\nОтчет классификации:")
    print(classification_report(test_labels, predictions, digits=3))

    # Визуализация примеров
    plot_sample_predictions(net, test_images, test_labels, output_dir)


def plot_sample_predictions(net, test_images, test_labels, output_dir, num_samples=10):
    """Визуализация примеров предсказаний"""
    plt.figure(figsize=(15, 5))
    indices = np.random.choice(len(test_images), num_samples)

    for i, idx in enumerate(indices):
        img = test_images[idx].reshape(IMAGE_SIZE)
        true_label = test_labels[idx]
        output = net.activate(test_images[idx])
        pred_label = np.argmax(output)
        confidence = output[pred_label]

        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()


# --- Основной блок ---
# ВАЖНО: Обернуть основной код в if __name__ == '__main__': для корректной работы multiprocessing
if __name__ == '__main__':
    # Убедимся, что директория для вывода существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Загрузка данных ---
    print("Загрузка и предобработка обучающих данных...")
    # Используем локальные переменные для загрузки
    local_train_images, local_train_labels = load_and_preprocess_custom_data(os.path.join(DATASET_BASE_PATH, 'train'), IMAGE_SIZE)

    # Проверка, что данные загружены
    if local_train_images.size == 0 or local_train_labels.size == 0:
        # Сообщение об ошибке уже выведено в load_and_preprocess_custom_data
        print("Обучающий датасет пуст или не удалось загрузить. Проверьте путь и формат данных. Выход.")
    else:
        # --- Делаем данные доступными глобально для дочерних процессов ---
        train_images_global = local_train_images
        train_labels_global = local_train_labels
        print(f"Обучающие данные ({len(train_images_global)} образцов) готовы.")


        # --- Запуск NEAT ---
        print(f"\nЗапуск NEAT с {NUM_GENERATIONS} поколениями (параллельно)...")
        # Вызываем функцию для параллельного запуска
        best_genome, neat_config, statistics = run_neat_parallel(CONFIG_PATH)

        # --- Опционально: Оценка лучшего генома на тестовом наборе ---
        print("\nЗагрузка и оценка на тестовом наборе...")
        test_images, test_labels = load_and_preprocess_custom_data(os.path.join(DATASET_BASE_PATH, 'test'), IMAGE_SIZE)

        if test_images.size > 0:
            print("\nАнализ результатов...")
            try:
                winner_net = neat.nn.FeedForwardNetwork.create(best_genome, neat_config)

                # Визуализация статистики обучения
                plot_training_stats(statistics, OUTPUT_DIR)

                # Анализ результатов
                analyze_results(winner_net, test_images, test_labels, OUTPUT_DIR)

                # Дополнительная визуализация NEAT
                node_names = {-i - 1: str(i) for i in range(10)}
                visualize.draw_net(neat_config, best_genome, True, node_names=node_names,
                                   directory=OUTPUT_DIR, filename='network_structure')

            except Exception as e:
                print(f"Ошибка при анализе результатов: {e}")