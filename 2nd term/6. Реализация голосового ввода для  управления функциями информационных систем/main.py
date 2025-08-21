import os
import sys
import wave
import subprocess
import webbrowser

import pyaudio
import whisper
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pymorphy3
import webbrowser
import pyautogui
import pyperclip
import pygetwindow as gw
import time
import requests
import json
import pydirectinput


# Настройка NLTK и лемматизатора
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download("stopwords",quiet=True)
stemmer = SnowballStemmer("russian")
sw = stopwords.words("russian")
morph = pymorphy3.MorphAnalyzer()

# Функция записи аудио через PyAudio
def record_audio(filename: str, record_seconds: int = 7):
    """Записывает record_seconds секунд с микрофона и сохраняет в filename."""
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1
    rate = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=fmt, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk)
    print(f"[Запись аудио] Начинаем запись: {record_seconds} сек...")
    frames = []
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("[Запись аудио] Окончание записи.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(fmt))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Функция распознавания речи с помощью Whisper
def transcribe_audio(model, audio_path: str) -> str:
    result = model.transcribe(audio_path)
    text = result.get('text', '').strip().lower()
    print(f"[Whisper] Распознан текст: «{text}»")
    return text

# Токенизация + лемматизация
def preprocess_text(text: str):
    tokens = word_tokenize(text, language='russian')
    # stemmed_words = [stemmer.stem(word) for word in tokens]
    lemmas = [morph.parse(tok)[0].normal_form for tok in tokens]
    print(f"[NLTK+pymorphy3] Леммы: {lemmas}")
    return lemmas

def generate_search(query: str):
    url = f"https://yandex.ru/search/?text={query.replace(' ', '+')}"
    print(f"[Команда] Открываю Яндекс-поиск: {url}")
    webbrowser.open(url)
def open_video_file(path: str):
    print(f"[Команда] Открываю видеофайл: {path}")
    if sys.platform.startswith('win'):
        os.startfile(path)
    else:
        subprocess.run(['xdg-open', path])
def search_video_online(query: str):
    url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    print(f"[Команда] Открываю YouTube: {url}")
    webbrowser.open(url)

def generate_cpp_sort(alg_name: str, model: str = "qwen2.5-coder:latest"):
    prompts = {
        'пузырьком': "Write C++ implementation of bubble sort with comments.",
        'пузырёк': "Write C++ implementation of bubble sort with comments.",
        'выбором':  "Write C++ implementation of selection sort with comments.",
        'вставками': "Write C++ implementation of insertion sort with comments.",
        'подсчетом': "Write C++ implementation of counting sort with comments.",
        'блочной':   "Write C++ implementation of bucket sort with comments.",
        'слиянием':  "Write C++ implementation of merge sort with comments."
    }
    if alg_name not in prompts:
        print(f"[Ollama] Алгоритм '{alg_name}' не поддерживается")
        return

    prompt = prompts[alg_name]
    print(f"[Ollama] Генерирую C++ для '{alg_name}'")


    # #subprocess
    # result = subprocess.run(
    #     ["ollama", "run", "Qwen-2.5-coder", "--quiet", "--prompt", prompt],
    #     capture_output=True, text=True
    # )
    # code = result.stdout
    # with open("sorting.cpp", "w", encoding="utf-8") as f:
    #     f.write(code)
    # print("[Ollama] Код сохранён в sorting.cpp")

    #requests
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }


    # url = "http://localhost:11434/api/chat"
    # payload = {
    #     "model": model,
    #     "messages":[{"role":"user", "content": prompt}],
    #     "stream": False
    # }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        code = data.get('response')
        if not code:
            print(
                f"[Ollama] ❌ Ответ пустой или не содержит ключ 'response'. Полный ответ: {json.dumps(data, indent=2)}")
            return

        with open("sorting.cpp", "w", encoding="utf-8") as f:
            f.write(code.strip())

        print("[Ollama] ✅ Код успешно сохранён в sorting.cpp!")

    except requests.exceptions.HTTPError as http_err:
        print(f"[Ollama] ❌ HTTP ошибка: {http_err}")
        print(f"[Ollama] ❌ Ответ сервера: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] ❌ Ошибка подключения к серверу Ollama: {e}")
    except json.JSONDecodeError:
        print(f"[Ollama] ❌ Не удалось декодировать JSON ответ от сервера. Ответ: {response.text}")
    except Exception as e:
        print(f"[Ollama] ❌ Непредвиденная ошибка: {e}")

def launch_messenger(app: str, message: str):
    if 'телеграм' in app:
        url = f"https://web.telegram.org/#/im?p=@{message.replace(' ', '%20')}"
    else:
        print(f"[Команда] Неизвестный мессенджер '{app}'")
        return
    print(f"[Команда] Открываю {app}: {url}")
    webbrowser.open(url)

# def open_telegram_dialog(user_id):
#     url = f"https://web.telegram.org/a/#{user_id}"
#     webbrowser.open(url)
#     time.sleep(5)
# def open_telegram_dialog(user_id):
#     window = gw.getWindowsWithTitle('Telegram')[0]
#     pyperclip.copy(user_id) # Нужно использовать имя пользователя
#     buf = pyperclip.copy(user_id)
#     print("Наш пользователь скопированное", buf)
#     time.sleep(2)
#     window.activate()
#     time.sleep(0.5)
#     pyautogui.hotkey('ctrl', 'f')
#     time.sleep(0.5)
#     pyautogui.hotkey('ctrl', 'v')
#     time.sleep(0.5)
#     pyautogui.press('enter')
#
# def send_telegram_message(message):
#     if not message:
#         print("[Ошибка] Сообщение пустое.")
#         return
#
#     print(f"Отправляем сообщение: {message}")
#     time.sleep(2)
#     pyperclip.copy(message)
#     buf=pyperclip.copy(message)
#     print("Наше сообщение скопированное", buf)
#     pyautogui.hotkey('ctrl', 'v')
#     time.sleep(1)
#     pyautogui.press('enter')

def open_telegram_dialog(user_id):
    try:
        # Проверяем наличие окна Telegram
        windows = gw.getWindowsWithTitle('Telegram')
        if not windows:
            raise Exception("Окно Telegram не найдено")

        window = windows[0]
        window.activate()
        pyperclip.copy(user_id)
        copied_data = pyperclip.paste()
        print("Скопированный пользователь:", copied_data)
        time.sleep(0.2)
        # pyautogui.hotkey('ctrl', 'f')
        pydirectinput.keyDown('ctrl')
        pydirectinput.press('f')
        pydirectinput.keyUp('ctrl')
        time.sleep(0.1)
        # pyautogui.hotkey('ctrl', 'v')
        pydirectinput.keyDown('ctrl')
        pydirectinput.press('v')
        pydirectinput.keyUp('ctrl')
        time.sleep(0.1)
        pydirectinput.press('enter')
        time.sleep(0.5)
    except Exception as e:
        print(f"[Ошибка]: {str(e)}")


def send_telegram_message(message):
    if not message:
        print("[Ошибка] Сообщение пустое.")
        return

    try:
        pyperclip.copy(message)
        copied_msg = pyperclip.paste()
        print("Скопированное сообщение:", copied_msg)
        time.sleep(0.5)
        pydirectinput.keyDown('ctrl')
        pydirectinput.press('v')
        pydirectinput.keyUp('ctrl')
        time.sleep(0.5)
        pydirectinput.press('enter')
        time.sleep(0.5)
    except Exception as e:
        print(f"[Ошибка отправки]: {str(e)}")

def main(from_file=False, text_file_path="input.txt"):
    if not from_file:
        print("[Инициализация] Загружаем модель Whisper...")
        model = whisper.load_model("base")

    # Списки шаблонных ключевых слов
    search_patterns = ['найти', 'поиск', 'найди', 'ищи', 'ищет']
    browser_patterns = ['в браузере','в браузер', 'через браузер', 'яндекс', 'в яндексе']
    youtube_patterns = ['на ютубе', 'youtube', 'на youtube', 'ютуб', 'ютуба', ' на ютуба']

    SEARCH_TRIGGERS = ['найти', 'найди', 'поиск', 'искать', 'поисковый', 'яндекс', 'в яндексе']
    BROWSER_WORDS = ['браузер', 'браузере', 'интернет', 'сеть']
    YOUTUBE_WORDS = ['youtube', 'ютуб', 'видео']
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']

    messenger_apps = ('телеграм', 'вк', 'вконтакте')

    TELEGRAM_USERS = {
        "я": 'избранное'
    }
    sort_algorithms = ('пузырьком', 'пузырёк', 'выбором', 'вставками', 'подсчетом', 'блочной', 'слиянием')

    def process_text(text):
        text = text.strip().lower()
        print(f"[Текстовый ввод] Прочитано: «{text}»")
        lemmas = preprocess_text(text)
        lemmas = [lemma for lemma in lemmas if lemma.isalpha()]
        normalized_text = ' '.join(lemmas)

        # Поиск в браузере (Яндекс)
        if any(trigger in lemmas for trigger in SEARCH_TRIGGERS):
            if any(word in lemmas for word in BROWSER_WORDS):
                idx = max(lemmas.index(w) for w in BROWSER_WORDS if w in lemmas)
                query = ' '.join(lemmas[idx + 1:])
                generate_search(query)
                return

        # Поиск видео на YouTube
        if any(pattern in lemmas for pattern in search_patterns) and any(
                phrase in normalized_text for phrase in youtube_patterns):
            idx = min(lemmas.index(word) for word in search_patterns if word in lemmas)
            query = ' '.join(lemmas[idx + 3:])
            search_video_online(query)
            return

        # Открыть локальное видео
        if 'открыть' in lemmas and 'видео' in lemmas and not any(
                phrase in normalized_text for phrase in youtube_patterns):
            idx = lemmas.index('видео')
            target_base = ' '.join(lemmas[idx + 1:]).strip()

            if not target_base:
                print("[Ошибка] Название видео не указано.")
                return

            found = False
            for ext in VIDEO_EXTENSIONS:
                candidate = f"{target_base}{ext}"
                if os.path.exists(candidate):
                    open_video_file(candidate)
                    found = True
                    break

            if not found:
                print(f"[Ошибка] Видео файл '{target_base}' не найден в известных расширениях {VIDEO_EXTENSIONS}.")
            return

        # Отправка сообщения в мессенджере
        # if 'написать' in lemmas and 'сообщение' in lemmas:
        #     try:
        #         idx = lemmas.index('сообщение')
        #         user_phrase = ' '.join(lemmas[lemmas.index('человеку') + 1:idx])
        #         message = ' '.join(lemmas[idx + 1:])
        #
        #         user_id = TELEGRAM_USERS.get(user_phrase)
        #         if user_id is not None:
        #             open_telegram_dialog(user_id)
        #             send_telegram_message(message)
        #         else:
        #             print(f"[Ошибка] Пользователь {user_phrase} не найден в словаре.")
        #     except Exception as e:
        #         print(f"[Ошибка обработки Telegram-команды]: {e}")
        #     return
        if 'написать' in lemmas and 'сообщение' in lemmas:
            try:
                idx = lemmas.index('сообщение')

                # Определяем начало поиска получателя
                if 'человеку' in lemmas:
                    start_idx = lemmas.index('человеку') + 1
                else:
                    start_idx = idx + 1
                user_phrase_parts = []
                current_idx = start_idx
                user_found = False
                while current_idx < len(lemmas):
                    lemma = lemmas[current_idx]
                    if lemma in ('в', 'на', 'из', 'тг', 'telegram', 'телеграм', 'теле'):
                        print(f"Пропускаем служебное: {lemma}")
                        current_idx += 1
                        continue
                    if lemma in TELEGRAM_USERS:
                        user_phrase_parts.append(lemma)
                        user_found = True
                        current_idx += 1
                        break
                    user_phrase_parts.append(lemma)
                    current_idx += 1

                user_phrase = ' '.join(user_phrase_parts).strip()
                print("Найден получатель:", user_phrase)

                # Определяем начало сообщения
                message_start_idx = current_idx

                # Собираем сообщение с фильтрацией служебных слов
                message_parts = []
                for lemma in lemmas[message_start_idx:]:
                    if lemma in ('в', 'на', 'из', 'тг', 'telegram', 'телеграм', 'теле'):
                        continue
                    message_parts.append(lemma)

                message = ' '.join(message_parts).strip()
                print("Текст сообщения:", message)

                # Отправка сообщения
                if user_phrase in TELEGRAM_USERS:
                    print("Ему пишем", TELEGRAM_USERS[user_phrase])
                    open_telegram_dialog(TELEGRAM_USERS[user_phrase])
                    if message:
                        send_telegram_message(message)
                else:
                    print(f"[Ошибка] Пользователь '{user_phrase}' не найден")

            except Exception as e:
                print(f"[Ошибка обработки команды]: {e}")
            return

        # Генерация кода сортировки
        for alg in sort_algorithms:
            if alg in lemmas:
                generate_cpp_sort(alg)
                return

        print("[Ожидание] Нет подходящей команды...")

    if from_file:
        if not os.path.exists(text_file_path):
            print(f"[Ошибка] Файл {text_file_path} не найден.")
            return
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        process_text(text)
    else:
        while True:
            record_audio("command.wav", record_seconds=7)
            text = transcribe_audio(model, "command.wav")
            if text:
                process_text(text)


if __name__ == "__main__":
    main(from_file=False)
#
#     while True:
#         if from_file:
#             if not os.path.exists(text_file_path):
#                 print(f"[Ошибка] Файл {text_file_path} не найден.")
#                 break
#             with open(text_file_path, "r", encoding="utf-8") as f:
#                 text = f.read().strip().lower()
#             print(f"[Текстовый ввод] Прочитано из файла: «{text}»")
#         else:
#             record_audio("command.wav", record_seconds=7)
#             text = transcribe_audio(model, "command.wav")
#         if not text:
#             continue
#
#         lemmas = preprocess_text(text)
#         lemmas = [lemma for lemma in lemmas if lemma.isalpha()]  # убираем лишние символы
#
#         normalized_text = ' '.join(lemmas)  # для удобства поиска фраз
#
#         # # Поиск в браузере (Яндекс)
#         if any(trigger in lemmas for trigger in SEARCH_TRIGGERS):
#             if any(word in lemmas for word in BROWSER_WORDS):
#                 # найти в браузере что-то
#                 idx = max(lemmas.index(w) for w in BROWSER_WORDS if w in lemmas)
#                 query = ' '.join(lemmas[idx + 1:])
#                 generate_search(query)
#                 continue
#
#         # # Поиск видео на YouTube
#         if any(pattern in lemmas for pattern in search_patterns) and any(
#                 phrase in normalized_text for phrase in youtube_patterns):
#             idx = min(lemmas.index(word) for word in search_patterns if word in lemmas)
#             query = ' '.join(lemmas[idx + 1:])
#             search_video_online(query)
#             continue
#
#
#         # # Открыть видеофайл или найти видео
#         # if 'открыть' in lemmas and 'видео' in lemmas:
#         #     idx = lemmas.index('видео')
#         #     target = ' '.join(lemmas[idx + 1:])
#         #     if os.path.exists(target):
#         #         open_video_file(target)
#         #     else:
#         #         search_video_online(target)
#         #     continue
#         if 'открыть' in lemmas and 'видео' in lemmas and not any(
#                 phrase in normalized_text for phrase in youtube_patterns):
#             idx = lemmas.index('видео')
#             target_base = ' '.join(lemmas[idx + 1:]).strip()
#
#             if not target_base:
#                 print("[Ошибка] Название видео не указано.")
#                 continue
#
#             found = False
#             for ext in VIDEO_EXTENSIONS:
#                 candidate = f"{target_base}{ext}"
#                 if os.path.exists(candidate):
#                     open_video_file(candidate)
#                     found = True
#                     break
#
#             if not found:
#                 print(f"[Ошибка] Видео файл '{target_base}' не найден в известных расширениях {VIDEO_EXTENSIONS}.")
#             continue
#
#         # # Отправка сообщения в мессенджере
#         # for app in messenger_apps:
#         #     if app in lemmas and ('сообщение' in lemmas or 'отправить' in lemmas):
#         #         if 'сообщение' in lemmas:
#         #             idx = lemmas.index('сообщение')
#         #         else:
#         #             idx = lemmas.index('отправить')
#         #         msg = ' '.join(lemmas[idx + 1:])
#         #         launch_messenger(app, msg)
#         #         break
#
#         if 'напиши' in lemmas and 'сообщение' in lemmas:
#             try:
#                 idx = lemmas.index('сообщение')
#                 user_phrase = ' '.join(lemmas[lemmas.index('человеку') + 1:idx])
#                 message = ' '.join(lemmas[idx + 1:])
#
#                 user_id = TELEGRAM_USERS.get(user_phrase)
#                 if user_id is not None:
#                     open_telegram_dialog(user_id)
#                     send_telegram_message(message)
#                 else:
#                     print(f"[Ошибка] Пользователь {user_phrase} не найден в словаре.")
#             except Exception as e:
#                 print(f"[Ошибка обработки Telegram-команды]: {e}")
#             continue
#
#         # Генерация C++ сортировки
#         for alg in sort_algorithms:
#             if alg in lemmas:
#                 generate_cpp_sort(alg)
#                 break
#
#         print("[Ожидание] Готов к новой команде...")
#
# if __name__ == "__main__":
#     main(from_file=True)

