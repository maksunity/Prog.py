import mss
import pytesseract
import cv2
import numpy as np
import time
import base64
from flask import Flask, render_template_string
import threading
import signal
import sys
import re

# --- Настройки ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SCAN_INTERVAL = 7
MONITOR_ID = 1
WEB_DOWNSCALE_WIDTH = 1920
TOP_SKIP_PIXELS = 80
BOTTOM_SKIP_PIXELS = 90

# --- Flask ---
app = Flask(__name__)
latest_text = ""
latest_image_base64 = ""
latest_questions = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Screen Text Recognition - Вопросы</title>
    <style>
        body { font-family: Arial; margin: 20px; background-color: #f0f0f0; }
        #text-container, #image-container, #questions-container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        #text-container { white-space: pre-wrap; }
        img { max-width: 100%; border-radius: 5px; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <h1>Распознанный текст с экрана</h1>
    <div id="text-container">{{ text }}</div>

    <h2>Скриншот</h2>
    <div id="image-container">
        <img src="data:image/png;base64,{{ image }}" alt="Скриншот">
    </div>

    <h2>Выделенные вопросы</h2>
    <div id="questions-container">
        {% for q in questions %}
            <pre>{{ q }}</pre>
        {% endfor %}
    </div>
    <button id="copy-button" onclick="copyQuestions()">Скопировать вопросы</button>


     <script>
    function copyQuestions() {
        const questionsContainer = document.getElementById('questions-container');
        if (!questionsContainer) {
            alert('Ошибка: контейнер с вопросами не найден');
            return;
        }

        const questions = questionsContainer.getElementsByTagName('pre');
        if (!questions || questions.length === 0) {
            alert('Нет вопросов для копирования');
            return;
        }

        const textToCopy = Array.from(questions)
            .map(pre => pre.textContent || pre.innerText)
            .join('\\n\\n');

        if (navigator.clipboard && window.isSecureContext) {
            // Современный метод
            navigator.clipboard.writeText(textToCopy)
                .catch(err => {
                    console.error('Ошибка копирования:', err);
                    fallbackCopyTextToClipboard(textToCopy);
                });
        } else {
            // Fallback для старых браузеров
            fallbackCopyTextToClipboard(textToCopy);
        }
    }

    function fallbackCopyTextToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        
        try {
            textArea.select();
            document.execCommand('copy');
        } catch (err) {
            console.error('Ошибка при копировании:', err);
            alert('Не удалось скопировать вопросы. Пожалуйста, выделите текст вручную.');
        } finally {
            document.body.removeChild(textArea);
        }
    }

    // Автообновление страницы
    setTimeout(() => { location.reload(); }, 7000);
    </script>
</body>
</html>
"""

# < script >
# function
# copyQuestions()
# {
#     const
# text = Array.
# from
#
# (document.querySelectorAll('#questions-container p'))
# .map(p= > p.innerText)
# .join('\\n');
# navigator.clipboard.writeText(text)
# .then(() = > alert('Вопросы скопированы!'))
# .catch(err= > alert('Ошибка при копировании: ' + err));
# }
#
# // Автообновление
# страницы
# каждые
# 7
# секунд
# setTimeout(function()
# {location.reload();}, 7000);
# < / script >


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        text=latest_text,
        image=latest_image_base64,
        questions=latest_questions
    )


# --- OCR + скриншот ---
def capture_text_and_image():
    with mss.mss() as sct:
        monitor = sct.monitors[MONITOR_ID]
        skip_top = min(TOP_SKIP_PIXELS, monitor["height"] - 1)
        max_bottom_possible = monitor["height"] - skip_top - 1
        skip_bottom = min(BOTTOM_SKIP_PIXELS, max_bottom_possible if max_bottom_possible > 0 else 0)
        remaining_height = max(1, monitor["height"] - skip_top - skip_bottom)

        region = {
            "top": monitor["top"] + skip_top,
            "left": monitor["left"],
            "width": monitor["width"],
            "height": remaining_height
        }

        img_full = np.array(sct.grab(region))
        gray = cv2.cvtColor(img_full, cv2.COLOR_BGRA2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, lang='rus+eng', config='--oem 3 --psm 6').strip()

        height = int(img_full.shape[0] * (WEB_DOWNSCALE_WIDTH / img_full.shape[1]))
        resized_img = cv2.resize(img_full, (WEB_DOWNSCALE_WIDTH, height), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.png', resized_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return text, image_base64


# --- Выделяем вопросы ---
def extract_question(text):
    pattern = r"ВОПРОС\s*\d*:\s*(.+?)(?=(ВОПРОС\s*\d*:|$))"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [m[0].strip() for m in matches]


# --- Основной цикл OCR ---
def ocr_loop():
    global latest_text, latest_image_base64, latest_questions
    previous_text = ""
    while True:
        text, image_b64 = capture_text_and_image()
        latest_questions = extract_question(text)
        if text != previous_text:
            latest_text = text
            latest_image_base64 = image_b64

            previous_text = text
            print(f"[OCR обновление] {time.strftime('%H:%M:%S')}\nВопросы:\n{latest_questions}\n{'-' * 50}")
        time.sleep(SCAN_INTERVAL)


# --- Завершение по Ctrl+C ---
def signal_handler(sig, frame):
    print("\n[Выход] Программа остановлена.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# --- Запуск ---
if __name__ == "__main__":
    threading.Thread(target=ocr_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
