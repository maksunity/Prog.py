import time
import threading
import io
from flask import Flask, render_template, jsonify, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from simulation_model import RollingMillStand, Furnace

app = Flask(__name__)

mill = None
furnace = None
simulation_state = {"sleep_time": 1.0}
sim_lock = threading.Lock()

sim_thread = None
simulation_running = True

def initialize_simulation():
    global mill, furnace
    mill = RollingMillStand("Mill-Stand-1")
    furnace = Furnace()
    with sim_lock:
        simulation_state["sleep_time"] = 1.0

def run_simulation():
    while simulation_running:
        try:
            if mill:
                mill.update(ingot_temperature=furnace.get_ingot())
            
            with sim_lock:
                current_sleep = simulation_state["sleep_time"]
            
            time.sleep(current_sleep)
        except Exception as e:
            print(f"Ошибка в потоке симуляции: {e}")
            time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    if mill:
        return jsonify(mill.get_state())
    return jsonify({"status": "error", "message": "Симуляция не инициализирована."})

@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    if not mill:
        return jsonify({"status": "error", "message": "Симуляция не инициализирована."}), 500

    if 'speed' in data:
        mill.set_speed(float(data['speed']))
    if 'furnace_temp' in data:
        furnace.set_temperature(float(data['furnace_temp']))
    if 'is_rolling' in data:
        mill.is_rolling = bool(data['is_rolling'])
    if 'cooling_intensity' in data:
        mill.set_cooling_intensity(float(data['cooling_intensity']))
    if 'sim_speed' in data:
        multiplier = float(data['sim_speed'])
        if multiplier > 0:
            with sim_lock:
                simulation_state["sleep_time"] = 1.0 / multiplier
    return jsonify({"status": "ok"})

@app.route('/reset', methods=['POST'])
def reset():
    initialize_simulation()
    return jsonify({"status": "ok", "message": "Симуляция сброшена."})

@app.route('/plot/<component>.png')
def plot_png(component):
    fig = Figure(figsize=(5, 3), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    
    history_data = []
    title = "Неизвестно"

    if mill:
        if component == 'roll1':
            history_data = mill.roll_1.history
            title = "Температура Вала 1"
        elif component == 'roll2':
            history_data = mill.roll_2.history
            title = "Температура Вала 2"
        elif component == 'bearing1':
            history_data = mill.roll_1.bearing.history
            title = "Температура Подшипника 1"
        elif component == 'bearing2':
            history_data = mill.roll_2.bearing.history
            title = "Температура Подшипника 2"

    ax.plot(history_data)
    ax.set_title(title)
    ax.set_ylabel("Температура (°C)")
    ax.grid(True)

    # Ограничение до одного знака после запятой
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    initialize_simulation()
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    app.run(debug=True, use_reloader=False)