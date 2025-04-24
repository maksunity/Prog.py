import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_graph(num_points):
    points = {i: (random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_points)}
    roads = {}
    for i in points:
        for j in points:
            if i != j:
                if random.random() > 0.5:  # С вероятностью 50% создаем связь
                    roads[(i, j)] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
    return points, roads

class AntColony:
    def __init__(self, points, roads, n_ants=50, alpha=1, beta=2, evaporation_rate=0.5, iterations=100):
        self.points = points
        self.roads = roads
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromones = {key: 1.0 for key in roads}

    def run(self):
        best_path = None
        best_length = float('inf')

        for _ in range(self.iterations):
            all_paths = self.construct_solutions()
            self.update_pheromones(all_paths)

            for path, length in all_paths:
                if length < best_length:
                    best_length = length
                    best_path = path

        return best_path, best_length

    def construct_solutions(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.generate_path()
            if path:
                length = self.calculate_path_length(path)
                all_paths.append((path, length))
        return all_paths

    def generate_path(self):
        unvisited = set(self.points.keys())
        current = random.choice(list(self.points.keys()))
        path = [current]
        unvisited.remove(current)

        while unvisited:
            probabilities = self.calculate_probabilities(current, unvisited)
            if sum(probabilities) == 0:
                return None
            next_point = random.choices(list(unvisited), weights=probabilities, k=1)[0]
            path.append(next_point)
            unvisited.remove(next_point)
            current = next_point

        return path

    def calculate_probabilities(self, current, unvisited):
        probabilities = []
        for next_point in unvisited:
            edge = (current, next_point)
            pheromone = self.pheromones.get(edge, 1.0)
            distance = self.roads.get(edge, float('inf'))
            if distance == float('inf'):
                probabilities.append(0)
            else:
                probabilities.append((pheromone ** self.alpha) * ((1 / distance) ** self.beta))
        return probabilities

    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            length += self.roads.get(edge, float('inf'))
        return length

    def update_pheromones(self, all_paths):
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.evaporation_rate)

        for path, length in all_paths:
            if length > 0:  # Избегаем деления на ноль
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    self.pheromones[edge] += 1 / length

def visualize_solution(points, path):
    if not path:
        print("Нет допустимого пути для визуализации.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = zip(*[points[p] for p in path])
    ax.plot(x, y, z, color='blue', marker='o')

    for i, coord in points.items():
        ax.text(coord[0], coord[1], coord[2], str(i))

    plt.show()


num_points = 100 #укажите кол-во точек
points, roads = generate_graph(num_points)
colony = AntColony(points, roads, iterations=200)
best_path, best_length = colony.run()
if best_path:
    print(f"Лучший путь: {best_path}\nДлина пути: {best_length}")
    visualize_solution(points, best_path)
else:
    print("Не удалось найти допустимый путь.")
