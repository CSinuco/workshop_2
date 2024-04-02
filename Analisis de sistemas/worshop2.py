import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Función para generar ciudades con coordenadas aleatorias en 3D
def generate_cities(number_cities):
    cities = []
    for _ in range(number_cities):
        city = np.random.rand(3)  # Generar coordenadas aleatorias en 3D para la ciudad
        cities.append(city)
    return cities

# Función para calcular la distancia euclidiana entre dos puntos
def calculate_distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)

# Función para la optimización de colonias de hormigas (ACO) para resolver el TSP
def ant_colony_optimization(cities, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    number_cities = len(cities)
    pheromone = np.ones((number_cities, number_cities))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * number_cities
            current_city = np.random.randint(number_cities)
            visited[current_city] = True
            path = [current_city]
            path_length = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                for i, unvisited_city in enumerate(unvisited):
                    probabilities[i] = (pheromone[current_city, unvisited_city] ** alpha) * \
                                        ((1 / calculate_distance(cities[current_city], cities[unvisited_city])) ** beta)

                probabilities /= probabilities.sum()

                next_city = np.random.choice(unvisited, p=probabilities)
                path.append(next_city)
                path_length += calculate_distance(cities[current_city], cities[next_city])
                visited[next_city] = True
                current_city = next_city

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        pheromone *= evaporation_rate

        for path, path_length in zip(paths, path_lengths):
            for i in range(number_cities - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length

    return best_path, best_path_length

# Función para generar colores aleatorios para visualización
def random_color():
    return [random.random(), random.random(), random.random()]

# Función para visualizar la ruta encontrada por ACO
def plot_aco_route(cities, best_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(len(best_path) - 1):
        ax.plot(
            [cities[best_path[i], 0], cities[best_path[i + 1], 0]],
            [cities[best_path[i], 1], cities[best_path[i + 1], 1]],
            [cities[best_path[i], 2], cities[best_path[i + 1], 2]],
            c=random_color(),
            linestyle="-",
            linewidth=3,
        )

    ax.plot(
        [cities[best_path[0], 0], cities[best_path[-1], 0]],
        [cities[best_path[0], 1], cities[best_path[-1], 1]],
        [cities[best_path[0], 2], cities[best_path[-1], 2]],
        c=random_color(),
        linestyle="-",
        linewidth=3,
    )

    ax.scatter(cities[0, 0], cities[0, 1], cities[0, 2], c="b", marker="o", label="Start")
    ax.scatter(cities[1:, 0], cities[1:, 1], cities[1:, 2], c="g", marker="o", label="Cities")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude")
    ax.legend()
    plt.show()

# Parámetros del modelo
number_cities = 30
number_ants = 100
number_iterations = 100
alpha = 1
beta = 1
evaporation_rate = 0.5
Q = 1

# Generar lista de ciudades
cities = generate_cities(number_cities)

# Ejecutar la optimización de colonias de hormigas
best_path, best_path_length = ant_colony_optimization(
    cities, number_ants, number_iterations, alpha, beta, evaporation_rate, Q
)

# Imprimir la mejor ruta y su longitud
print("Best path:", best_path)
print("Best path length:", best_path_length)

# Visualizar la ruta encontrada por ACO
plot_aco_route(np.array(cities), best_path)

