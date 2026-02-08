import numpy as np
import random
import vrplib
import re
from pathlib import Path
import copy
from collections import defaultdict
import time

# Фиксируем сид для воспроизводимости
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

folder_path = Path("./Tests")

def read_vrp(path):
    # Парсит файл с данными
    instance = vrplib.read_instance(path)

    name = instance['name']
    dimension = instance['dimension']
    capacity = instance['capacity']
    w_type = instance['edge_weight_type']

    comment = instance['comment']
    match = re.search(r'(?:No|Min no) of trucks: (\d+).*(?:Best|Optimal) value: (\d+)', comment)
    m = int(match.group(1))
    ub = int(match.group(2))

    if w_type == 'EUC_2D':
        coords = np.array(instance['node_coord'])
        dist_matrix = None
    elif w_type in ['EXPLICIT', 'FULL_MATRIX']:
        coords = None
        dist_matrix = np.array(instance['edge_weight'])

    demands = np.array(instance['demand'])

    depot_node = instance['depot'][0]

    # print("Загружено:", name)
    # print("Клиентов:", dimension - 1)  # минус депо
    # print("Ёмкость:", capacity)
    # print("Расчет расстояния по системе:", w_type)
    # print(f'Координаты точек: {coords}')
    # print(f'Спрос клиентов: {demands}')
    # print(f'Кол-во машин: {m}, оптимальный результат: {ub}')
    # print(f'Индекс депо: {depot_node}')

    return name, dimension, capacity, m, ub, coords, dist_matrix, demands, depot_node, w_type

def euc_2d_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def euc_2d_matrix(coords):
    N = len(coords)
    dist_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = euc_2d_distance(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1]
            )

    return dist_matrix

# Парсим данные из файлов для дальнейшей работы
vrp_data = {}
vrp_files = [f for f in folder_path.iterdir() if f.suffix == '.vrp']
for vrp_file in vrp_files:
    name, dimension, capacity, K, opt_value, coords, distance_matrix, demands, depot_idx, weight_type = read_vrp(str(vrp_file))
    if weight_type == 'EUC_2D':
        distance_matrix = euc_2d_matrix(coords)

    vrp_data[name] = [dimension, capacity, K, opt_value, coords, demands, depot_idx, weight_type, distance_matrix]

def calculate_route_cost(route, distance_matrix, depot_idx):
    # Вычисляет сумму расстояний
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]
    cost += distance_matrix[route[-1]][depot_idx]  # возврат в депо
    return cost


def calculate_total_cost(solution, distance_matrix, depot_idx):
    # Вычисляет общее расстояние
    total_cost = 0.0
    for route in solution:
        total_cost += calculate_route_cost(route, distance_matrix, depot_idx)
    return total_cost


def is_feasible_route(route, demands, capacity, depot_idx):
    # Проверяет выполнимость маршрута по грузоподъемности
    if route[0] != depot_idx or route[-1] != depot_idx:
        return False
    current_load = 0.0
    for customer in route[1:-1]:  # исключаем депо
        idx = int(customer)
        current_load += demands[idx]
        if current_load > capacity:
            return False
    return True

def clarke_wright_savings(dimension, depot_idx, demands, capacity, distance_matrix):
    # Генерирует начальное решение
    customers = [i for i in range(dimension) if i != depot_idx]

    savings = defaultdict(list)
    for i in customers:
        for j in customers:
            if i != j:
                saving = (distance_matrix[i][depot_idx] +
                          distance_matrix[depot_idx][j] -
                          distance_matrix[i][j])
                savings[saving].append((i, j))

    sorted_savings = sorted(savings.items(), reverse=True)

    routes = [[depot_idx, i, depot_idx] for i in customers]
    route_dict = {tuple(route): idx for idx, route in enumerate(routes)}

    for saving, pairs in sorted_savings:
        for i, j in pairs:
            route_i = route_j = None
            pos_i = pos_j = -1

            for route_tup, idx in route_dict.items():
                route = list(route_tup)
                if i in route[1:-1] and route_j is None:
                    route_i = idx
                    pos_i = route.index(i)
                if j in route[1:-1] and route_i != idx:
                    route_j = idx
                    pos_j = route.index(j)

            if route_i is not None and route_j is not None and pos_i != -1 and pos_j != -1:
                route1 = routes[route_i]
                route2 = routes[route_j]

                new_route = route1[:-1] + route2[1:]
                if is_feasible_route(new_route, demands, capacity, depot_idx):
                    routes[route_i] = new_route
                    del routes[route_j]

                    route_dict = {tuple(r): idx for idx, r in enumerate(routes)}
                    break

    routes = [r for r in routes if len(r) > 2]
    return routes


def local_improvement(solution, distance_matrix, demands, capacity, depot_idx, max_iters=50):
    # Ищет локальное решение, которое может быть лучше
    best_solution = copy.deepcopy(solution)
    improved = True

    while improved and max_iters > 0:
        improved = False
        max_iters -= 1

        for route_idx in range(len(best_solution)):
            route = best_solution[route_idx]
            if len(route) < 4:
                continue

            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    if is_feasible_route(new_route, demands, capacity, depot_idx):
                        new_cost = calculate_route_cost(new_route, distance_matrix, depot_idx)
                        old_cost = calculate_route_cost(route, distance_matrix, depot_idx)

                        if new_cost < old_cost:
                            best_solution[route_idx] = new_route
                            improved = True
                            route = new_route

    return best_solution

def get_neighborhood(solution, demands, capacity, depot_idx):
    # Генерирует несколько попыток и возвращает лучшее валидное соседство
    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        new_solution = copy.deepcopy(solution)
        attempts += 1

        route1_idx = random.randint(0, len(new_solution) - 1)
        route2_idx = random.randint(0, len(new_solution) - 1)

        op = random.choice(['relocate', 'exchange', '2opt', 'intra_relocate'])

        if op == 'relocate' and route1_idx != route2_idx and len(new_solution[route1_idx]) > 2:
            route1 = new_solution[route1_idx]
            route2 = new_solution[route2_idx]

            cust1_idx = random.randint(1, len(route1) - 2)
            customer = route1[cust1_idx]

            route1.pop(cust1_idx)

            insert_pos = random.randint(1, len(route2) - 1)
            route2.insert(insert_pos, customer)

            if (is_feasible_route(route1, demands, capacity, depot_idx) and
                    is_feasible_route(route2, demands, capacity, depot_idx)):
                return new_solution

            route2.pop(insert_pos)
            route1.insert(cust1_idx, customer)

        elif op == 'exchange' and route1_idx != route2_idx:
            route1 = new_solution[route1_idx]
            route2 = new_solution[route2_idx]

            if len(route1) > 2 and len(route2) > 2:
                cust1_idx = random.randint(1, len(route1) - 2)
                cust2_idx = random.randint(1, len(route2) - 2)

                customer1 = route1[cust1_idx]
                customer2 = route2[cust2_idx]

                route1[cust1_idx], route2[cust2_idx] = customer2, customer1

                if (is_feasible_route(route1, demands, capacity, depot_idx) and
                        is_feasible_route(route2, demands, capacity, depot_idx)):
                    return new_solution

                route1[cust1_idx], route2[cust2_idx] = customer1, customer2

        elif op == '2opt':
            route_idx = random.randint(0, len(new_solution) - 1)
            route = new_solution[route_idx]

            if len(route) > 4:
                i = random.randint(1, len(route) - 3)
                j = random.randint(i + 1, len(route) - 2)

                new_route = route[:i] + route[i:j][::-1] + route[j:]

                if is_feasible_route(new_route, demands, capacity, depot_idx):
                    new_solution[route_idx] = new_route
                    return new_solution

        elif op == 'intra_relocate':
            route_idx = random.randint(0, len(new_solution) - 1)
            route = new_solution[route_idx]

            if len(route) > 4:
                old_pos = random.randint(1, len(route) - 3)
                new_pos = random.randint(1, len(route) - 2)
                if new_pos >= old_pos:
                    new_pos += 1

                customer = route.pop(old_pos)
                route.insert(new_pos, customer)

                if is_feasible_route(route, demands, capacity, depot_idx):
                    return new_solution

    return None

def simulated_annealing(dimension, capacity, demands, depot_idx, distance_matrix,
                        initial_temp=None, cooling_rate=0.998, min_temp=0.01,
                        iterations_per_temp=500, max_iter=200000, restarts=3):

    # Реализация метода Simulated Annealing
    if initial_temp is None:
        base_temp = calculate_total_cost(
            [[depot_idx, i, depot_idx] for i in range(1, dimension)],
            distance_matrix, depot_idx
        )
        initial_temp = base_temp * 0.1

    best_global_cost = float('inf')
    best_global_solution = None

    for restart in range(restarts):
        current_solution = clarke_wright_savings(dimension, depot_idx, demands, capacity, distance_matrix)
        current_solution = local_improvement(current_solution, distance_matrix, demands, capacity, depot_idx)
        current_cost = calculate_total_cost(current_solution, distance_matrix, depot_idx)

        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost

        temp = initial_temp
        iter_count = 0
        no_improvement = 0

        while temp > min_temp and iter_count < max_iter:
            for _ in range(iterations_per_temp):
                neighbor = get_neighborhood(current_solution, demands, capacity, depot_idx)

                if neighbor is not None:
                    neighbor_cost = calculate_total_cost(neighbor, distance_matrix, depot_idx)
                    delta_cost = neighbor_cost - current_cost

                    if delta_cost < 0 or random.random() < np.exp(-delta_cost / temp):
                        current_solution = neighbor
                        current_cost = neighbor_cost

                        if current_cost < best_cost:
                            best_solution = copy.deepcopy(current_solution)
                            best_cost = current_cost
                            no_improvement = 0
                        else:
                            no_improvement += 1

                iter_count += 1
                if iter_count >= max_iter or no_improvement > 2000:
                    break

            temp *= cooling_rate

        if best_cost < best_global_cost:
            best_global_cost = best_cost
            best_global_solution = best_solution

    return best_global_cost, best_global_solution

def run_simulated_annealing_on_all_tasks(vrp_data):
    # Запускает Simulated Annealing для всех задач из словаря vrp_data
    results = {}
    total_tasks = len(vrp_data)
    i = 0

    for task_name, task_info in vrp_data.items():
        start_time = time.time()
        dimension, capacity, K, opt_value, coords, demands, depot_idx, weight_type, distance_matrix = task_info

        print(f"\n[{i + 1}/{total_tasks}] Решение задачи: {task_name}")
        print(f"   Размер: {dimension}, Количество маршрутов: {K}, Грузоподъемность: {capacity}")

        # Запускаем SA
        cost, solution = simulated_annealing(
            dimension=dimension,
            capacity=capacity,
            demands=demands,
            depot_idx=depot_idx,
            distance_matrix=distance_matrix,
            initial_temp=500.0,
            cooling_rate=0.998,
            min_temp=0.01,
            iterations_per_temp=300,
            max_iter=200000
        )

        solve_time = time.time() - start_time

        deviation = abs((cost - opt_value) / opt_value) * 100

        results[task_name] = {
            'found_cost': cost,
            'optimal_cost': opt_value,
            'deviation_percent': deviation,
            'solution': solution,
            'num_routes': len(solution)
        }

        print(f"   SA: {cost:.2f} | UB: {opt_value:.2f} | Кол-во маршрутов: {K} | Отклонение: {deviation:.2f}%")
        print(f"   Время: {solve_time:.2f}с | Прогресс: {((i + 1) / total_tasks) * 100:.1f}%")

        i+=1

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА ПО ВСЕМ ЗАДАЧАМ")
    print("=" * 60)

    deviations = []
    total_deviation = 0

    for task_name, result in results.items():
        dev = result['deviation_percent']
        deviations.append(dev)
        total_deviation += dev
        print(f"{task_name}: {result['found_cost']:.2f} (опт: {result['optimal_cost']:.2f}), отклонение: {dev:.2f}%")

    avg_deviation = total_deviation / len(results)
    print(f"\nСреднее отклонение: {avg_deviation:.2f}%")

    return results

# Итоговый расчет по всем задачам
results = run_simulated_annealing_on_all_tasks(vrp_data)