import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
from tqdm import tqdm
import time
import math


# Параметры задачи
FILE_PATH = 'products_with_prices_short.csv'
K_PRODUCTS = 4  # Количество продуктов в рационе (k)
# Для N=186, k=5 -> ~200 млн комбинаций (несколько минут)
# Для N=186, k=7 -> ~35 млрд комбинаций (много часов/дней)
# Установите RUN_BRUTE_FORCE = False, если k > 5
RUN_BRUTE_FORCE = True

# Медицинские нормы (соотношение Б:Ж:У = 1:1:4)
PROTEIN_RATIO = 1.0
FAT_RATIO = 1.0
CARB_RATIO = 4.0

# Параметры генетического алгоритма
POPULATION_SIZE = 100
GENERATIONS = 150
MUTATION_RATE = 0.1
# Коэффициент штрафа за отклонение от нормы БЖУ.
# Должен быть достаточно большим, чтобы цена стала второстепенным фактором при плохом БЖУ.
PENALTY_COEFFICIENT = 5000

# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

def load_and_prepare_data(filepath):
    """Загружает данные из CSV и подготавливает их к работе."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filepath}' не найден.")
        return None

    # Очистка и переименование колонок
    df.columns = ['product_name', 'proteins', 'fats', 'carbs', 'calories', 'price']

    # Преобразование в числовой формат, заменяя ошибки на NaN
    for col in ['proteins', 'fats', 'carbs', 'calories', 'price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Удаление строк с пропущенными значениями
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Данные успешно загружены. Всего продуктов: {len(df)}")
    return df

# РЕШЕНИЕ ПОЛНЫМ ПЕРЕБОРОМ (BRUTE FORCE)

def calculate_error(p, f, c):
    """Рассчитывает отклонение от заданной нормы БЖУ."""
    if p == 0:  # Избегаем деления на ноль
        return float('inf')

    # Нормализуем относительно белка
    actual_fat_ratio = f / p
    actual_carb_ratio = c / p

    # Ошибка - сумма абсолютных отклонений от целевых соотношений
    error = abs(actual_fat_ratio - FAT_RATIO / PROTEIN_RATIO) + \
            abs(actual_carb_ratio - CARB_RATIO / PROTEIN_RATIO)
    return error

def solve_brute_force(df, k, tolerance=0.5):
    """
    Находит оптимальный рацион полным перебором.
    Tolerance - допустимое отклонение от нормы БЖУ.
    """
    if not RUN_BRUTE_FORCE:
        print("Полный перебор отключен в конфигурации.")
        return None, None

    print(f"\nЗапуск полного перебора для k={k} продуктов...")
    start_time = time.time()

    best_combination_indices = None
    min_price = float('inf')

    # Вычисляем количество комбинаций математически, не создавая список в памяти
    try:
        num_combinations = math.comb(len(df), k)
    except AttributeError:  # Для старых версий Python, где нет math.comb
        from operator import mul
        from functools import reduce
        if k < 0 or k > len(df):
            num_combinations = 0
        else:
            if k == 0 or k == len(df):
                num_combinations = 1
            elif k > len(df) // 2:
                k = len(df) - k
            num_combinations = reduce(mul, range(len(df) - k + 1, len(df) + 1), 1) // reduce(mul, range(1, k + 1), 1)

    all_combinations = combinations(range(len(df)), k)

    # Используем tqdm для отображения прогресса
    for indices in tqdm(all_combinations, total=num_combinations, desc="Полный перебор"):
        subset = df.iloc[list(indices)]

        total_p = subset['proteins'].sum()
        total_f = subset['fats'].sum()
        total_c = subset['carbs'].sum()
        total_price = subset['price'].sum()

        error = calculate_error(total_p, total_f, total_c)

        if error <= tolerance:
            if total_price < min_price:
                min_price = total_price
                best_combination_indices = list(indices)

    end_time = time.time()
    print(f"Полный перебор завершен за {end_time - start_time:.2f} секунд.")

    if best_combination_indices:
        best_diet = df.iloc[best_combination_indices]
        return best_diet, min_price
    else:
        print(f"Полный перебор не нашел решения с отклонением <= {tolerance}.")
        return None, None

# ГЕНЕТИЧЕСКИЙ АЛГОРИТМ

# Представление индивида и фитнес-функция
def create_individual(n_products, k_products):
    """Создает одного индивида (хромосому) с k выбранными продуктами."""
    individual = np.zeros(n_products, dtype=int)
    chosen_indices = np.random.choice(n_products, k_products, replace=False)
    individual[chosen_indices] = 1
    return individual


def calculate_fitness(individual, df):
    """Рассчитывает приспособленность (фитнес) индивида."""
    selected = df[individual == 1]

    total_price = selected['price'].sum()
    total_p = selected['proteins'].sum()
    total_f = selected['fats'].sum()
    total_c = selected['carbs'].sum()

    error = calculate_error(total_p, total_f, total_c)

    # Фитнес = цена + штраф за отклонение от нормы
    fitness = total_price + PENALTY_COEFFICIENT * error
    return fitness


# Операторы скрещивания (Crossover)

def crossover_single_point(parent1, parent2, k):
    """Одноточечное скрещивание с механизмом 'ремонта' хромосомы."""
    point = random.randint(1, len(parent1) - 1)
    child = np.concatenate([parent1[:point], parent2[point:]])
    return _repair_chromosome(child, k)


def crossover_two_point(parent1, parent2, k):
    """Двухточечное скрещивание с 'ремонтом'."""
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(1, size), 2))
    child = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
    return _repair_chromosome(child, k)


def crossover_uniform(parent1, parent2, k):
    """Равномерное скрещивание с 'ремонтом'."""
    mask = np.random.randint(0, 2, size=len(parent1))
    child = np.where(mask, parent1, parent2)
    return _repair_chromosome(child, k)


def _repair_chromosome(chromosome, k):
    """'Ремонтирует' хромосому, чтобы она содержала ровно k продуктов."""
    ones = np.where(chromosome == 1)[0]
    zeros = np.where(chromosome == 0)[0]

    # Если продуктов больше, чем нужно, случайно убираем лишние
    while len(ones) > k:
        idx_to_remove = np.random.choice(ones)
        chromosome[idx_to_remove] = 0
        ones = np.where(chromosome == 1)[0]

    # Если продуктов меньше, чем нужно, случайно добавляем недостающие
    while len(ones) < k:
        idx_to_add = np.random.choice(zeros)
        chromosome[idx_to_add] = 1
        zeros = np.where(chromosome == 0)[0]
        ones = np.where(chromosome == 1)[0]

    return chromosome


# Операторы мутации (Mutation)

def mutation_swap(individual):
    """Swap-мутация: меняет местами два случайных гена."""
    indices = random.sample(range(len(individual)), 2)
    individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]
    return individual


def mutation_inversion(individual):
    """Инверсионная мутация: инвертирует случайный участок хромосомы."""
    size = len(individual)
    p1, p2 = sorted(random.sample(range(size), 2))
    individual[p1:p2] = individual[p1:p2][::-1]
    return individual


def mutation_scramble(individual):
    """Scramble-мутация: перемешивает гены на случайном участке."""
    size = len(individual)
    p1, p2 = sorted(random.sample(range(size), 2))
    subset = individual[p1:p2]
    np.random.shuffle(subset)
    individual[p1:p2] = subset
    return individual

# Основной цикл генетического алгоритма

def genetic_algorithm(df, k, crossover_fn, mutation_fn):
    """Основная функция, реализующая генетический алгоритм."""
    n_products = len(df)

    # 1. Инициализация популяции
    population = [create_individual(n_products, k) for _ in range(POPULATION_SIZE)]

    best_fitness_history = []

    for generation in range(GENERATIONS):
        # 2. Оценка фитнеса
        fitness_scores = [calculate_fitness(ind, df) for ind in population]

        # Сортируем популяцию по фитнесу (от лучшего к худшему)
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]

        best_fitness = calculate_fitness(sorted_population[0], df)
        best_fitness_history.append(best_fitness)

        # 3. Создание новой популяции
        new_population = []

        # Элитизм: сохраняем 10% лучших индивидов
        elitism_count = int(POPULATION_SIZE * 0.1)
        new_population.extend(sorted_population[:elitism_count])

        # 4. Скрещивание и мутация
        while len(new_population) < POPULATION_SIZE:
            # Турнирный отбор
            p1_idx = min(random.sample(range(POPULATION_SIZE), 5))
            p2_idx = min(random.sample(range(POPULATION_SIZE), 5))
            parent1 = sorted_population[p1_idx]
            parent2 = sorted_population[p2_idx]

            child = crossover_fn(parent1, parent2, k)

            if random.random() < MUTATION_RATE:
                child = mutation_fn(child)

            new_population.append(child)

        population = new_population

    best_individual = sorted_population[0]
    best_fitness = calculate_fitness(best_individual, df)
    best_diet = df[best_individual == 1]

    return best_diet, best_fitness, best_fitness_history


# ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТОВ И ВИЗУАЛИЗАЦИЯ

def run_experiments(df, k):
    """Запускает ГА с разными комбинациями операторов и собирает результаты."""
    crossovers = {
        "Одноточечный": crossover_single_point,
        "Двухточечный": crossover_two_point,
        "Равномерный": crossover_uniform,
    }
    mutations = {
        "Swap": mutation_swap,
        "Inversion": mutation_inversion,
        "Scramble": mutation_scramble,
    }

    results = {}

    for c_name, c_func in crossovers.items():
        for m_name, m_func in mutations.items():
            experiment_name = f"{c_name} + {m_name}"
            print(f"\n--- Запуск эксперимента: {experiment_name} ---")

            _, best_fitness, history = genetic_algorithm(df, k, c_func, m_func)

            results[experiment_name] = {
                "fitness": best_fitness,
                "history": history
            }
            print(f"Результат (фитнес): {best_fitness:.2f}")


    return results


def plot_results(results, brute_force_price):
    """Строит и сохраняет графики по результатам экспериментов."""

    # 1. График сходимости
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, data in results.items():
        ax.plot(data['history'], label=name, alpha=0.8)

    if brute_force_price is not None:
        ax.axhline(y=brute_force_price, color='r', linestyle='--',
                   label=f'Оптимум (полный перебор) = {brute_force_price:.2f} руб.')

    ax.set_title('Сходимость генетического алгоритма для разных методов', fontsize=16)
    ax.set_xlabel('Поколение', fontsize=12)
    ax.set_ylabel('Лучший фитнес (цена + штраф)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('ga_convergence.png')
    print("\nГрафик сходимости сохранен в 'ga_convergence.png'")
    # plt.show() # показать график сразу

    # 2. Гистограмма сравнения итоговых результатов
    fig, ax = plt.subplots(figsize=(14, 8))

    experiment_names = list(results.keys())
    final_fitness = [res['fitness'] for res in results.values()]

    bars = ax.barh(experiment_names, final_fitness, color=plt.cm.viridis(np.linspace(0, 1, len(experiment_names))))

    if brute_force_price is not None:
        ax.axvline(x=brute_force_price, color='r', linestyle='--', label=f'Оптимум = {brute_force_price:.2f} руб.')
        ax.legend()

    ax.set_title('Сравнение итоговых результатов ГА', fontsize=16)
    ax.set_xlabel('Итоговый лучший фитнес (цена + штраф)', fontsize=12)
    ax.set_ylabel('Комбинация методов', fontsize=12)
    ax.invert_yaxis()  # Лучшие результаты сверху
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center')

    plt.tight_layout()
    plt.savefig('ga_results_comparison.png')
    print("Гистограмма сравнения сохранена в 'ga_results_comparison.png'")
    # plt.show() # показать график сразу


def print_diet_info(diet_df, title):
    """Красиво выводит информацию о рационе."""
    if diet_df is None or diet_df.empty:
        print(f"\n--- {title} ---")
        print("Рацион не найден.")
        return

    total_p = diet_df['proteins'].sum()
    total_f = diet_df['fats'].sum()
    total_c = diet_df['carbs'].sum()
    total_price = diet_df['price'].sum()
    error = calculate_error(total_p, total_f, total_c)

    print(f"\n--- {title} ---")
    print("Состав рациона:")
    for _, row in diet_df.iterrows():
        print(f"  - {row['product_name']} ({row['price']:.2f} руб.)")
    print("-" * 30)
    print(f"Итоговая стоимость: {total_price:.2f} руб.")
    print(f"Белки: {total_p:.1f} г | Жиры: {total_f:.1f} г | Углеводы: {total_c:.1f} г")
    if total_p > 0:
        print(f"Соотношение БЖУ: 1 : {total_f / total_p:.2f} : {total_c / total_p:.2f} (цель 1:1:4)")
    print(f"Отклонение от нормы: {error:.3f}")
    print("-" * 30)


if __name__ == "__main__":
    df = load_and_prepare_data(FILE_PATH)

    if df is not None:
        # Решение полным перебором
        brute_force_diet, brute_force_price = solve_brute_force(df, K_PRODUCTS)
        print_diet_info(brute_force_diet, "Оптимальный рацион (Полный перебор)")

        # Решение генетическим алгоритмом
        ga_results = run_experiments(df, K_PRODUCTS)

        # Находим лучший результат среди всех экспериментов ГА
        best_experiment_name = min(ga_results, key=lambda k: ga_results[k]['fitness'])
        print(f"\nЛучший результат ГА получен с помощью: '{best_experiment_name}'")

        # Запускаем лучший ГА еще раз, чтобы получить сам рацион
        c_name, m_name = best_experiment_name.split(' + ')
        crossovers_map = {"Одноточечный": crossover_single_point, "Двухточечный": crossover_two_point,
                          "Равномерный": crossover_uniform}
        mutations_map = {"Swap": mutation_swap, "Inversion": mutation_inversion, "Scramble": mutation_scramble}
        best_c_func = crossovers_map[c_name]
        best_m_func = mutations_map[m_name]

        best_ga_diet, _, _ = genetic_algorithm(df, K_PRODUCTS, best_c_func, best_m_func)
        print_diet_info(best_ga_diet, "Лучший рацион (Генетический алгоритм)")

        # Визуализация
        plot_results(ga_results, brute_force_price)