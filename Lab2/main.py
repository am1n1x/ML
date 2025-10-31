import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Входная переменная "Освещенность помещения"
# Физический диапазон: от 0 (полная темнота) до 1000 люкс
illumination = ctrl.Antecedent(np.arange(0, 1001, 1), 'Освещенность')

# Входная переменная "Время суток"
# Физический диапазон: от 0 до 24 часов.
time_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'Время суток')

# Выходная переменная "Мощность лампы"
# Физический диапазон: от 0% (выключена) до 100% (максимальная яркость)
lamp_power = ctrl.Consequent(np.arange(0, 101, 1), 'Мощность лампы')

# Определение функций принадлежности (трапециевидные)

# Функции принадлежности для Освещенности
illumination['темно'] = fuzz.trapmf(illumination.universe, [0, 0, 50, 150])
illumination['приглушенно'] = fuzz.trapmf(illumination.universe, [100, 200, 300, 400])
illumination['ярко'] = fuzz.trapmf(illumination.universe, [350, 500, 700, 800])
illumination['слишком ярко'] = fuzz.trapmf(illumination.universe, [750, 850, 1000, 1000])

# Функции принадлежности для Времени суток
time_of_day['ночь'] = fuzz.trapmf(time_of_day.universe, [0, 0, 4, 6])
time_of_day['утро'] = fuzz.trapmf(time_of_day.universe, [5, 7, 9, 11])
time_of_day['день'] = fuzz.trapmf(time_of_day.universe, [10, 12, 18, 20])
time_of_day['вечер'] = fuzz.trapmf(time_of_day.universe, [19, 21, 23, 24])

# Функции принадлежности для Мощности лампы
lamp_power['выключить'] = fuzz.trapmf(lamp_power.universe, [0, 0, 10, 25])
lamp_power['слабая'] = fuzz.trapmf(lamp_power.universe, [15, 30, 50, 65])
lamp_power['сильная'] = fuzz.trapmf(lamp_power.universe, [60, 80, 100, 100])

# Визуализация функций принадлежности для проверки
illumination.view()
time_of_day.view()
lamp_power.view()

# Определение нечетких правил с операцией ОБЪЕДИНЕНИЯ (ИЛИ)

# Правило 1: Включить сильный свет, ЕСЛИ ночь ИЛИ в помещении темно.
rule1 = ctrl.Rule(time_of_day['ночь'] | illumination['темно'], lamp_power['сильная'])

# Правило 2: Включить слабый свет, ЕСЛИ утро ИЛИ вечер ИЛИ в помещении приглушенно.
rule2 = ctrl.Rule(time_of_day['утро'] | time_of_day['вечер'] | illumination['приглушенно'], lamp_power['слабая'])

# Правило 3: Выключить свет, ЕСЛИ день ИЛИ в помещении ярко ИЛИ слишком ярко.
rule3 = ctrl.Rule(time_of_day['день'] | illumination['ярко'] | illumination['слишком ярко'], lamp_power['выключить'])

# Создание системы управления

light_control_system = ctrl.ControlSystem([rule1, rule2, rule3])
lighting_simulation = ctrl.ControlSystemSimulation(light_control_system)

# Задаем входные четкие значения для симуляции
input_time = 20
input_illumination = 250

# Передаем значения в систему управления
lighting_simulation.input['Время суток'] = input_time
lighting_simulation.input['Освещенность'] = input_illumination

lighting_simulation.compute()

print(f"Время суток: {input_time}:00")
print(f"Освещенность: {input_illumination} люкс")
print(f"Рекомендуемая мощность лампы: {lighting_simulation.output['Мощность лампы']:.2f}%")

# Визуализация результата
lamp_power.view(sim=lighting_simulation)

plt.show()
