import pandas as pd
import re
from thefuzz import process, fuzz
import time

FOOD_FILE = 'foods.csv'
PRICE_FILE = 'prices.csv'
OUTPUT_FILE = 'products_with_prices.csv'  # Новое имя для отфильтрованного файла

# Пороги для каскадного поиска
HIGH_CONFIDENCE_CUTOFF = 90
KEYWORD_CUTOFF = 85
LAST_RESORT_CUTOFF = 75

print("--- Начало процесса: Поиск цен для продуктов из файла КБЖУ ---")

try:
    print(f"Загрузка данных из '{FOOD_FILE}' и '{PRICE_FILE}'...")
    food_df = pd.read_csv(FOOD_FILE)
    price_df = pd.read_csv(PRICE_FILE)
    print("Данные успешно загружены.")
except FileNotFoundError as e:
    print(f"Ошибка: Файл не найден. Убедитесь, что '{FOOD_FILE}' и '{PRICE_FILE}' находятся в той же папке.")
    exit()

# Подготовка и очистка данных

print("Переименование и очистка колонок...")
food_df.rename(columns={
    'Продукт': 'food_name',
    'Жиры, г': 'fat',
    'Белки, г': 'protein',
    'Углеводы, г': 'carbs',
    'Калорийность, Ккал': 'calories'
}, inplace=True)

price_df.rename(columns={
    'Название продукта': 'price_name',
    'Цена': 'price_per_kg'
}, inplace=True)


def preprocess_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\(.*\)', '', text)
    text = re.sub(r'[,.()]', '', text)
    text = re.sub(r'\b(кг|г)\b', '', text)
    text = ' '.join(text.split())
    text = text.strip()
    return text


food_df['clean_name'] = food_df['food_name'].apply(preprocess_text)
price_df['clean_name'] = price_df['price_name'].apply(preprocess_text)

# Инвертированное многоэтапное сопоставление

print("Выполнение многоэтапного сопоставления...")
start_time = time.time()

price_choices = price_df['clean_name'].tolist()
match_results = []

manual_map = {}

for index, row in food_df.iterrows():
    name_to_match = row['clean_name']
    best_match_name = None

    if name_to_match in manual_map:
        best_match_name = manual_map[name_to_match]
    else:
        match = process.extractOne(name_to_match, price_choices, scorer=fuzz.WRatio,
                                   score_cutoff=HIGH_CONFIDENCE_CUTOFF)
        if match:
            best_match_name = match[0]
        else:
            keywords = name_to_match.split()
            best_score = 0
            for keyword in keywords:
                if len(keyword) > 3:
                    keyword_choices = [s for s in price_choices if keyword in s]
                    if 0 < len(keyword_choices) < 50:
                        match = process.extractOne(name_to_match, keyword_choices, scorer=fuzz.token_set_ratio,
                                                   score_cutoff=KEYWORD_CUTOFF)
                        if match and match[1] > best_score:
                            best_match_name = match[0]
                            best_score = match[1]
            if not best_match_name:
                match = process.extractOne(name_to_match, price_choices, scorer=fuzz.token_set_ratio,
                                           score_cutoff=LAST_RESORT_CUTOFF)
                if match:
                    best_match_name = match[0]

    match_results.append({'food_name': row['food_name'], 'matched_price_clean_name': best_match_name})

end_time = time.time()
print(f"Сопоставление завершено за {end_time - start_time:.2f} секунд.")

# Объединение данных и формирование финального файла

print("Объединение результатов...")
results_df = pd.DataFrame(match_results)
final_df = pd.merge(food_df, results_df, on='food_name', how='left')

print("Добавление цен и названий соответствий...")
price_dict = price_df.set_index('clean_name')['price_per_kg'].to_dict()
original_name_dict = price_df.set_index('clean_name')['price_name'].to_dict()

final_df['Цена за 100гр'] = (final_df['matched_price_clean_name'].map(price_dict) / 10).round(2)
final_df['Соответствие'] = final_df['matched_price_clean_name'].map(original_name_dict)

# Финальная очистка и формирование

print("Удаление продуктов, для которых не нашлась цена...")
rows_before_dropping = len(final_df)
# Используем .dropna() для удаления строк, где в колонке 'Цена за 100гр' стоит NaN
final_df.dropna(subset=['Цена за 100гр'], inplace=True)
rows_after_dropping = len(final_df)
print(
    f"Было строк: {rows_before_dropping}. Осталось строк: {rows_after_dropping}. Удалено: {rows_before_dropping - rows_after_dropping}.")

print("Формирование итогового датасета...")
final_df = final_df[[
    'food_name',
    'protein',
    'fat',
    'carbs',
    'calories',
    'Цена за 100гр',
    'Соответствие'
]].rename(columns={
    'food_name': 'Продукт',
    'protein': 'Белки, г',
    'fat': 'Жиры, г',
    'carbs': 'Углеводы, г',
    'calories': 'Калорийность, Ккал'
})

# Сбрасываем индекс, чтобы он шел по порядку после удаления строк
final_df.reset_index(drop=True, inplace=True)

# Вывод статистики и сохранение результата
print("\n--- Итоговая статистика ---")
total_items_in_food_file = len(food_df)
final_items = len(final_df)
match_percentage = (final_items / total_items_in_food_file) * 100
print(f"Всего продуктов в файле КБЖУ: {total_items_in_food_file}")
print(f"Найдена цена для: {final_items} продуктов ({match_percentage:.2f}%)")

print("\nПримеры продуктов из финального файла:")
print(final_df.head(20).to_string())

try:
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nИтоговый датасет сохранен в файл: '{OUTPUT_FILE}'")
    print(f"Он содержит {len(final_df)} строк только с найденными ценами.")
except Exception as e:
    print(f"\nОшибка при сохранении файла: {e}")

print("\n--- Процесс завершен ---")