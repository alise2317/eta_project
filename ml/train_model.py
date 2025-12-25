import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os

# --- 1. Генерация данных (Имитация истории) ---
print("Генерируем данные...")
data_size = 2000

data = {
    # Случайное расстояние от 0.5 до 25 км
    'distance_km': np.random.uniform(0.5, 25.0, data_size),

    # Пробки от 1 до 10 баллов
    'traffic_level': np.random.randint(1, 11, data_size),

    # Тип курьера
    'courier_type': np.random.choice(['foot', 'bike', 'car'], data_size)
}

df = pd.DataFrame(data)


# Функция для расчета "идеального" времени (Target), чтобы модель могла найти закономерность
def calculate_target_time(row):
    # Базовая скорость (мин на км)
    speed_map = {'car': 2.0, 'bike': 4.0, 'foot': 10.0}

    dist = row['distance_km']
    traffic = row['traffic_level']
    ctype = row['courier_type']

    # Время в пути = (дистанция * мин/км)
    travel_time = dist * speed_map[ctype]

    # Влияние пробок (только для авто сильное, для вело слабое, пешим почти нет)
    traffic_penalty = 0
    if ctype == 'car':
        traffic_penalty = traffic * 3.0  # +3 мин за каждый балл пробки
    elif ctype == 'bike':
        traffic_penalty = traffic * 0.5

    # Базовое время на "забрать/отдать" заказ (5-10 мин)
    service_time = 10

    # Итоговое время + небольшой случайный шум (реальность)
    total_time = travel_time + traffic_penalty + service_time + np.random.normal(0, 3)

    return max(5.0, total_time)  # Не может быть меньше 5 минут


# Создаем колонку с ответами (то, что модель будет предсказывать)
df['delivery_time_minutes'] = df.apply(calculate_target_time, axis=1)

print(f"Датасет готов: {data_size} строк.")
print(df.head(3))

# --- 2. Обучение модели ---
print("\nНачинаем обучение CatBoost...")

X = df[['distance_km', 'traffic_level', 'courier_type']]
y = df['delivery_time_minutes']

# Указываем CatBoost, что 'courier_type' - это текст (категория)
cat_features = ['courier_type']

model = CatBoostRegressor(
    iterations=500,  # Количество деревьев
    learning_rate=0.1,  # Скорость обучения
    depth=6,  # Глубина дерева
    loss_function='RMSE',  # Метрика ошибки
    verbose=100  # Выводить отчет каждые 100 итераций
)

model.fit(X, y, cat_features=cat_features)

print("\nОбучение завершено.")

# --- 3. Сохранение модели ---
# Определяем путь: сохраним в ту же папку, где лежит скрипт, или в корень
# Давай сохраним прямо в корень проекта, чтобы API легко её нашел
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Если скрипт в eta_project/ml, то project_root будет eta_project

model_path = os.path.join(project_root, "delivery_model.cbm")
model.save_model(model_path)

print(f"✅ Модель сохранена в файл: {model_path}")

# --- 4. Тестовый прогноз ---
print("\nПроверка модели:")
test_case = [5.5, 8, 'car']  # 5.5 км, 8 баллов, авто
pred = model.predict(test_case)
print(f"Вход: {test_case} -> Прогноз: {pred:.2f} мин.")
