import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# 1. формула харвесина для расчёта расстояния с учётом кривизны Земли
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # радиус Земли в км
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# 2. загрузка и очистка данных
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень проекта eta_project/
csv_path = os.path.join(base_dir, "train.csv")

print(f"загружаем данные из {csv_path}...")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("ОШИБКА: Файл train.csv не найден!")
    exit(1)

# убираем пробелы в названиях колонок (если есть)
df.columns = df.columns.str.strip()

print(f"Исходный размер: {df.shape}")

# 3. Feature Engineering (создание признаков)

# расчет дистанции
df['Restaurant_latitude'] = df['Restaurant_latitude'].abs()
df['Restaurant_longitude'] = df['Restaurant_longitude'].abs()
df['Delivery_location_latitude'] = df['Delivery_location_latitude'].abs()
df['Delivery_location_longitude'] = df['Delivery_location_longitude'].abs()

df['distance_km'] = haversine_distance(
    df['Restaurant_latitude'], df['Restaurant_longitude'],
    df['Delivery_location_latitude'], df['Delivery_location_longitude']
)

# Б. очистка целевой переменной (остаётся только число)
df['delivery_time'] = df['Time_taken(min)'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['delivery_time'] = pd.to_numeric(df['delivery_time'], errors='coerce')

# В. очистка категориальных полей
df['weather'] = df['Weatherconditions'].astype(str).str.replace('conditions ', '', case=False).str.strip()
df['weather'] = df['weather'].replace('NaN', 'Sunny')

df['traffic_level'] = df['Road_traffic_density'].astype(str).str.strip()
df['traffic_level'] = df['traffic_level'].replace('NaN', 'Low')
traffic_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}
df['traffic_level_int'] = df['traffic_level'].map(traffic_map).fillna(2).astype(int)

df['courier_type'] = df['Type_of_vehicle'].astype(str).str.strip()

# Г. извлекаем час
def extract_hour(time_str):
    try:
        if pd.isna(time_str): return 12
        parts = str(time_str).split(':')
        if len(parts) >= 1:
            return int(parts[0])
        return 12
    except:
        return 12

df['hour_of_day'] = df['Time_Orderd'].apply(extract_hour)
df['hour_of_day'] = df['hour_of_day'].clip(0, 23)

# 4. фильтрация мусора
df = df[(df['distance_km'] > 0.1) & (df['distance_km'] < 100)]
df = df.dropna(subset=['delivery_time'])

print(f"Размер после очистки: {df.shape}")

# 5. подготовка к обучению
features = ['distance_km', 'traffic_level_int', 'weather', 'courier_type', 'hour_of_day']
target = 'delivery_time'

X = df[features]
y = df[target]

print("\nПример данных для обучения:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. обучение
print("\nОбучение CatBoost на реальных данных...")

# текстовые признаки
cat_features = ['weather', 'courier_type']

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=100
)

model.fit(X_train, y_train, cat_features=cat_features)

# 7. оценка
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\nРезультаты:")
print(f"MAE: {mae:.2f} мин")
print(f"R2: {r2:.3f}")

# 8. сохранение
model_path = os.path.join(base_dir, "delivery_model.cbm")
model.save_model(model_path)
print(f"Модель сохранена в {model_path}")
