import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor

app = FastAPI(title="Delivery ETA Service")
model = CatBoostRegressor()

# === ПРАВИЛЬНЫЙ ПУТЬ ===
# Ищем файл delivery_model.cbm в папке выше, чем лежит этот скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # папка app
ROOT_DIR = os.path.dirname(BASE_DIR)                  # папка eta_project
MODEL_PATH = os.path.join(ROOT_DIR, "delivery_model.cbm")
# ========================

class DeliveryRequest(BaseModel):
    distance_km: float
    traffic_level: int
    courier_type: str

# Используем правильный event lifespan (вместо устаревшего startup)
# Но пока для простоты и совместимости оставим так, это работает:
@app.on_event("startup")
def load_model():
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
        print(f"✅ УСПЕХ: Модель загружена из {MODEL_PATH}")
    else:
        print(f"❌ ОШИБКА: Файл не найден по пути: {MODEL_PATH}")
        # Выведем список файлов в корне, чтобы понять, что там есть
        print(f"Файлы в папке {ROOT_DIR}: {os.listdir(ROOT_DIR)}")


@app.get("/")
def home():
    return {"message": "ETA Service is running", "docs": "/docs"}


@app.post("/predict_eta")
def predict_eta(request: DeliveryRequest):
    """
    Главный метод: принимает данные заказа -> отдает прогноз времени
    """
    try:
        # 1. Подготовка фичей в том же порядке, как при обучении!
        features = [
            request.distance_km,
            request.traffic_level,
            request.courier_type
        ]

        # 2. Предсказание
        prediction = model.predict(features)

        # 3. Ответ
        return {
            "eta_minutes": round(prediction, 1),
            "details": {
                "distance": request.distance_km,
                "courier": request.courier_type
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Блок для запуска прямо из PyCharm (кнопкой Run)
if __name__ == "__main__":
    # reload=True позволяет менять код и сразу видеть изменения без перезапуска
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
