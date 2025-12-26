import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from catboost import CatBoostRegressor
from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from sqlalchemy.orm import Session

# Импортируем подключение к базе
from .database import Base, engine, get_db


# 1. Описание таблицы БД (Model)
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    distance_km = Column(Float)
    traffic_level = Column(Integer)
    courier_type = Column(String)
    weather = Column(String, default="clear")
    hour_of_day = Column(Integer, default=12)

    predicted_eta = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# создаем таблицы в базе (если их нет)
Base.metadata.create_all(bind=engine)

# 2. Инициализация приложения
app = FastAPI(title="Delivery ETA Service")
model = CatBoostRegressor()

# 3. Загрузка ML модели
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "delivery_model.cbm")


@app.on_event("startup") # выполняется только один раз
def load_model():
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
        print(f"УСПЕХ: Модель загружена из {MODEL_PATH}")
    else:
        print(f"ОШИБКА: Файл модели не найден: {MODEL_PATH}")


# 4. Pydantic схемы (валидация данных)
class DeliveryRequest(BaseModel):
    distance_km: float
    traffic_level: int
    courier_type: str
    weather: str = "clear"
    hour_of_day: int = 12


class OrderResponse(BaseModel):
    id: int
    distance_km: float
    predicted_eta: float

    # позволяет Pydantic читать данные из ORM-объектов
    class Config:
        orm_mode = True


# 5. эндпоинты API

@app.get("/")
def home():
    return {"message": "ETA Service is running", "docs": "/docs"}


@app.post("/predict_eta", response_model=OrderResponse)
def predict_eta(request: DeliveryRequest, db: Session = Depends(get_db)):
    # принимает параметры (включая погоду и время), считает ETA и сохраняет заказ в базу.
    try:
        # 1. подготовка признаков для модели

        features = [
            request.distance_km,
            request.traffic_level,
            request.weather,
            request.courier_type,
            request.hour_of_day
        ]

        prediction = model.predict(features)
        eta_value = float(round(prediction, 1))

        # 2. сохраняем в БД
        new_order = Order(
            distance_km=request.distance_km,
            traffic_level=request.traffic_level,
            courier_type=request.courier_type,
            weather=request.weather,
            hour_of_day=request.hour_of_day,
            predicted_eta=eta_value
        )
        db.add(new_order)
        db.commit()
        db.refresh(new_order)  # получаем ID новой записи

        return new_order

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")


@app.get("/orders")
def get_orders(db: Session = Depends(get_db)):
    # показать историю всех заказов
    return db.query(Order).all()


# 6. запуск
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
