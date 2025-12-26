import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Пытаемся взять адрес базы из переменных окружения (в Docker он будет задан)
# Если переменной нет (например, локальный запуск), используем файл SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

# Настройка движка базы данных
if DATABASE_URL.startswith("sqlite"):
    # Для SQLite нужны особые настройки потоков
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # Для Postgres просто создаем движок
    engine = create_engine(DATABASE_URL)

# Создаем фабрику сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для моделей
Base = declarative_base()

# Функция-генератор для получения сессии
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
