from fastapi import FastAPI, HTTPException
from app.models import ModelManager, ModelType
from app.logging_config import setup_logger
from pydantic import BaseModel, Field
from typing import List
import psutil
from datetime import datetime

# настроенный логгер
logger = setup_logger(name=__name__)

app = FastAPI()
model_manager = ModelManager()


# валидация входных данных
class TrainRequest(BaseModel):
    model_type: ModelType
    params: dict = Field(default_factory=dict, description="model hyperparams")
    X_train: List[List[float]] = Field(
        default=[
            [10, 12, 13, 12, 15],
            [0.1, 0.2, 0.3, 0.6, 0.2],
            [11, 13, 13, 15, 20],
            [0.2, 0.6, 0.1, 0.22, 0.3],
        ],
        description="features",
    )

    y_train: List[int] = Field(default=[1, 0, 1, 0], description="target")


@app.post("/train/")
async def train_model(request: TrainRequest):
    logger.info(
        f"API call to train model of type {request.model_type} with params: {request.params}"
    )

    try:
        model_id = model_manager.train(
            model_type=request.model_type,
            params=request.params,
            X_train=request.X_train,
            y_train=request.y_train,
        )
        logger.info(f"Model trained with ID: {model_id}")
        return {"model_id": model_id}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/")
async def get_models():
    logger.info("API call to get list of models")
    return model_manager.get_available_models()


@app.post("/predict/")
async def predict(model_id: int, data: str):

    data = list(map(float, data.split(" ")))  # меняем на норм формат для модели
    logger.info(f"API call to predict with model ID: {model_id} and data: {data}")

    try:
        prediction = model_manager.predict(model_id, data)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete/")
async def delete_model(model_id: int):
    logger.info(f"API call to delete model with ID: {model_id}")

    try:
        model_manager.delete(model_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/")
async def status():
    logger.info("API call to get status")

    # Получение системных метрик
    memory_info = psutil.virtual_memory()  # Информация об оперативной памяти
    uptime = datetime.now() - datetime.fromtimestamp(
        psutil.boot_time()
    )  # Время работы системы

    # метрика
    custom_metric = (
        "Все сервисы работают штатно"
        if memory_info.percent < 80
        else "Высокая нагрузка"
    )

    # Возврат данных
    return {
        "status": "running",
        "memory_usage": {
            "total": memory_info.total,
            "available": memory_info.available,
            "used": memory_info.used,
            "percent": memory_info.percent,
        },
        "uptime": str(uptime),  # Формат времени работы
        "business_logic_metric": custom_metric,
    }
