from enum import Enum
import os
import joblib
from app.logging_config import setup_logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from minio import Minio
from minio.error import S3Error
import subprocess
# import shutil
# import requests

# Настроенный логгер
logger = setup_logger(name="model_manager")


class ModelType(str, Enum):
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"


class ModelManager:
    def __init__(self, storage_dir="models_storage", bucket_name="models", dvc_bucket = "datasets"):
        """Инициализация менеджера моделей с поддержкой Minio."""
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.s3_bucket = bucket_name
        self.dvc_bucket = dvc_bucket
        self.minio_client = Minio(
            "minio:9000",
            access_key='access_key',
            secret_key='secret_key',
            secure=False,
        )

        # Проверяем, существует ли бакет для моделей и создаем если нет
        if not self.minio_client.bucket_exists(self.s3_bucket):
            self.minio_client.make_bucket(self.s3_bucket)
            logger.info(f"Bucket {self.s3_bucket} created")

        # Проверяем, существует ли бакет для dvc
        if not self.minio_client.bucket_exists(self.dvc_bucket):
            self.minio_client.make_bucket(self.dvc_bucket)
            logger.info(f"Bucket {self.dvc_bucket} created")

    def upload_to_minio(self, file_path):
        """Загрузка файла модели в Minio."""
        file_name = os.path.basename(file_path)
        try:
            self.minio_client.fput_object(self.s3_bucket, file_name, file_path)
            logger.info(f"File {file_name} uploaded to Minio bucket {self.s3_bucket}")
        except S3Error as e:
            logger.error(f"Failed to upload file to Minio: {e}")
            raise e

    def download_from_minio(self, model_id):
        """Загрузка модели из Minio."""
        file_name = f"model_{model_id}.pkl"
        local_path = os.path.join(self.storage_dir, file_name)
        try:
            self.minio_client.fget_object(self.s3_bucket, file_name, local_path)
            logger.info(f"File {file_name} downloaded from Minio bucket {self.s3_bucket}")
        except S3Error as e:
            logger.error(f"Failed to download file from Minio: {e}")
            raise e
        return local_path

    def delete_from_minio(self, model_id):
        """Удаление модели из Minio."""
        file_name = f"model_{model_id}.pkl"
        try:
            self.minio_client.remove_object(self.s3_bucket, file_name)
            logger.info(f"File {file_name} deleted from Minio bucket {self.s3_bucket}")
        except S3Error as e:
            logger.error(f"Failed to delete file from Minio: {e}")
            raise e

    def train(self, model_type: ModelType, params: dict, X_train, y_train):
        """Обучение модели."""
        logger.info(f"Starting training for model type: {model_type}")

        if model_type == ModelType.LOGISTIC:
            model = LogisticRegression(**params)
        elif model_type == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(**params)
        else:
            logger.error("Unknown model type")
            raise ValueError("Unknown model type")

        model.fit(X_train, y_train)
        model_id = self._get_next_model_id()

        # Сохранение модели
        path_to_save = os.path.join(self.storage_dir, f"model_{model_id}.pkl")
        joblib.dump(model, path_to_save)
        self.upload_to_minio(path_to_save)  # Загружаем модель в Minio

        logger.info(f"Model trained and saved with ID: {model_id}")
        return model_id

    def get_available_models(self):
        """Получение списка доступных моделей."""
        logger.info("Fetching list of available models")

        model_files = [f for f in os.listdir(self.storage_dir) if f.endswith(".pkl")]
        models = [
            {"id": int(f.split("_")[1].split(".")[0]), "path": f} for f in model_files
        ]
        return sorted(models, key=lambda x: x["id"])

    def predict(self, model_id: int, data):
        """Выполнение предсказания с использованием модели."""
        logger.info(f"Received data for prediction with model ID: {model_id}")
        model_path = os.path.join(self.storage_dir, f"model_{model_id}.pkl")
        if not os.path.exists(model_path):
            logger.info("Model not found in local storage. Trying to download from Minio.")
            model_path = self.download_from_minio(model_id)  # Загружаем из Minio

        model = joblib.load(model_path)
        prediction = model.predict([data]).tolist()
        logger.info(f"Prediction result: {prediction}")
        return prediction

    def delete(self, model_id: int):
        """Удаление модели локально и в Minio."""
        logger.info(f"Attempting to delete model with ID: {model_id}")
        model_path = os.path.join(self.storage_dir, f"model_{model_id}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Model {model_id} deleted locally")

        try:
            self.delete_from_minio(model_id)  # Удаляем модель из Minio
        except Exception as e:
            logger.warning(f"Failed to delete model {model_id} from Minio: {e}")

        return True

    def _get_next_model_id(self):
        """Получение следующего доступного ID модели."""
        existing_models = self.get_available_models()
        if not existing_models:
            return 1
        return max(model["id"] for model in existing_models) + 1
    
    def save_dataset_with_dvc(self, dataset_path: str):
        """Сохранение датасета с версионированием через DVC."""

        #проверка что получается подключиться к endpointurl
        #response = requests.head('http://minio:9000', timeout=5)
        #print(response, "http://minio:9000 доступно")

        try:
            # Добавляем файл в DVC
            subprocess.run(["dvc", "add", dataset_path], check=True)
            
            # Коммит изменений в DVC
            subprocess.run(["git", "add", dataset_path + ".dvc"], check=True)
            subprocess.run(["git", "commit", "-m", f"Added dataset: {dataset_path}"], check=True)
            
            # Отправляем данные в удалённое хранилище
            subprocess.run(["dvc", "push"], check=True)
            logger.info(f"Dataset {dataset_path} saved to DVC.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при работе с DVC: {e}")
            raise RuntimeError("DVC operation failed.")

    def load_dataset_with_dvc(self, dataset_name: str):
        """Загрузка датасета из DVC."""
        try:
            # Загружаем файл из DVC
            subprocess.run(["dvc", "pull", dataset_name], check=True)
            local_path = dataset_name  # Локальный путь после pull
            logger.info(f"Dataset {dataset_name} успешно загружен через DVC.")
            return local_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при загрузке через DVC: {e}")
            raise RuntimeError("DVC operation failed.")