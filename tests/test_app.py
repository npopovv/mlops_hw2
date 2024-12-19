import pytest
from unittest.mock import patch, MagicMock
from app.models import ModelManager, ModelType
from minio import Minio
import joblib
import os
import numpy as np


@pytest.fixture
def mock_minio_client():
    """Фикстура для мока Minio client."""
    mock_minio = MagicMock(spec=Minio)

    # определяем поведение
    mock_minio.bucket_exists.return_value = True  # типо бакет создан
    mock_minio.fput_object.return_value = None  # успешная загрузка
    mock_minio.fget_object.return_value = None  # успешный get
    mock_minio.remove_object.return_value = None  # успешное удаление

    return mock_minio


@pytest.fixture
def model_manager(mock_minio_client):
    """Фикстура для ModelManager с замоканным Minio client."""
    # Патчим создание клиента Minio в конструкторе ModelManager
    with patch("app.models.Minio", return_value=mock_minio_client):
        manager = ModelManager(
            storage_dir=".", bucket_name="models", dvc_bucket="datasets"
        )

        return manager


def test_download_model(model_manager, mock_minio_client):
    """Тестирует загрузку моделей из Minio"""
    model_id = 1
    file_name = f"model_{model_id}.pkl"
    local_path = f"./{file_name}"

    # Вызываем метод download_from_minio
    result = model_manager.download_from_minio(model_id)

    # Проверяем, что результат возвращает правильный путь
    assert result == local_path

    # Проверяем, что метод fget_object был вызван с правильными аргументами
    mock_minio_client.fget_object.assert_called_with("models", file_name, local_path)


def test_train_method(model_manager):
    """Тест обучения модели."""
    model_type = ModelType.LOGISTIC
    params = {"C": 1.0}
    X_train = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    y_train = [0, 1]

    with patch.object(model_manager, "_get_next_model_id", return_value=1):
        with patch.object(joblib, "dump") as mock_dump:
            with patch.object(model_manager, "upload_to_minio") as mock_upload:
                model_id = model_manager.train(model_type, params, X_train, y_train)

                # Проверяем ID модели
                assert model_id == 1

                # Проверяем, что модель сохранена с правильным ID
                mock_dump.assert_called_once()
                saved_path = mock_dump.call_args[0][1]
                assert saved_path == "./model_1.pkl"

                # Проверяем, что модель загружена в Minio
                mock_upload.assert_called_once_with(saved_path)


def test_download_model(model_manager, mock_minio_client):
    """Тест загрузки из Minio"""
    model_id = 1
    file_name = f"model_{model_id}.pkl"
    local_path = f"./{file_name}"

    # Вызываем метод download_from_minio
    result = model_manager.download_from_minio(model_id)

    # Проверяем, что результат возвращает правильный путь
    assert result == local_path

    # Проверяем, что метод fget_object был вызван с правильными аргументами
    mock_minio_client.fget_object.assert_called_with("models", file_name, local_path)


def test_predict(model_manager):
    """Проверка теста"""
    model_id = 1
    data = [1.0, 2.0, 3.0]

    # Мокаем загрузку модели
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    with patch.object(joblib, "load", return_value=mock_model):
        result = model_manager.predict(model_id, data)

        # Проверяем вызов метода predict
        mock_model.predict.assert_called_once_with([data])
        assert result == [0]


def test_delete_model(model_manager, mock_minio_client):
    """Тестирование удаления моделей"""
    model_id = 1
    local_path = f"./model_{model_id}.pkl"

    # Создаем временный файл для теста
    open(local_path, "w").close()

    # Проверяем удаление локального файла
    model_manager.delete(model_id)
    assert not os.path.exists(local_path)

    # Проверяем вызов метода remove_object для Minio
    mock_minio_client.remove_object.assert_called_with(
        "models", f"model_{model_id}.pkl"
    )


def test_save_dataset_with_dvc(model_manager):
    """Тест сохранения DVC"""
    dataset_path = "test_dataset.csv"

    # Создаем временный файл для теста
    open(dataset_path, "w").close()

    with patch("subprocess.run") as mock_subprocess:
        model_manager.save_dataset_with_dvc(dataset_path)

        # Проверяем вызовы DVC
        mock_subprocess.assert_any_call(["dvc", "add", dataset_path], check=True)
        mock_subprocess.assert_any_call(
            ["git", "add", f"{dataset_path}.dvc"], check=True
        )
        mock_subprocess.assert_any_call(
            ["git", "commit", "-m", f"Added dataset: {dataset_path}"], check=True
        )
        mock_subprocess.assert_any_call(["dvc", "push"], check=True)

    # Удаляем временный файл
    os.remove(dataset_path)


def test_load_dataset_with_dvc(model_manager):
    """Тест загрузки в DVC."""
    dataset_name = "test_dataset.csv"

    with patch("subprocess.run") as mock_subprocess:
        model_manager.load_dataset_with_dvc(dataset_name)

        # Проверяем вызов DVC pull
        mock_subprocess.assert_called_once_with(
            ["dvc", "pull", dataset_name], check=True
        )
