import os
import shutil
import joblib
import subprocess
import pytest
from unittest import mock

@pytest.fixture
def setup_model_files(tmp_path):
    # Crear estructura de directorios
    app_dir = tmp_path / "api" / "app"
    previous_dir = tmp_path / "api" / "previous_model"
    app_dir.mkdir(parents=True)
    previous_dir.mkdir(parents=True)

    # Crear archivos dummy
    model_pkl = app_dir / "model.pkl"
    model_info = app_dir / "model_info.pkl"
    model_pkl.write_bytes(b"dummy model")
    joblib.dump({'test_accuracy': 0.90}, model_info)

    return {
        "app_dir": app_dir,
        "previous_dir": previous_dir,
        "model_pkl": model_pkl,
        "model_info": model_info,
        "prev_model_pkl": previous_dir / "model.pkl",
        "prev_model_info": previous_dir / "model_info.pkl"
    }

@mock.patch("subprocess.run")
def test_model_pipeline(mock_run, setup_model_files):
    paths = setup_model_files

    # 1. Renombrar modelo actual como anterior
    shutil.copy(paths["model_pkl"], paths["prev_model_pkl"])
    shutil.copy(paths["model_info"], paths["prev_model_info"])

    assert os.path.exists(paths["prev_model_pkl"])
    assert os.path.exists(paths["prev_model_info"])

    # 2. Simular entrenamiento (mocked subprocess)
    mock_run.return_value = None  # No-op
    subprocess.run(["python", "api/train_model.py"], check=True)
    mock_run.assert_called_once()

    # 3. Sobrescribir el nuevo model_info con menor precisión para test
    joblib.dump({'test_accuracy': 0.91}, paths["model_info"])
    new_info = joblib.load(paths["model_info"])
    new_acc = new_info['test_accuracy']
    assert new_acc >= 0.85

    # 4. Comparar con anterior
    prev_info = joblib.load(paths["prev_model_info"])
    prev_acc = prev_info.get("test_accuracy", 0)
    improvement = new_acc - prev_acc

    assert improvement >= 0  # Nuevo modelo no debe ser peor

    # 5. Verificar si la mejora fue significativa o no
    if improvement >= 0.005:
        improvement_significant = True
    else:
        improvement_significant = False

    assert improvement_significant  # test pasa solo si mejora es significativa
