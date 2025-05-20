import os
import shutil
import joblib
import subprocess

# Rutas
model_pkl = 'api/app/model.pkl'
model_info = 'api/app/model_info.pkl'
previous_model_dir = 'api/previous_model'
prev_model_pkl = os.path.join(previous_model_dir, 'model.pkl')
prev_model_info = os.path.join(previous_model_dir, 'model_info.pkl')

# === 1. Renombrar modelo actual como "anterior" ===
print("📦 Guardando modelo anterior...")

os.makedirs(previous_model_dir, exist_ok=True)

if os.path.exists(model_pkl):
    shutil.copy(model_pkl, prev_model_pkl)
    print(f'✅ model.pkl copiado a {prev_model_pkl}')
else:
    print(f'⚠ No se encontró {model_pkl} para mover.')

if os.path.exists(model_info):
    shutil.copy(model_info, prev_model_info)
    print(f'✅ model_info.pkl copiado a {prev_model_info}')
else:
    print(f'⚠ No se encontró {model_info} para mover.')

# === 2. Entrenar nuevo modelo ===
print("⚙️ Entrenando nuevo modelo...")
subprocess.run(["python", "api/train_model.py"], check=True)

# === 3. Cargar precisión del nuevo modelo ===
print("📥 Cargando info del nuevo modelo...")
if not os.path.exists(model_info):
    print("❌ No se encontró model_info.pkl después del entrenamiento.")
    exit(1)

new_model_info = joblib.load(model_info)
new_test_accuracy = new_model_info['test_accuracy']
print(f'🎯 Precisión del nuevo modelo: {new_test_accuracy:.4f}')

# === 4. Comparar con modelo anterior (si existe) ===
if os.path.exists(prev_model_pkl):
    if os.path.exists(prev_model_info):
        prev_info = joblib.load(prev_model_info)
        prev_test_accuracy = prev_info.get('test_accuracy', 0)
    else:
        prev_test_accuracy = 0

    improvement = new_test_accuracy - prev_test_accuracy
    print(f'📊 Mejora: {improvement:.4f}')

    if improvement < 0:
        print('❌ El nuevo modelo es peor. Abortando.')
        exit(1)
    elif improvement < 0.005:
        print('⚠ Mejora no significativa (< 0.005)')
    else:
        print('✅ Mejora aceptable.')
else:
    print("ℹ No se encontró modelo anterior. Continuando...")

# === 5. Verificar umbral mínimo de precisión ===
if new_test_accuracy < 0.85:
    print(f'❌ Precisión insuficiente: {new_test_accuracy:.4f} < 0.85')
    exit(1)
else:
    print(f'✅ Precisión aceptable: {new_test_accuracy:.4f}')
