import joblib
import os

new_model_info = joblib.load('app/model_info.pkl')
new_test_accuracy = new_model_info['test_accuracy']
print(f'Nuevo modelo - Precisión en prueba: {new_test_accuracy:.4f}')

if os.path.exists('previous_model/model.pkl'):
    if os.path.exists('previous_model/model_info.pkl'):
        prev_model_info = joblib.load('previous_model/model_info.pkl')
        prev_test_accuracy = prev_model_info['test_accuracy']
    else:
        prev_test_accuracy = 0  # placeholder si no hay info
    improvement = new_test_accuracy - prev_test_accuracy
    print(f'Mejora: {improvement:.4f}')
    if improvement < 0:
        print('El nuevo modelo es peor. Abortando.')
        exit(1)
    elif improvement < 0.005:
        print('La mejora no es significativa.')
else:
    print('No se encontró modelo anterior.')

if new_test_accuracy < 0.85:
    print(f'Precisión insuficiente: {new_test_accuracy:.4f}')
    exit(1)
else:
    print('Precisión aceptable.')
