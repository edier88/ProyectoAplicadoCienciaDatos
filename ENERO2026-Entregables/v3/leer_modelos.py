import os
import glob
import joblib
from pathlib import Path

#MODELOS_DIR = Path("modelos_guardados")
MODELOS_DIR = "modelos_guardados/"

archivos = glob.glob(os.path.join(MODELOS_DIR, "*.joblib"))

for archivo in archivos:
    nombre_zona = os.path.basename(archivo)
    modelo_completo = joblib.load(open(archivo, 'rb'))
    model_type = modelo_completo['metadata']['model_type']

    print(f"{nombre_zona}:")

    if model_type == "Random Forest":
        """
        kernel = modelo_completo['metadata']['kernel']
        n_estimators = modelo_completo['metadata']['n_estimators_optimizado']
        max_depth = modelo_completo['metadata']['max_depth_optimizado']
        lags = modelo_completo['metadata']['lags']

        print(f"{model_type}:")
        print(f"['kernel': {kernel}, 'n_estimators': {n_estimators}, 'max_depth': {max_depth}, 'lags': {lags}]")
        """

        # 2. Acceder al RandomForest
        print(f"{model_type}:")
        rf = modelo_completo['forecaster'].regressor if 'forecaster' in modelo_completo else modelo_completo

        # 3. Obtener hiperparámetros
        print(f"n_estimators: {rf.n_estimators}")
        print(f"max_depth: {rf.max_depth}")
        print(f"random_state: {rf.random_state}")

        # Para kernel (si está en metadata)
        if isinstance(modelo_completo, dict) and 'metadata' in modelo_completo:
            print(f"kernel: {modelo_completo['metadata'].get('kernel', 'N/A')}")


    elif model_type == "SVR":
        """
        kernel = modelo_completo['metadata']['kernel']
        c_optimizado = modelo_completo['metadata']['C_optimizado']
        epsilon = modelo_completo['metadata']['epsilon_optimizado']
        gamma = modelo_completo['metadata']['gamma_optimizado']
        lags = modelo_completo['metadata']['lags']

        print(f"{model_type}:")
        print(f"['kernel': {kernel}, 'C': {c_optimizado}, 'epsilon': {epsilon}, 'gamma': {gamma}, 'lags': {lags}]")
        """

        # 2. Acceder al SVR
        print(f"{model_type}:")
        svr = modelo_completo['forecaster'].regressor if 'forecaster' in modelo_completo else modelo_completo

        # 3. Imprimir hiperparámetros
        print(f"kernel: {svr.kernel}")
        print(f"C: {svr.C}")
        print(f"epsilon: {svr.epsilon}")
        print(f"gamma: {svr.gamma}")

    elif model_type == "Perceptron":
        """
        hidden_layer = modelo_completo['metadata']['hidden_layer_sizes_optimizado']
        activation = modelo_completo['metadata']['activation_optimizado']
        solver = modelo_completo['metadata']['solver_optimizado']
        max_iter = modelo_completo['metadata']['max_iter']
        alpha = modelo_completo['metadata']['alpha_optimizado']
        learning_rate = modelo_completo['metadata']['learning_rate_optimizado']
        learning_rate_init = modelo_completo['metadata']['learning_rate_init_optimizado']
        lags = modelo_completo['metadata']['lags']

        print(f"{model_type}:")
        print(f"['hidden_layer': {hidden_layer}, 'activation': {activation}, 'solver': {solver}, 'max_iter': {max_iter}, 'alpha': {alpha}, 'learning_rate': {learning_rate}, 'learning_rate_init': {learning_rate_init}, 'lags': {lags}]")
        """

        # 2. Acceder al MLP
        print(f"{model_type}:")
        mlp = modelo_completo['forecaster'].regressor if 'forecaster' in modelo_completo else modelo_completo

        # 3. Imprimir hiperparámetros
        print(f"hidden_layer_sizes: {mlp.hidden_layer_sizes}")
        print(f"activation: {mlp.activation}")
        print(f"solver: {mlp.solver}")
        print(f"max_iter: {mlp.max_iter}")
        print(f"alpha: {mlp.alpha}")
        print(f"learning_rate: {mlp.learning_rate}")
        print(f"learning_rate_init: {mlp.learning_rate_init}")
    
    else:
        print(f"{model_type}:")



