California Housing

Proyecto de regresión para predecir el valor medio de viviendas utilizando RandomForestRegressor. El entrenamiento se ejecuta desde train.py y las corridas se registran con MLflow.

Instalación

Instalar las dependencias:

python -m pip install -r requirements.txt

Iniciar MLflow

Desde la carpeta del proyecto:

mlflow server

Abrir en el navegador:

http://127.0.0.1:5000

Ejecutar el modelo

Con los valores por defecto:

python train.py

Valores por defecto:

n_estimators = 100
max_depth = 10
min_samples_split = 2
seed = 42

También pueden modificarse desde la terminal:

python train.py --n_estimators 200 --max_depth 15 --min_samples_split 5 --seed 42

Seguimiento de experimentos

Las ejecuciones se registran en el experimento:

California_Housing_RandomForest

Cada corrida registra:

Hiperparámetros
MAE, MSE, RMSE y R²
Gráfica de valores reales vs. predichos
Modelo entrenado
Reproducibilidad

La semilla se fija mediante --seed y se utiliza tanto para la división de los datos como para el entrenamiento del Random Forest. Ejecutar el modelo con los mismos datos, argumentos, semilla y entorno permite reproducir los resultados.