# Implementación de algoritmo KNN y comparación

El algoritmo desarrollado encapsula la lógica de K-Nearest-Neighbors y mantienen una estructura similar a los métodos ofrecidos en Scikit-learn

## Clase K-Nearest-Neighbors
Brinda 3 métodos:
- 'fit(X,y)' : A menudo X_train y Y_train
- 'predict(X)': A menudo X_test o una fila individual 
- 'score(X,y)': A menudo X_test, Y_test

## Limitaciones
Se requiere un label-encoder en la columna "target" de un dataset. El módulo adjunto "utils" brinda algunos métodos que facilitan el manejo de los datasets.  
Verá que son adaptaciones de otras librerias y dan un formato específico.

Este proyecto busca replicar la lógica del algoritmo KNN y no busca complicarse con los métodos de lectura y procesamiento de los conjuntos de datos,  
aún asi se agregó soporte para tipos "numpy.ndarray" y "pandas.Dataframe".

## Pruebas

### Dataset: Iris.data.csv
En el archivo "knn/__init__.py" encontrará las sentencias a ejecutar para obtener predicciones y el score del dataset de Iris

### Comparación con modelo KNN de Scikit-learn y sus parámetros por defecto:
En busca de validar que tan preciso es nuestro modelo respecto a una librería con un desarrollo y dedicación considerable, se hizo uso de un notebook
proveido en los ejemplos de la propia librería en su documentación. Le dejaré el hipervinculo abajo:

[Enlace](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)

El resultado fue muy similar:

<img src="results/comparative.png">

El scripts se puede encontrar en "main.py", el cual fue modificado en cierto modo para soportar el algoritmo creado.

## Instalación y ejecución
- Es recomendable instalar la versión 1.1.0 o superior de la librería scikit-learn
- Encontrará un archivo "requirements.txt", uselo para crear un entorno venv e instalar las librerias
```python
python -m venv "path/venv"
pip install -r requirements.txt
```
- Ejecute main.py




