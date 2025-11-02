# me-verifier

Sistema de **verificación de identidad facial** basado en embeddings faciales y aprendizaje automático (Logistic Regression).  
El proyecto permite detectar, recortar rostros, generar embeddings con una red preentrenada y clasificar si una imagen corresponde (“ME” vs “NOT ME”).

## Versiones recomendada para python

Python 3.11 o 3.12

## Estructura del proyecto

``` 
me-verifier/ 
├─ api/ 
│ └─ app.py # Flask (/healthz, /verify) 
├─ models/ 
│ ├── embeddings.joblib # Embeddings preprocesados
│ ├── model_verifier.joblib # Modelo entrenado (Logistic Regression)
│ ├── scaler.joblib # Escalador normalizador (StandardScaler)
│ └── val_data.joblib # Datos reservados para la evaluacion del modelo
├─ data/ 
│ ├─ me/ # fotos crudas 
│ ├─ not_me/ # negativos 
│ └─ cropped/ # recortes faciales 
├─ scripts/ 
│ ├─ crop_faces.py 
│ ├─ embeddings.py 
│ └─ run_gunicorn.sh 
├── reports/
│ ├── metrics.json # Métricas de evaluación
│ └── confusion_matrix.png # Gráfico de matriz de confusión
├── ui/
│ ├── ui_app.py # Aplicación Flask para predecir desde imágenes
│ ├── static/
│ └── templates/
├─ train.py 
├─ evaluate.py 
├─ tests/test_api.py
├─ .gitignore
├─ .env.example 
├─ README.md 
└─ requirements.txt
```

## Instalacion

1. Clonar el repositorio

    ```bash
    git clone https://github.com/Ignacio-Us/me-verifier.git
    ```

    ```bash
    cd me-verifier
    ```

2. Crear el entorno virtual
    ```
    py -3.11 -m venv venv
    ```

    ```
    source venv/bin/activate   # Linux / Mac
    ```

    ```
    venv\Scripts\activate      # Windows
    ```

3. Instalar dependencias

    ```
    pip install -r requirements.txt
    ```

## Proceso de ejecucion

1. Recorte facial:

    ```
    python scripts/crop_faces.py
    ```

    Detecta rostros en 'data/me' y 'data/not_me' y guarda versiones recortadas en 'data/cropped/'.

2. Generación de embeddings:

    ```
    python scripts/embeddings.py
    ```

    Convierte las imágenes recortadas en vectores de características ('embeddings.joblib').

3. Entrenamiento del modelo:

    ```
    python train.py
    ```

    Entrena una regresión logística (model_verifier.joblib) y guarda el scaler.

4. Evaluación del modelo:

    ```
    python evaluate.py
    ```

    Calcula métricas de rendimiento y genera una matriz de confusión.

5. Despliegue del servicio web:

    ```
    python ui/ui_app.py
    ```

    O

    ```
    bash run_gunicorn.sh
    ```

    Inicia el servidor Flask con o sin Gunicorn para realizar predicciones sobre nuevas imágenes.