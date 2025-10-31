# # ⛏️ Text Mining: Análisis de Sentimiento de Reviews de Vuelos

Este proyecto, desarrollado por **Santiago Londoño Pérez** (Versión 2025), es un notebook de Jupyter (`TextMising.ipynb`) dedicado al **Procesamiento de Lenguaje Natural (NLP)**.

El objetivo principal es construir y evaluar un modelo de Machine Learning capaz de clasificar el sentimiento (positivo, negativo o neutral) de tweets dirigidos a aerolíneas estadounidenses.

El dataset utilizado es `Tweets.csv`, que contiene miles de tweets reales del "Twitter US Airline Sentiment".

## 🤖 Flujo de Trabajo del Proyecto

El notebook sigue un flujo de trabajo estructurado de NLP y Machine Learning, dividido en 8 etapas clave:

### 1. Carga de Datos
* Importación de librerías (`pandas`, `numpy`, `matplotlib`, `seaborn`).
* Carga del dataset `Tweets.csv`.
* Exploración inicial con `df.head()`, `df.info()` y `df.airline_sentiment.value_counts()`.

### 2. Análisis Exploratorio de Datos (EDA)
* **Distribución de Sentimientos:** Visualización con `sns.countplot` para entender el balance de las clases (negativo, neutral, positivo).
* **Sentimiento por Aerolínea:** Análisis de qué aerolíneas reciben qué tipo de sentimiento, usando `sns.countplot` con el parámetro `hue`.

### 3. Preprocesamiento y Limpieza de Texto
Esta es la etapa fundamental de NLP. Se crea una función `clean_text` que aplica las siguientes transformaciones a cada tweet:
* Elimina URLs (`http...`).
* Elimina menciones de usuario (`@usuario`).
* Elimina el símbolo de hashtag (`#`), pero conserva la palabra.
* Elimina todos los caracteres no alfabéticos (puntuación, números, etc.).
* Convierte todo el texto a **minúsculas**.
* Elimina **Stopwords** (palabras comunes como 'the', 'is', 'in') usando la librería `NLTK`.

### 4. Vectorización (Bag of Words)
* Para que el modelo entienda el texto, se convierte en números.
* Se utiliza `CountVectorizer` de `sklearn` para crear una matriz de "Bolsa de Palabras".
* Se limita el vocabulario a las 5,000 palabras (features) más frecuentes para optimizar el modelo.

### 5. Preparación para el Modelo
* Mapeo de las etiquetas de sentimiento a números (ej. `negative` -> 0, `neutral` -> 1, `positive` -> 2).
* División de los datos en conjuntos de entrenamiento y prueba (`train_test_split`) con una proporción 80/20.

### 6. Creación y Entrenamiento del Modelo
* Se selecciona el algoritmo **Multinomial Naive Bayes** (`MultinomialNB`), un modelo clásico y muy eficaz para la clasificación de texto.
* El modelo se entrena con los datos de `X_train` y `y_train`.

### 7. Evaluación del Modelo
* Se realizan predicciones sobre el conjunto de prueba (`X_test`).
* Se evalúa el rendimiento del modelo usando:
    * **Accuracy Score:** Precisión general.
    * **Classification Report:** Detalle de Precisión (Precision), Sensibilidad (Recall) y F1-Score para cada clase.
    * **Matriz de Confusión:** Visualización con `sns.heatmap` para entender qué clases confunde el modelo (ej. predecir 'neutral' cuando era 'negativo').

### 8. Prueba del Modelo con Nuevas Frases
* Se crea una función final `predict_sentiment` que toma un texto nuevo, aplica **todo el pipeline de limpieza y vectorización**, y devuelve la predicción del modelo.
* Se prueba con ejemplos como "The flight was amazing" y "The flight was delayed".

## 🛠️ Librerías Utilizadas

* **Análisis de Datos:** `pandas`, `numpy`
* **Visualización:** `matplotlib`, `seaborn`
* **NLP:** `re` (Expresiones Regulares), `nltk` (para Stopwords)
* **Machine Learning (Scikit-learn):**
    * `feature_extraction.text.CountVectorizer`
    * `model_selection.train_test_split`
    * `naive_bayes.MultinomialNB`
    * `metrics` (accuracy_score, classification_report, confusion_matrix)
