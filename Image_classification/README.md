
# Image Classification Project / Proyecto de Clasificación de Imágenes

## Walkthrough (English)

### 1. Data Exploration
- **EDA:** The process starts with `exploratory_data_analysis.ipynb`, where the dataset is checked for class balance and image quality. Unwanted files are removed, and sample images are visualized. The class distribution is analyzed to ensure a fair training set.

### 2. Model Creation
- **Data Preparation:** Images are loaded using TensorFlow, resized to 48x48 pixels, and converted to grayscale. The dataset is split into training and validation sets (80/20 split). The 'disgust' class is discarded due to insufficient samples.
- **Model Architecture:**
  - Input: 48x48 grayscale images
  - Several Conv2D layers (ReLU, He initialization)
  - MaxPooling and Global Average Pooling
  - Dense layers with dropout
  - Output: Dense layer with softmax for 6 emotion classes
- **Training:**
  - Optimizer: Adam
  - Loss: Sparse categorical crossentropy
  - Class weights are computed for balance
  - Trained for 13 epochs, batch size 32
- **Saving:** Model and metadata are saved for later use.

### 3. Model Evaluation
- After training, the model is evaluated on the validation set. Accuracy and loss are plotted per epoch. A classification report and confusion matrix are generated for detailed analysis.

### 4. Testing and Results
- **Testing:** In `Testing_model.ipynb`, the model is loaded and tested on unseen data. The test dataset is prepared similarly to training data.
- **Results:**
  - Overall test accuracy and a detailed classification report are printed.
  - Confusion matrices (raw and normalized) are plotted.
  - Sample predictions are shown, comparing model output to ground truth.
- **Performance:**
  - The model performs well across most classes, with precision, recall, and F1-scores reported for each.
  - Any weaknesses are highlighted in the confusion matrix analysis.

---

## Recorrido (Español)

### 1. Exploración de Datos
- **Análisis exploratorio:** El proceso inicia con `exploratory_data_analysis.ipynb`, donde se revisa el balance de clases y la calidad de las imágenes. Se eliminan archivos no deseados y se visualizan imágenes de muestra. Se analiza la distribución de clases para asegurar un conjunto de entrenamiento justo.

### 2. Creación del Modelo
- **Preparación de datos:** Las imágenes se cargan con TensorFlow, se redimensionan a 48x48 píxeles y se convierten a escala de grises. El conjunto se divide en entrenamiento y validación (80/20). La clase 'disgust' se descarta por falta de muestras.
- **Arquitectura del modelo:**
  - Entrada: imágenes en escala de grises de 48x48
  - Varias capas Conv2D (ReLU, inicialización He)
  - MaxPooling y Global Average Pooling
  - Capas densas con dropout
  - Salida: capa densa con softmax para 6 emociones
- **Entrenamiento:**
  - Optimizador: Adam
  - Pérdida: sparse categorical crossentropy
  - Se calculan pesos de clase para balancear
  - 13 épocas, batch size 32
- **Guardado:** El modelo y metadatos se guardan para uso posterior.

### 3. Evaluación del Modelo
- Tras el entrenamiento, el modelo se evalúa en el set de validación. Se grafican precisión y pérdida por época. Se genera un reporte de clasificación y una matriz de confusión para análisis detallado.

### 4. Pruebas y Resultados
- **Pruebas:** En `Testing_model.ipynb`, el modelo se carga y prueba con datos no vistos. El set de prueba se prepara igual que el de entrenamiento.
- **Resultados:**
  - Se imprime la precisión general y un reporte de clasificación detallado.
  - Se grafican matrices de confusión (cruda y normalizada).
  - Se muestran predicciones de ejemplo, comparando salida del modelo con la verdad.
- **Desempeño:**
  - El modelo tiene buen desempeño en la mayoría de las clases, reportando precisión, recall y F1-score para cada una.
  - Cualquier debilidad se resalta en el análisis de la matriz de confusión.

