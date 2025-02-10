# NN_Explained

# Lienzo Conceptual Extenso: Arquitecturas de Redes Neuronales y sus Aplicaciones

Imaginemos un gran lienzo (o mural) dividido en diferentes secciones (cuadros). Cada una de estas secciones se dedica a explorar un tipo de red neuronal, profundizando en sus principios de funcionamiento, componentes, ventajas, desventajas, **aplicaciones**, **definiciones matemáticas**, **ejemplos paso a paso**, **cómo preparar los datos** y ahora también **ejemplos básicos de implementación en Python** (usando Keras, parte de TensorFlow).

---

## CUADRO 1: Redes Neuronales Feedforward (FNN) o MLP

### 1.1. Definición
Las Redes Neuronales Feedforward (FNN), también conocidas como **Multilayer Perceptron (MLP)**, constituyen la forma más clásica y básica de red neuronal. En ellas, la información fluye en un solo sentido: desde la capa de entrada, pasando por una o más capas ocultas, hasta la capa de salida.

### 1.2. Arquitectura
1. **Capa de entrada:** Recibe los datos de entrada (características o "features"). Por ejemplo, en un problema de clasificación de dígitos, cada píxel puede ser una neurona de la capa de entrada.
2. **Capas ocultas:** Una o varias capas que realizan transformaciones intermedias sobre los datos. Cada neurona en estas capas calcula una suma ponderada de sus entradas, seguida de una función de activación (sigmoid, tanh, ReLU, etc.).
3. **Capa de salida:** Emite la predicción final. Por ejemplo, en un problema de clasificación binaria, podría ser una sola neurona con activación sigmoid, mientras que en una clasificación multiclase se utiliza softmax.

### 1.3. Entrenamiento
El entrenamiento comúnmente se hace mediante **backpropagation** (retropropagación), usando un optimizador como Gradient Descent, Adam u otros. La idea:
1. Se calcula la salida de la red para una entrada.
2. Se mide el error en función de la salida deseada.
3. Se propaga el error hacia atrás, ajustando los pesos para minimizar la pérdida.

### 1.4. Aplicaciones
- **Clasificación:** Detección de spam, clasificación de imágenes simples, reconocimiento de patrones generales.
- **Regresión:** Predicción de precios (acciones, inmuebles), estimaciones de rendimiento.
- **Sistemas de recomendación:** Combinado con otras técnicas.

### 1.5. Fortalezas y Limitaciones
- **Ventajas:**
  - Estructura simple y fácil de implementar.
  - Útiles como "redes base" en muchos problemas.
- **Desventajas:**
  - Escalan mal cuando el problema requiere detectar estructuras complejas (ej. imágenes, secuencias).
  - Pueden precisar muchas capas para problemas más avanzados, incrementando el riesgo de overfitting y el coste computacional.

### 1.6. Definición Matemática y Ejemplo Paso a Paso
Para tener una idea más formal, consideremos una red neuronal feedforward sencilla con una capa oculta. Suponiendo:

- Una capa de entrada de dimensión \(d\) (\(x \in \mathbb{R}^{d}\)).
- Una capa oculta con \(h\) neuronas.
- Una capa de salida con \(k\) neuronas.

**Paso 1: Capa Oculta**

Sea \(W^{(1)}\) una matriz de pesos de tamaño \(h \times d\) y \(b^{(1)}\) un sesgo (bias) de dimensión \(h\). Para una entrada \(x\), la activación de la capa oculta \(z^{(1)}\) se obtiene como:
\[
  z^{(1)} = W^{(1)} x + b^{(1)}.\]

Luego, aplicamos una función de activación \(\sigma\) (por ejemplo, ReLU o sigmoid) de manera elemento a elemento:
\[
  a^{(1)} = \sigma\bigl(z^{(1)}\bigr),
\]

obteniendo así un vector de dimensión \(h\) que representa la salida de la capa oculta.

**Paso 2: Capa de Salida**

Definimos \(W^{(2)}\) como la matriz de pesos de tamaño \(k \times h\) y \(b^{(2)}\) como el sesgo de dimensión \(k\). La salida de la red (antes de la activación final) es:
\[
  z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}.
\]

Si es un problema de clasificación multiclase, se suele aplicar softmax:
\[
  \text{salida} = \text{softmax}\bigl(z^{(2)}\bigr),
\]

mientras que para una regresión podemos usar una salida lineal o alguna otra función activación apropiada.

**Paso 3: Ejemplo Numérico Sencillo**

Supongamos:
- \(d = 2\) (dos entradas), \(h = 2\) (dos neuronas en la capa oculta), \(k = 1\) (una neurona de salida para clasificación binaria).
- Una función de activación \(\sigma\) = ReLU(x) = max(0, x).

1. **Entrada:** \(x = \begin{pmatrix} 0.5 \\ 1.0 \end{pmatrix}\).
2. **Pesos y sesgos de la capa oculta:**
   \(
   W^{(1)} = \begin{pmatrix}
   0.2 & -0.1 \\
   0.4 &  0.3 
   \end{pmatrix}, \quad
   b^{(1)} = \begin{pmatrix}
   0.0 \\ -0.1
   \end{pmatrix}.
   \)
3. **Cálculo en la capa oculta:**
   \(
   z^{(1)} = W^{(1)} x + b^{(1)} = \begin{pmatrix}
   0.2 & -0.1 \\
   0.4 &  0.3 
   \end{pmatrix} \begin{pmatrix} 0.5 \\ 1.0 \end{pmatrix} + \begin{pmatrix} 0.0 \\ -0.1 \end{pmatrix}
   = \begin{pmatrix} 0.1 - 0.1 \\ 0.2 + 0.3 \end{pmatrix} + \begin{pmatrix} 0.0 \\ -0.1 \end{pmatrix}
   = \begin{pmatrix} 0.0 \\ 0.5 \end{pmatrix} + \begin{pmatrix} 0.0 \\ -0.1 \end{pmatrix}
   = \begin{pmatrix} 0.0 \\ 0.4 \end{pmatrix}.
   \)
   Aplicando ReLU:
   \(
   a^{(1)} = \begin{pmatrix} max(0, 0.0) \\ max(0, 0.4) \end{pmatrix} = \begin{pmatrix} 0.0 \\ 0.4 \end{pmatrix}.
   \)
4. **Pesos y sesgos de la capa de salida:**
   \(
   W^{(2)} = \begin{pmatrix} 0.5 & -0.2 \end{pmatrix}, \quad b^{(2)} = 0.1.
   \)
   Nótese que la salida es escalar (dimensión 1), así que \(W^{(2)}\) es 1x2 y \(b^{(2)}\) es un escalar.
5. **Cálculo de la salida:**
   \(
   z^{(2)} = W^{(2)} a^{(1)} + b^{(2)} = \begin{pmatrix} 0.5 & -0.2 \end{pmatrix} \begin{pmatrix} 0.0 \\ 0.4 \end{pmatrix} + 0.1
   = 0.5 * 0.0 + -0.2 * 0.4 + 0.1 = 0.0 - 0.08 + 0.1 = 0.02.
   \)
   Si fuera una clasificación binaria con sigmoid, la salida final sería:
   \(
     \text{salida} = \sigma(z^{(2)}) = \frac{1}{1+ e^{-0.02}} \approx 0.505.
   \)
   Esto indicaría, por ejemplo, una probabilidad de 50.5% de pertenecer a la clase "1".

Este pequeño ejemplo ilustra cómo se calculan paso a paso las salidas en una red feedforward.

### 1.7. Preparación de Datos e Inputs
- **Normalización o Estandarización:** Se suele escalar cada característica para que tenga media 0 y desviación 1 (o mínimo 0 y máximo 1). Esto acelera la convergencia.
- **Formateo de Entradas:** Para MLP, se suelen emplear vectores o tensores 1D (si hay más dimensiones, se suelen "aplanar" o condensar).
- **Manejo de Valores Perdidos:** Es importante imputar o descartar valores nulos para evitar inconsistencias en el entrenamiento.
- **Codificación de Variables Categóricas:** One-hot, label encoding u otras técnicas si hay atributos no numéricos.

### 1.8. Ejemplo de Creación de la Red y Entrenamiento en Python

**Ejemplo con Keras (TensorFlow):**
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Datos sintéticos de ejemplo
X_train = np.random.rand(1000, 2)  # 1000 muestras, 2 features
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)  # Etiqueta binaria

# Definir modelo
model = Sequential()
model.add(Dense(2, input_shape=(2,), activation='relu'))  # Capa oculta con 2 neuronas
model.add(Dense(1, activation='sigmoid'))                 # Capa de salida (binaria)

# Compilar
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=10, batch_size=32)
```
En este ejemplo:
1. Se generan datos sintéticos aleatorios con 2 características. La etiqueta es 1 si la suma de las características es mayor que 1.
2. Definimos una red MLP con 1 capa oculta.
3. Se entrena con 10 épocas y lotes de 32.

---

## CUADRO 2: Redes Neuronales Convolucionales (CNN)

### 2.1. Definición
Las CNN (Convolutional Neural Networks) son redes pensadas inicialmente para analizar imágenes, aunque también se aplican en audio, texto (transformando en "imágenes" de embeddings), y otros datos con estructura de cuadrícula.

### 2.2. Elementos Principales
1. **Capas Convolucionales:** Se aplican "filtros" que recorren la imagen o matriz de datos para extraer patrones locales (bordes, contornos, texturas). Cada filtro produce un "mapa de características".
2. **Capas de Pooling:** Reducen la dimensión espacial (ej. max pooling). Esto ayuda a lograr invariancia frente a pequeñas traslaciones.
3. **Capas Completamente Conectadas:** Al final de la CNN, a menudo se incluyen para la clasificación.

### 2.3. Entrenamiento
Se basa también en backpropagation, pero adaptada a las operaciones convolucionales. El uso de GPUs se vuelve especialmente efectivo dado el gran volumen de datos e imágenes.

### 2.4. Aplicaciones
- **Visión por Computadora:**
  - Clasificación de imágenes (por ejemplo, reconocer animales, objetos, rostros).
  - Detección de objetos (ej. YOLO, Faster R-CNN).
  - Segmentación de imágenes (ej. U-Net).
- **Análisis de audio:** Reconocimiento de patrones en espectrogramas.
- **Procesamiento de texto en 2D:** Embeddings dispuestos como mapas 2D.

### 2.5. Fortalezas y Limitaciones
- **Ventajas:**
  - Capturan patrones espaciales de manera eficiente.
  - Menos parámetros que una red totalmente conectada equivalente.
- **Desventajas:**
  - Modelan bien la información local, pero pueden requerir trucos extra para capturar relaciones globales.
  - Siguen siendo costosas en términos de cómputo.

### 2.6. Definición Matemática y Ejemplo de Cálculo

Considere una imagen de entrada de tamaño \(H \times W\) con \(C\) canales (p. ej., RGB). Una operación de convolución con un filtro de tamaño \(k_h \times k_w\) (alto y ancho), y "stride" = 1 para simplificar.

**Paso 1: Cálculo Convolucional**

Sea \(X\) la porción de la imagen cubierta por el filtro \(F\) en un paso dado. Si \(F\) también posee \(C\) canales, entonces el valor resultante para un elemento \(y\) del mapa de características se define:
\[
  y = \sum_{c=1}^{C} \sum_{i=1}^{k_h} \sum_{j=1}^{k_w} F_{c,i,j} \cdot X_{c,i,j} + b,
\]

donde \(b\) es un sesgo asociado al filtro.

**Paso 2: Ejemplo Simplificado**

Suponiendo:
- Una imagen en escala de grises (\(C = 1\)) de tamaño 3x3:
  \(
   X = \begin{pmatrix}
   1 & 2 & 3\\
   4 & 5 & 6\\
   7 & 8 & 9
   \end{pmatrix}.
  \)
- Filtro (kernel) de tamaño 2x2 (\(k_h = 2, k_w = 2\)):
  \(
   F = \begin{pmatrix}
   1 & 0\\
   0 & -1
   \end{pmatrix}, \quad b = 0.
  \)
- Stride = 1, sin padding.

La salida \(Y\) tendrá dimensión 2x2 (para cada posición del filtro en la imagen). Los cálculos:
1. **Posición (fila=1, col=1)**: \(
     X_{\text{sub}} = \begin{pmatrix} 1 & 2\\ 4 & 5 \end{pmatrix}.
   \)
   \(
     y_{1,1} = 1*1 + 0*2 + 0*4 + (-1)*5 = 1 - 5 = -4.
   \)
2. **Posición (fila=1, col=2)**:
   \(
     X_{\text{sub}} = \begin{pmatrix} 2 & 3\\ 5 & 6 \end{pmatrix},
   \)
   \(
     y_{1,2} = 1*2 + 0*3 + 0*5 + (-1)*6 = 2 - 6 = -4.
   \)
3. **Posición (fila=2, col=1)**:
   \(
     X_{\text{sub}} = \begin{pmatrix} 4 & 5\\ 7 & 8 \end{pmatrix},
   \)
   \(
     y_{2,1} = 1*4 + 0*5 + 0*7 + (-1)*8 = 4 - 8 = -4.
   \)
4. **Posición (fila=2, col=2)**:
   \(
     X_{\text{sub}} = \begin{pmatrix} 5 & 6\\ 8 & 9 \end{pmatrix},
   \)
   \(
     y_{2,2} = 1*5 + 0*6 + 0*8 + (-1)*9 = 5 - 9 = -4.
   \)

Así, el mapa de características resultante \(Y\) es:
\(
  Y = \begin{pmatrix}
  -4 & -4 \\
  -4 & -4
  \end{pmatrix}.
\)

Este ejemplo demuestra cómo se lleva a cabo la operación de convolución en un caso muy sencillo.

### 2.7. Preparación de Datos e Inputs
- **Formateo de Imágenes:** Se acostumbra tener tensores \((N, C, H, W)\), donde \(N\) es el número de ejemplos, \(C\) los canales, \(H\) y \(W\) la altura y anchura.
- **Normalización por Canal:** Es común restar la media y dividir por la desviación estándar (calculadas en el conjunto de entrenamiento) para cada canal.
- **Data Augmentation (Imágenes):** Rotaciones, flips, cambios de color para aumentar la diversidad de los datos y evitar overfitting.
- **Para Audio:** Se suele convertir a espectrogramas (2D) y tratarlos como "imágenes".
- **Para Texto (en 2D):** Representar embeddings como matrices si se quiere usar convoluciones.

### 2.8. Ejemplo de Creación de la Red y Entrenamiento en Python

**Ejemplo con Keras (TensorFlow) para Clasificación de Imágenes (simplificado):**
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Supongamos que tenemos 1000 imágenes de 28x28 en escala de grises (1 canal)
X_train = np.random.rand(1000, 28, 28, 1).astype(np.float32)
# Etiquetas para 10 clases (por ejemplo, dígitos 0-9)
y_train = np.random.randint(0, 10, size=(1000,))

# Convertimos las etiquetas a one-hot
y_train_onehot = np.zeros((1000, 10))
for i, label in enumerate(y_train):
    y_train_onehot[i, label] = 1

model = Sequential()
model.add(Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))  # salida para 10 clases

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_onehot, epochs=5, batch_size=32)
```
1. Se generan datos sintéticos (28x28, 1 canal, 10 clases).
2. Definimos una CNN muy sencilla con una capa conv, pooling, flatten y capa densa final.
3. Entrenamos durante 5 épocas.

---

## CUADRO 3: Redes Neuronales Recurrentes (RNN)

### 3.1. Definición
Las RNN (Recurrent Neural Networks) se utilizan para datos secuenciales (texto, audio, series temporales), permitiendo que la red "recuerde" información de pasos previos a través de conexiones recurrentes en su arquitectura.

### 3.2. Arquitectura
1. **Estado oculto (hidden state):** actúa como "memoria", se actualiza en cada paso de la secuencia.
2. **Entrada secuencial:** cada nuevo elemento de la secuencia (palabra, vector temporal) se procesa con el estado oculto anterior.
3. **Salida secuencial u oculta:** se puede generar una salida por cada paso (muchos a muchos) o solo tras procesar toda la secuencia (muchos a uno).

### 3.3. Entrenamiento: Backpropagation Through Time (BPTT)
Se "desenrolla" la red en el tiempo, lo que puede hacer que el entrenamiento sea más costoso y provoque problemas de vanishing o exploding gradients.

### 3.4. Aplicaciones
- **NLP (Procesamiento de Lenguaje Natural):** Análisis de sentimiento, clasificación de oraciones, modelado de lenguaje.
- **Traducción automática:** Un RNN puede codificar la frase de entrada y otro RNN decodificar al idioma de destino.
- **Series temporales:** Predicción de datos financieros, sensores, clima.

### 3.5. Fortalezas y Limitaciones
- **Ventajas:**
  - Manejan datos secuenciales de forma natural.
  - Capturan dependencias temporales.
- **Desventajas:**
  - Dificultad para aprender dependencias a largo plazo.
  - Entrenamiento lento en secuencias largas.

### 3.6. Definición Matemática y Ejemplo Paso a Paso

Sea una RNN simple con estado oculto \(h_t\) en el paso \(t\). Dada una entrada \(x_t\), se define:
\[
  h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b),
\]
donde \(W_h\) y \(W_x\) son matrices de pesos y \(b\) es el sesgo.

**Paso 1: Ejemplo en 2 pasos**
- Dimensión del estado oculto: 2.
- Dimensión de la entrada: 1 (un solo valor por paso).

Supongamos:
\(
  W_h = \begin{pmatrix} 0.5 & 0.0 \\ 0.1 & 0.5 \end{pmatrix}, \quad
  W_x = \begin{pmatrix} 0.6 \\ -0.4 \end{pmatrix}, \quad
  b = \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}.
\)

- **Paso 0**: \(h_0 = \begin{pmatrix} 0.0 \\ 0.0 \end{pmatrix}\) (estado inicial).
- **Entrada en t=1**: \(x_1 = 1.0\).

\(
  h_1 = \tanh\Bigl(W_h h_0 + W_x x_1 + b\Bigr) = \tanh\Bigl(\begin{pmatrix} 0.5 & 0.0 \\ 0.1 & 0.5 \end{pmatrix}\begin{pmatrix} 0.0 \\ 0.0 \end{pmatrix} + \begin{pmatrix} 0.6 \\ -0.4 \end{pmatrix} * 1.0 + \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}\Bigr).
\)

\(
  = \tanh\Bigl(\begin{pmatrix} 0.0 \\ 0.0 \end{pmatrix} + \begin{pmatrix} 0.6 \\ -0.4 \end{pmatrix} + \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}\Bigr)
  = \tanh\Bigl(\begin{pmatrix} 0.6 \\ -0.3 \end{pmatrix}\Bigr).
\)

\(\tanh(0.6) \approx 0.537\), \(\tanh(-0.3) \approx -0.291\). Por tanto:
\(
  h_1 \approx \begin{pmatrix} 0.537 \\ -0.291 \end{pmatrix}.
\)

- **Entrada en t=2**: \(x_2 = 2.0\).
\(
  h_2 = \tanh\Bigl(W_h h_1 + W_x x_2 + b\Bigr).
\)

Primero calculamos \(W_h h_1\):
\(
  W_h h_1 = \begin{pmatrix} 0.5 & 0.0 \\ 0.1 & 0.5 \end{pmatrix}\begin{pmatrix} 0.537 \\ -0.291 \end{pmatrix}
             = \begin{pmatrix} 0.5 * 0.537 + 0.0 * -0.291 \\ 0.1 * 0.537 + 0.5 * -0.291 \end{pmatrix}
             = \begin{pmatrix} 0.2685 \\ 0.0537 - 0.1455 \end{pmatrix}
             = \begin{pmatrix} 0.2685 \\ -0.0918 \end{pmatrix}.
\)

\(
  W_x x_2 = \begin{pmatrix} 0.6 \\ -0.4 \end{pmatrix} * 2.0 = \begin{pmatrix} 1.2 \\ -0.8 \end{pmatrix}.
\)

Sumamos todo:
\(
  z_2 = \begin{pmatrix} 0.2685 \\ -0.0918 \end{pmatrix} + \begin{pmatrix} 1.2 \\ -0.8 \end{pmatrix} + \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}
       = \begin{pmatrix} 1.4685 \\ -0.7918 \end{pmatrix}.
\)

Aplicando \(\tanh\):
\(
  h_2 = \tanh\Bigl(\begin{pmatrix} 1.4685 \\ -0.7918 \end{pmatrix}\Bigr) \approx \begin{pmatrix} 0.899 \  -0.659 \end{pmatrix}.
\)

Así, se ha obtenido el estado oculto en el paso 2. Este ejemplo ilustra cómo una RNN simple procesa secuencias y mantiene un estado que va evolucionando.

### 3.7. Preparación de Datos e Inputs
- **Representación de Secuencias:** Para texto, normalmente se tokeniza cada frase y se convierte en índices o embeddings. Las secuencias pueden rellenarse (padding) para tener una longitud fija.
- **Normalización de Series Temporales:** Se puede restar la media y dividir por la desviación estándar de cada característica temporal.
- **Batching de Secuencias:** Para entrenar en lotes, se suelen agrupar secuencias de longitud similar o recortar/padding para tener tamaños uniformes.
- **Enmascaramiento:** Cuando hay diferentes longitudes, se utiliza un "mask" para ignorar la parte "inútil".

### 3.8. Ejemplo de Creación de la Red y Entrenamiento en Python

**Ejemplo con Keras (TensorFlow) para secuencias simples:**
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Ejemplo: supongamos secuencias de longitud 5, con 1 feature
# Generamos 1000 secuencias aleatorias
X_train = np.random.rand(1000, 5, 1)
# Etiqueta binaria: 1 si la media de la secuencia > 0.5
y_train = (X_train.mean(axis=1).flatten() > 0.5).astype(int)

model = Sequential()
model.add(SimpleRNN(4, input_shape=(5, 1), activation='tanh'))  # Estado oculto de dimensión 4
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)
```
1. Se crean secuencias de 5 pasos, cada paso con 1 feature.
2. La RNN (SimpleRNN) produce un solo vector de salida al final, que se pasa a una capa densa binaria.
3. Se entrena por 5 épocas.

---

## CUADRO 4: LSTM (Long Short-Term Memory) y GRU (Gated Recurrent Unit)

### 4.1. LSTM
Las LSTM solucionan en gran parte el problema de desvanecimiento de gradientes usando una "celda" interna y **tres puertas**:
1. **Puerta de Olvido (forget gate):** decide qué parte del estado anterior mantener.
2. **Puerta de Entrada (input gate):** determina cuánta información nueva se agrega.
3. **Puerta de Salida (output gate):** regula la salida de la celda.

**Aplicaciones:**
- Traducción automática.
- Modelado del lenguaje.
- Reconocimiento de voz.

### 4.2. GRU
Las GRU simplifican la arquitectura de las LSTM, reduciendo a dos puertas (actualización y reinicio), manteniendo un rendimiento similar en muchos casos.

**Aplicaciones:**
- Chatbots.
- Análisis de series temporales (más corto y eficiente que LSTM).

### 4.3. Fortalezas y Limitaciones
- **Ventajas:**
  - LSTM y GRU pueden retener información de secuencias más largas que una RNN simple.
  - Menos problemas de vanishing gradient.
- **Desventajas:**
  - Más parámetros que una RNN básica.
  - Entrenamientos largos, especialmente para secuencias muy extensas.

### 4.4. Definición Matemática y Ejemplo Paso a Paso (LSTM)

Las LSTM mantienen dos tipos de estado: \(h_t\) (estado oculto) y \(c_t\) (estado de celda). Para un paso \(t\):
\[
  f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad (\text{forget gate}),
\]
\[
  i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad (\text{input gate}),
\]
\[
  \tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad (\text{candidata de celda}),
\]
\[
  c_t = f_t * c_{t-1} + i_t * \tilde{c}_t,\]
\[
  o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad (\text{output gate}),
\]
\[
  h_t = o_t * \tanh(c_t).
\]

**Ejemplo Simplificado**:
- Dimensiones pequeñas y valores aleatorios.
- Suponga \(h_0 = 0, c_0 = 0\).
- \(x_1\) es un vector de dimensión 2.

Aunque sea más complejo, la idea es que cada puerta filtra cierta parte de la información, ayudando a evitar la saturación o la pérdida de gradientes a lo largo del tiempo.

### 4.5. Preparación de Datos e Inputs
- **Similar a RNN:** Necesitamos vectorizar o "embeddizar" las secuencias (texto, audio, etc.).
- **Longitudes de Secuencia:** Igual que en RNN, se usan técnicas de padding o truncamiento para batches.
- **Normalización:** A menudo se normalizan características si son datos numéricos (p. ej. series temporales).
- **Considerar Librerías Específicas:** Muchos frameworks (TensorFlow, PyTorch) tienen capas de LSTM/GRU que requieren tensores con la forma (longitud, batch_size, features) o similar.

### 4.6. Ejemplo de Creación de la Red y Entrenamiento en Python
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# Ejemplo: secuencias de longitud 10, 1 feature
X_train = np.random.rand(500, 10, 1)
y_train = (X_train.mean(axis=1).flatten() > 0.5).astype(int)

model = Sequential()
# Podemos usar LSTM o GRU. Aquí usamos LSTM:
model.add(LSTM(8, input_shape=(10, 1)))  # 8 neuronas LSTM
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)
```
1. Genera 500 secuencias de longitud 10.
2. LSTM con 8 neuronas en su capa.
3. Entrena durante 5 épocas.

---

## CUADRO 5: Autoencoders

### 5.1. Definición
Un **Autoencoder** es una red diseñada para aprender representaciones comprimidas (codificaciones) de los datos de manera no supervisada. El objetivo principal es reconstruir la entrada en la salida, forzando a la red a comprimir información.

### 5.2. Arquitectura
1. **Encoder:** transforma los datos de entrada en una representación de menor dimensión (bottleneck).
2. **Latent Space:** capa intermedia de dimensión reducida.
3. **Decoder:** intenta reconstruir la entrada original a partir de la representación latente.

### 5.3. Variantes
- **Denoising Autoencoder:** se entrena con entradas ruidosas para que aprenda a recuperar la señal limpia.
- **Variational Autoencoders (VAE):** abordan la generación de datos desde un punto de vista probabilístico.

### 5.4. Aplicaciones
- **Reducción de dimensionalidad:** de forma no lineal, a diferencia de PCA.
- **Detección de anomalías:** si algo es "muy diferente" a los datos de entrenamiento, se reconstruye mal.
- **Generación de datos:** en VAE, se pueden generar muestras realistas de datos similares a los originales.

### 5.5. Fortalezas y Limitaciones
- **Ventajas:**
  - Aprenden sin necesidad de etiquetas.
  - Útiles para entender la estructura de los datos.
- **Desventajas:**
  - Pueden tender a memorizar si la capacidad es alta.
  - Más complejos que métodos lineales como PCA.

### 5.6. Definición Matemática y Ejemplo Paso a Paso

Sea un autoencoder con:
- Encoder: \(h = f_\theta(x)\)
- Decoder: \(\hat{x} = g_\phi(h)\)

El entrenamiento minimiza un error de reconstrucción, por ejemplo \(\|x - \hat{x}\|^2\) o la entropía cruzada.

**Paso 1: Ejemplo sencillo**
- Entrada \(x\) de dimensión 2.
- Bottleneck (espacio latente) de dimensión 1.

\(
  h = f_\theta(x) = \sigma(W^{(enc)} x + b^{(enc)}),
\)
\(
  \hat{x} = g_\phi(h) = \sigma(W^{(dec)} h + b^{(dec)}).
\)

Si por ejemplo:
\(
  W^{(enc)} = \begin{pmatrix} 0.5 & 0.2 \end{pmatrix}, \quad b^{(enc)} = 0.0,
\)
\(
  x = \begin{pmatrix} 1.0 \\ 2.0 \end{pmatrix}, \quad h = 0.5 * 1.0 + 0.2 * 2.0 = 0.9,
\)
\(
  h = \sigma(0.9) \approx 0.71 \quad (\text{si usaramos sigmoid}).
\)

Luego el decoder multiplica ese valor por \(W^{(dec)}\) para aproximar \(x\). La red se entrena para que \(\hat{x}\approx x\).

### 5.7. Preparación de Datos e Inputs
- **Normalización:** Se suele escalar cada dimensión para que el encoder maneje valores en rangos similares.
- **Dimensionalidad:** Si se trabaja con imágenes, se vectorizan o se aplican arquitecturas convolucionales (Conv Autoencoder).
- **Ruido en Denoising Autoencoder:** Se inyecta ruido gaussiano o de "dropout" a la entrada.
- **Batching:** Como en otras redes, se agrupan ejemplos en lotes.

### 5.8. Ejemplo de Creación de la Red y Entrenamiento en Python
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Datos 1000 muestras, 3 features
X_train = np.random.rand(1000, 3)

# Autoencoder con bottleneck en 2D
model = Sequential()
# Encoder
model.add(Dense(2, activation='relu', input_shape=(3,)))
# Decoder
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=5, batch_size=32)
```
1. Se reconstruye la misma entrada.
2. La capa intermedia de dimensión 2 actúa como "codificación".
3. Entrenamiento con MSE durante 5 épocas.

---

## CUADRO 6: Redes Generativas Adversariales (GANs)

### 6.1. Definición
Las GANs constan de dos modelos en competencia:
1. **Generador (G):** produce datos ficticios que intentan parecerse a los datos reales.
2. **Discriminador (D):** aprende a distinguir datos reales de los generados.

El juego "minimax" consiste en que G trata de engañar a D, mientras D se vuelve más estricto.

### 6.2. Funcionamiento
- El **Generador** toma un ruido aleatorio y genera muestras.
- El **Discriminador** recibe ejemplos (reales y generados) y predice si son reales o falsas.
- Ambos se entrenan simultáneamente, adaptando sus pesos para optimizar objetivos opuestos.

### 6.3. Aplicaciones
- **Generación de Imágenes:** producir rostros de personas, paisajes, etc.
- **Superresolución:** aumentar la calidad de imágenes de baja resolución.
- **Transferencia de estilo:** convertir una imagen a la "estética" de otra.
- **Generación de Música y Voz:** sintetizar voces naturales.

### 6.4. Fortalezas y Limitaciones
- **Ventajas:**
  - Generan resultados realistas sin requerir modelos probabilísticos explícitos.
  - Creatividad en imágenes, audio y más.
- **Desventajas:**
  - Entrenamiento inestable, delicado de ajustar.
  - Puede ocurrir el problema de "mode collapse" (generar muestras similares).

### 6.5. Definición Matemática y Ejemplo Paso a Paso

Sea \(z\) una variable aleatoria (ruido) con distribución \(p_z\). El generador \(G\) aprende una función \(G(z;\theta_g)\). El discriminador \(D\) es una red que produce una probabilidad de que la muestra venga de los datos reales (en vez de ser generada): \(D(x;\theta_d)\).

La función de costo para el discriminador es:
\[
  \max_{\theta_d} \mathbb{E}_{x \sim p_{data}} [\log(D(x))] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)) )].
\]

El generador quiere minimizar:
\[
  \min_{\theta_g} \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].
\]

**Ejemplo:**
- \(z\) es un escalar aleatorio uniformemente distribuido en [-1,1].
- \(G(z)\) es una red MLP con salida 1D.
- \(D(x)\) es otra MLP que sale en [0,1].

En cada iteración, se:
1. Toma un minibatch de muestras reales de \(p_{data}\). Se entrena \(D\) para que asigne alta probabilidad a estas.
2. Toma un minibatch de \(z\) y genera \(G(z)\). Se entrena \(D\) para asignar baja probabilidad a las falsas.
3. Actualiza \(G\) para engañar mejor a \(D\).

### 6.6. Preparación de Datos e Inputs
- **Datos Reales:** Normalmente se normalizan o escalan (en imágenes, entre [-1,1] o [0,1], por ejemplo).
- **Ruido (Input de Generador):** Se muestrea de distribuciones uniformes, gaussianas, etc.
- **Estructura de Tensores:** Para imágenes, se usan tensores \((N, C, H, W)\) para el discriminador.
- **Balance en Batches:** Suele mezclarse un batch de reales con uno de generados en cada iteración.

### 6.7. Ejemplo de Creación de la Red y Entrenamiento en Python

**Ejemplo simplificado (usando Keras):**
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generador
def build_generator(latent_dim=1):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(latent_dim,)))
    model.add(Dense(1, activation='linear'))  # salida 1D
    return model

# Discriminador
def build_discriminator():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(1,)))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Datos reales: y = x + ruido
X_real = np.random.uniform(-1, 1, (1000, 1))
y_real = X_real + np.random.normal(0, 0.05, (1000, 1))

latent_dim = 1
G = build_generator(latent_dim)
D = build_discriminator()
D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN combinada
z = tf.keras.Input(shape=(latent_dim,))
img = G(z)
D.trainable = False
valid = D(img)
combined = tf.keras.Model(z, valid)
combined.compile(optimizer='adam', loss='binary_crossentropy')

# Entrenamiento simplificado
batch_size = 32
for epoch in range(10):
    # 1) Entrenar D con datos reales
    idx = np.random.randint(0, X_real.shape[0], batch_size)
    real_samples = y_real[idx]
    D_loss_real = D.train_on_batch(real_samples, np.ones((batch_size, 1)))

    # 2) Entrenar D con datos generados
    noise = np.random.uniform(-1, 1, (batch_size, latent_dim))
    fake_samples = G.predict(noise)
    D_loss_fake = D.train_on_batch(fake_samples, np.zeros((batch_size, 1)))

    # 3) Entrenar G (para engañar a D)
    noise = np.random.uniform(-1, 1, (batch_size, latent_dim))
    G_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 2 == 0:
        print(f"Epoch {epoch}: D_loss_real={D_loss_real}, D_loss_fake={D_loss_fake}, G_loss={G_loss}")
```
1. Se definen dos redes (Generador y Discriminador).
2. Se entrena el Discriminador con muestras reales y falsas, luego se entrena el Generador para engañar al Discriminador.
3. El ejemplo es muy simplificado, no representa un caso típico de imágenes, pero ilustra el flujo.

---

## CUADRO 7: Redes Basadas en Atención y Transformers

### 7.1. Definición
La llegada de los **Transformers** (en el artículo "Attention is all you need" - 2017) revolucionó el procesamiento secuencial y otras áreas, al usar mecanismos de **atención** para modelar relaciones a larga distancia sin recurrencias.

### 7.2. Mecanismo de Auto-Atención
Permite que cada "token" o elemento en la secuencia preste atención a cada otro, calculando pesos de relevancia. Así, se capturan dependencias distantes de manera más efectiva que con RNN.

### 7.3. Arquitectura
1. **Encoder:** varias capas de auto-atención y redes feedforward.
2. **Decoder:** similar al encoder, pero con atención "enmascarada" para predecir secuencias paso a paso.
3. **Positional Encoding:** agrega información sobre la posición de cada token.

### 7.4. Aplicaciones
- **NLP:**
  - Modelos de lenguaje (GPT, BERT, T5).
  - Traducción y resumen automático.
- **Vision Transformers (ViT):** clasificación de imágenes, detección.
- **AlphaFold:** Plegamiento de proteínas.

### 7.5. Fortalezas y Limitaciones
- **Ventajas:**
  - Gran capacidad para capturar dependencias a largo plazo.
  - Procesamiento paralelo de tokens.
- **Desventajas:**
  - Complejidad O(n^2) en la longitud de la secuencia.
  - Requiere mucha memoria y potencia de cómputo.

### 7.6. Definición Matemática y Ejemplo Paso a Paso

En el mecanismo de auto-atención, se definen para cada token tres vectores: \(Q\) (query), \(K\) (key) y \(V\) (value). Si la matriz de embeddings de una capa es \(X\), entonces:
\[
  Q = X W^Q, \quad K = X W^K, \quad V = X W^V.
\]

La atención se calcula como:
\[
  \text{Attention}(Q,K,V) = \text{softmax}\Bigl(\frac{Q K^T}{\sqrt{d_k}}\Bigr) V,
\]
donde \(d_k\) es la dimensión de los vectores key.

**Ejemplo Simplificado**:
- Supongamos 2 tokens, cada uno de dimensión 2: \(X = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}\) con \(x_1,x_2 \in \mathbb{R}^2\).
- \(W^Q, W^K, W^V\) son matrices 2x2.

Se calcula \(Q, K, V\) y luego la matriz de atención 2x2, con la cual se ponderan los values. El resultado final se usa como input a la siguiente capa.

### 7.7. Preparación de Datos e Inputs
- **Tokenización de Texto:** Separar el texto en tokens y transformarlos en vectores (por ejemplo, con embeddings).
- **Positional Encoding:** Se añade un vector de posición (sinusoidal o trainable) para cada token.
- **Máscaras:** En problemas de traducción o secuencias, se enmascaran tokens futuros o posiciones inexistentes.
- **Batching y Padding:** Igual que en RNN, se unifican longitudes y se aplican máscaras donde no hay contenido.
- **Vision Transformers:** Dividen la imagen en parches, cada uno se vectoriza y se añade un embedding posicional.

### 7.8. Ejemplo de Creación de la Red y Entrenamiento en Python

**Ejemplo con la API de Keras para un Transformer simple (versión simplificada):**
```python
import tensorflow as tf
from tensorflow.keras import layers

# Ejemplo simplificado: supongamos secuencias de 10 tokens, vocabulario de 1000.
max_len = 10
vocab_size = 1000
embed_dim = 32  # dimensión de embedding

inputs = tf.keras.Input(shape=(max_len,))  # asumiendo entradas tokenizadas

# Embedding
x = layers.Embedding(vocab_size, embed_dim)(inputs)

# Pequeña capa de atención (Self-Attention)
attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x, x)
x = x + attn_output  # Residual
x = layers.LayerNormalization()(x)

# Feedforward
ff = layers.Dense(embed_dim*2, activation='relu')(x)
ff = layers.Dense(embed_dim)(ff)
x = x + ff
x = layers.LayerNormalization()(x)

# Salida final (por ejemplo, para clasificación de 5 clases)
x = layers.Flatten()(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Datos sintéticos
import numpy as np
X_train = np.random.randint(0, vocab_size, size=(1000, max_len))
y_train = np.random.randint(0, 5, size=(1000,))

# One-hot
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=5)

model.fit(X_train, y_train_onehot, epochs=3, batch_size=32)
```
1. Definimos un modelo con embedding, MultiHeadAttention y una pequeña red feedforward (Transformer block simplificado).
2. Se generan datos sintéticos (1000 secuencias, cada una de longitud 10).
3. Se entrena por 3 épocas.

---

## CUADRO 8: Otras Redes Neuronales Especializadas

### 8.1. Redes de Funciones de Base Radial (RBF)
- **Uso de funciones de base radial** (típicamente Gaussiana) como activación.
- Se pueden interpretar como redes con una capa de características en medio.
- **Aplicaciones:** aproximación de funciones, problemas de control.

**Definición Matemática (Breve):**
\(
  \phi_i(x) = \exp\bigl(-\gamma \|x - c_i\|^2\bigr),
\)
donde \(c_i\) es el centro de la base radial y \(\gamma\) un parámetro de ancho.

### 8.2. Mapas Auto-Organizados (SOM)
- **No supervisadas:** buscan proyección en 2D preservando la topología.
- **Aplicaciones:** análisis exploratorio de datos, clustering visual.

**Definición (Breve):** Ajustan pesos de un mapa 2D para que cada "neurona" represente una región de las características de entrada, preservando la vecindad.

### 8.3. Máquinas de Boltzmann Restringidas (RBM) y Redes de Creencias Profundas (DBN)
- **RBM:** modelo energético que aprende representaciones de forma probabilística.
- **DBN:** pila de RBMs usada históricamente en preentrenamiento de redes profundas.
- **Aplicaciones:** detección de características, reducción de dimensionalidad, recomendación.

**Definición (Breve):** En una RBM, la energía de una configuración \((v,h)\) es:
\(
  E(v,h) = -v^T W h - b^T v - c^T h.
\)

### 8.4. Redes Neuronales de Grafos (GNN)
- Diseñadas para datos con estructura de grafo (nodos, aristas).
- **Aplicaciones:** análisis de redes sociales, química computacional, sistemas de recomendación y ruteo.

**Idea Principal:** Cada nodo aprende una representación a partir de sus vecinos y las conexiones iterativamente.

### 8.5. Redes Neuronales de Picos (Spiking Neural Networks, SNN)
- Inspiradas en el cerebro biológico, usan "spikes" (eventos discretos) en lugar de valores continuos.
- **Aplicaciones:** computación neuromórfica, sistemas de bajo consumo.

### 8.6. Capsule Networks
- Propuestas por Geoffrey Hinton para superar limitaciones de CNN al reconocer poses y composiciones.
- Una "cápsula" agrupa neuronas y modela una entidad y su relación espacial.

### 8.7. Preparación de Datos e Inputs en Redes Especializadas
- **RBF:** Habitualmente se normaliza el espacio de entrada para facilitar la definición de \(c_i\) y \(\gamma\).
- **SOM:** Se vectoriza cada dato de entrada y se "presenta" al mapa; importante escalar/normalizar para que la distancia sea representativa.
- **RBM/DBN:** Para datos binarios (0/1) o escalados a [0,1], se alimentan como vectores. En caso de imágenes, se aplanan a 1D.
- **GNN:** Se define un grafo (nodos, aristas) y se crean vectores iniciales para cada nodo. La entrada se pasa a capas que hacen "propagación de mensajes".
- **SNN:** Se codifican estímulos continuos en secuencias de "spikes"; requiere un preprocesamiento específico (ej. codificación rate-based o temporal).
- **Capsule Networks:** Para imágenes, se suelen usar pequeñas "patches" como inputs iniciales a las cápsulas.

### 8.8. Ejemplo de Creación y Entrenamiento en Python

Dado que estas arquitecturas especializadas suelen requerir librerías y técnicas específicas, aquí va un ejemplo muy básico (no oficial) de una RBM en Python (pseudo-implementación):
```python
import numpy as np

class RBM:
    def __init__(self, n_visible=6, n_hidden=3, lr=0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        # Inicializar pesos
        self.W = np.random.normal(0, 0.1, size=(n_visible, n_hidden))
        self.b = np.zeros(n_visible)  # bias visibles
        self.c = np.zeros(n_hidden)   # bias ocultos

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sample_hidden(self, v):
        # Prob ocultas
        h_prob = self.sigmoid(np.dot(v, self.W) + self.c)
        return (np.random.rand(*h_prob.shape) < h_prob).astype(np.float32)

    def sample_visible(self, h):
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.b)
        return (np.random.rand(*v_prob.shape) < v_prob).astype(np.float32)

    def contrastive_divergence(self, v):
        h0 = self.sample_hidden(v)
        v1 = self.sample_visible(h0)
        h1 = self.sample_hidden(v1)
        # Actualizar pesos
        # v*h.T - v1*h1.T (promedio)
        dW = np.dot(v.reshape(-1,1), h0.reshape(1,-1)) - np.dot(v1.reshape(-1,1), h1.reshape(1,-1))
        self.W += self.lr * dW
        self.b += self.lr * (v - v1)
        self.c += self.lr * (h0 - h1)

# Ejemplo de uso
rbm = RBM(n_visible=6, n_hidden=3, lr=0.1)
X_train = np.random.randint(0,2,size=(100,6))  # 100 muestras binarias de 6 bits

# Entrenamiento simple
for epoch in range(5):
    for x in X_train:
        rbm.contrastive_divergence(x)
```
Es un ejemplo mínimo, no optimizado ni verificado, que muestra la idea de Contrastive Divergence para entrenar una RBM.

---

# Conclusión del Lienzo

Este **lienzo extenso** abarca las arquitecturas de redes neuronales más influyentes hasta la fecha. Aunque existen muchas otras variantes y se proponen nuevas ideas constantemente, las descritas aquí forman los pilares fundamentales del **aprendizaje profundo** y sirven de guía para elegir la mejor herramienta en función del problema:

1. **MLP/FNN:** si no hay estructura espacial o secuencial clara.
2. **CNN:** tareas de visión o señales estructuradas.
3. **RNN/LSTM/GRU:** secuencias y dependencias temporales.
4. **Transformers:** secuencias largas o contextos complejos en NLP y visión.
5. **Autoencoders:** aprendizaje no supervisado, compresión, generación.
6. **GANs:** generación de datos, aumento de datos.
7. **Otras especializadas (SOM, GNN, etc.):** dominios específicos con estructuras únicas.

La **combinación** de diferentes arquitecturas y la **creatividad** en su uso sigue impulsando el campo de la inteligencia artificial hacia nuevas fronteras.

