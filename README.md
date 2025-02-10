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

