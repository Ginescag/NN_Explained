# NN_Explained

In this repository, i've made a compilation of the most used NN models. You have access to a brief explanation of a bunch of these neural network models as well as some Python code that you can run to train and test them all, enjoy!

# Extensive Conceptual Canvas: Neural Network Architectures and Their Applications

Imagine a large canvas (or mural) divided into different sections (panels). Each section is dedicated to exploring a type of neural network, delving into its operational principles, components, advantages, disadvantages, **applications**, **mathematical definitions**, **step-by-step examples**, **how to prepare the data**, and now also **basic Python implementation examples** (using Keras, part of TensorFlow).

---

## PANEL 1: Feedforward Neural Networks (FNN) or MLP

### 1.1. Definition
Feedforward Neural Networks (FNN), also known as **Multilayer Perceptron (MLP)**, are the most classic and basic form of neural networks. In these networks, information flows in a single direction: from the input layer, through one or more hidden layers, to the output layer.

### 1.2. Architecture
1. **Input Layer:** Receives the input data (features). For example, in a digit classification problem, each pixel might be a neuron in the input layer.
2. **Hidden Layers:** One or more layers that perform intermediate transformations on the data. Each neuron in these layers computes a weighted sum of its inputs, followed by an activation function (sigmoid, tanh, ReLU, etc.).
3. **Output Layer:** Produces the final prediction. For instance, in a binary classification problem, it could be a single neuron with a sigmoid activation, while in a multi-class classification, a softmax is used.

### 1.3. Training
Training is commonly done via **backpropagation** using an optimizer such as Gradient Descent, Adam, or others. The idea is:
1. Compute the network's output for a given input.
2. Measure the error based on the desired output.
3. Propagate this error backward, adjusting the weights to minimize the loss.

### 1.4. Applications
- **Classification:** Spam detection, simple image classification, general pattern recognition.
- **Regression:** Price prediction (stocks, real estate), performance estimations.
- **Recommendation Systems:** Combined with other techniques.

### 1.5. Strengths and Limitations
- **Advantages:**
  - Simple structure and easy to implement.
  - Useful as "base networks" for many problems.
- **Disadvantages:**
  - Do not scale well when the problem requires detecting complex structures (e.g., images, sequences).
  - May need many layers for more advanced problems, increasing the risk of overfitting and computational cost.

### 1.7. Data Preparation and Inputs
- **Normalization or Standardization:** It’s common to scale each feature to have mean 0 and standard deviation 1 (or min 0 and max 1). This speeds up convergence.
- **Input Formatting:** For MLPs, data is typically in 1D vectors or tensors (if there are more dimensions, they’re often “flattened”).
- **Handling Missing Values:** Important to impute or discard null values to avoid inconsistencies in training.
- **Categorical Variable Encoding:** One-hot, label encoding, or other techniques if there are non-numerical attributes.

### 1.8. Example of Network Creation and Training in Python

**Example with Keras (TensorFlow):**
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Synthetic example data
X_train = np.random.rand(1000, 2)  # 1000 samples, 2 features
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)  # Binary label

# Define model
model = Sequential()
model.add(Dense(2, input_shape=(2,), activation='relu'))  # Hidden layer with 2 neurons
model.add(Dense(1, activation='sigmoid'))                 # Output layer (binary)

# Compile
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)
```
In this example:
1. Random synthetic data with 2 features. The label is 1 if the sum of the features is greater than 1.
2. A one-hidden-layer MLP is defined.
3. Trained for 10 epochs with batches of 32.

---

## PANEL 2: Convolutional Neural Networks (CNN)

### 2.1. Definition
CNNs (Convolutional Neural Networks) were initially designed for image analysis, although they also apply to audio, text (transforming into "embeddings images"), and other grid-structured data.

### 2.2. Main Elements
1. **Convolutional Layers:** "Filters" or kernels scan the image or data matrix to extract local patterns (edges, contours, textures). Each filter produces a "feature map."
2. **Pooling Layers:** Reduce spatial dimensions (e.g., max pooling), helping achieve translation invariance.
3. **Fully Connected Layers:** Often included at the end of the CNN for classification.

### 2.3. Training
Also based on backpropagation, but adapted for convolutional operations. GPU usage is particularly beneficial due to the large amount of data and images.

### 2.4. Applications
- **Computer Vision:**
  - Image classification (e.g., recognize animals, objects, faces).
  - Object detection (e.g., YOLO, Faster R-CNN).
  - Image segmentation (e.g., U-Net).
- **Audio Analysis:** Recognize patterns in spectrograms.
- **Text Processing in 2D:** Embeddings arranged as 2D maps.

### 2.5. Strengths and Limitations
- **Advantages:**
  - Efficiently capture spatial patterns.
  - Fewer parameters than an equivalent fully connected network.
- **Disadvantages:**
  - Primarily model local information, may require extra tricks to capture global relationships.
  - Still computationally expensive.

### 2.7. Data Preparation and Inputs
- **Image Formatting:** Tensors usually of shape \((N, C, H, W)\), where \(N\) is the number of examples, \(C\) is the number of channels, and \(H\) and \(W\) are height and width.
- **Per-Channel Normalization:** It’s common to subtract the mean and divide by the standard deviation (calculated on the training set) for each channel.
- **Data Augmentation (Images):** Rotations, flips, color changes to increase data diversity and avoid overfitting.
- **For Audio:** Often converted to spectrograms (2D) and treated like "images."
- **For Text (in 2D):** Use embeddings as matrices if we want to apply convolutions.

### 2.8. Example of Network Creation and Training in Python

**Example with Keras (TensorFlow) for Image Classification (simplified):**
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Suppose we have 1000 images of 28x28 in grayscale (1 channel)
X_train = np.random.rand(1000, 28, 28, 1).astype(np.float32)
# Labels for 10 classes (e.g., digits 0-9)
y_train = np.random.randint(0, 10, size=(1000,))

# Convert labels to one-hot
y_train_onehot = np.zeros((1000, 10))
for i, label in enumerate(y_train):
    y_train_onehot[i, label] = 1

model = Sequential()
model.add(Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))  # output for 10 classes

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_onehot, epochs=5, batch_size=32)
```
1. Synthetic data is generated (28x28, 1 channel, 10 classes).
2. We define a very simple CNN with one conv layer, pooling, flatten, and a final dense layer.
3. Trained for 5 epochs.

---

## PANEL 3: Recurrent Neural Networks (RNN)

### 3.1. Definition
RNNs (Recurrent Neural Networks) are used for sequential data (text, audio, time series), enabling the network to "remember" information from previous steps via recurrent connections in its architecture.

### 3.2. Architecture
1. **Hidden State:** Acts as "memory," updated at each step in the sequence.
2. **Sequential Input:** Each new element in the sequence (word, temporal vector) is processed along with the previous hidden state.
3. **Sequential or Hidden Output:** Can produce an output at each step (many-to-many) or only after processing the entire sequence (many-to-one).

### 3.3. Training: Backpropagation Through Time (BPTT)
The network is "unrolled" in time, which can be more computationally expensive and cause vanishing or exploding gradient problems.

### 3.4. Applications
- **NLP (Natural Language Processing):** Sentiment analysis, sentence classification, language modeling.
- **Machine Translation:** One RNN can encode the input sentence and another RNN decode into the target language.
- **Time Series:** Prediction of financial data, sensor data, weather patterns.

### 3.5. Strengths and Limitations
- **Advantages:**
  - Naturally handle sequential data.
  - Capture temporal dependencies.
- **Disadvantages:**
  - Difficulty in learning long-term dependencies.
  - Slow training on long sequences.

### 3.7. Data Preparation and Inputs
- **Sequence Representation:** For text, typically tokenize each sentence and convert to indices or embeddings. Sequences may be padded to a fixed length.
- **Time Series Normalization:** Often subtract mean and divide by the standard deviation of each temporal feature.
- **Sequence Batching:** Group sequences of similar length or use truncation/padding for uniform lengths.
- **Masking:** When sequences vary in length, use a "mask" to ignore unnecessary parts.

### 3.8. Example of Network Creation and Training in Python

**Example with Keras (TensorFlow) for simple sequences:**
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Example: sequences of length 5, with 1 feature
# Generate 1000 random sequences
X_train = np.random.rand(1000, 5, 1)
# Binary label: 1 if the average of the sequence > 0.5
y_train = (X_train.mean(axis=1).flatten() > 0.5).astype(int)

model = Sequential()
model.add(SimpleRNN(4, input_shape=(5, 1), activation='tanh'))  # Hidden state dimension of 4
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)
```
1. Create sequences of 5 steps, each with 1 feature.
2. The RNN (SimpleRNN) produces a single output vector at the end, passed to a binary dense layer.
3. Train for 5 epochs.

---

## PANEL 4: LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit)

### 4.1. LSTM
LSTMs largely solve the vanishing gradient problem using an internal "cell" and **three gates**:
1. **Forget Gate:** Decides which part of the previous state to keep.
2. **Input Gate:** Determines how much new information to add.
3. **Output Gate:** Regulates how much of the cell state to expose.

**Applications:**
- Machine translation.
- Language modeling.
- Speech recognition.

### 4.2. GRU
GRUs simplify the LSTM architecture, reducing it to two gates (update and reset) while maintaining similar performance in many cases.

**Applications:**
- Chatbots.
- Time series analysis (often shorter and more efficient than LSTM).

### 4.3. Strengths and Limitations
- **Advantages:**
  - LSTM/GRU can retain information over longer sequences than a simple RNN.
  - Fewer vanishing gradient issues.
- **Disadvantages:**
  - More parameters than a basic RNN.
  - Longer training times, especially for very long sequences.

### 4.6. Example of Network Creation and Training in Python
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# Example: sequences of length 10, 1 feature
X_train = np.random.rand(500, 10, 1)
y_train = (X_train.mean(axis=1).flatten() > 0.5).astype(int)

model = Sequential()
# We can use LSTM or GRU. Here we use LSTM:
model.add(LSTM(8, input_shape=(10, 1)))  # 8 LSTM neurons
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)
```
1. Generates 500 sequences of length 10.
2. An LSTM with 8 neurons.
3. Trained for 5 epochs.

---

## PANEL 5: Autoencoders

### 5.1. Definition
An **Autoencoder** is a network designed to learn compressed (encoded) representations of the data in an unsupervised manner. The primary goal is to reconstruct the input at the output, forcing the network to compress the information.

### 5.2. Architecture
1. **Encoder:** Transforms the input data into a lower-dimensional representation (bottleneck).
2. **Latent Space:** Intermediate layer of reduced dimension.
3. **Decoder:** Attempts to reconstruct the original input from the latent representation.

### 5.3. Variants
- **Denoising Autoencoder:** Trained on noisy inputs so it learns to recover the clean signal.
- **Variational Autoencoders (VAE):** Address data generation from a probabilistic perspective.

### 5.4. Applications
- **Dimensionality Reduction:** Non-linear alternative to PCA.
- **Anomaly Detection:** If something is very different from the training data, it will be poorly reconstructed.
- **Data Generation:** In VAEs, can generate realistic samples resembling the original data.

### 5.5. Strengths and Limitations
- **Advantages:**
  - Learn without needing labels.
  - Useful for understanding the data structure.
- **Disadvantages:**
  - May tend to memorize if capacity is high.
  - More complex than linear methods like PCA.

### 5.7. Data Preparation and Inputs
- **Normalization:** Often scale each dimension so the encoder handles values in similar ranges.
- **Dimensionality:** For images, they can be flattened or we can apply convolutional architectures (Conv Autoencoder).
- **Noise in Denoising Autoencoder:** Gaussian or "dropout" noise can be injected into the input.
- **Batching:** As with other networks, group examples in batches.

### 5.8. Example of Network Creation and Training in Python
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data: 1000 samples, 3 features
X_train = np.random.rand(1000, 3)

# Autoencoder with a 2D bottleneck
model = Sequential()
# Encoder
model.add(Dense(2, activation='relu', input_shape=(3,)))
# Decoder
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=5, batch_size=32)
```
1. Reconstruct the same input.
2. The intermediate layer of dimension 2 acts as the "encoding."
3. Trained with MSE for 5 epochs.

---

## PANEL 6: Generative Adversarial Networks (GANs)

### 6.1. Definition
GANs consist of two competing models:
1. **Generator (G):** Produces fake data that tries to resemble real data.
2. **Discriminator (D):** Learns to distinguish real data from generated data.

The "minimax" game is such that G tries to fool D, while D becomes more stringent.

### 6.2. How They Work
- The **Generator** takes random noise and generates samples.
- The **Discriminator** receives examples (both real and generated) and predicts whether they are real or fake.
- Both are trained simultaneously, adjusting their weights to optimize opposing objectives.

### 6.3. Applications
- **Image Generation:** Producing faces of people, landscapes, etc.
- **Super-Resolution:** Enhancing low-resolution images.
- **Style Transfer:** Converting an image to the "style" of another.
- **Music and Voice Generation:** Synthesizing natural-sounding voices.

### 6.4. Strengths and Limitations
- **Advantages:**
  - Generate realistic results without requiring explicit probabilistic models.
  - Creativity in images, audio, and more.
- **Disadvantages:**
  - Unstable training, delicate to tune.
  - "Mode collapse" may occur (generating similar samples).

### 6.6. Data Preparation and Inputs
- **Real Data:** Usually normalized or scaled (for images, in [-1,1] or [0,1], for example).
- **Noise (Generator Input):** Sampled from uniform, Gaussian, or other distributions.
- **Tensor Structure:** For images, typically \((N, C, H, W)\) for the discriminator.
- **Batch Balancing:** Usually mixing a batch of real data with generated data in each iteration.

### 6.7. Example of Network Creation and Training in Python

**Simplified example (using Keras):**
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generator
def build_generator(latent_dim=1):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(latent_dim,)))
    model.add(Dense(1, activation='linear'))  # 1D output
    return model

# Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(1,)))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Real data: y = x + noise
X_real = np.random.uniform(-1, 1, (1000, 1))
y_real = X_real + np.random.normal(0, 0.05, (1000, 1))

latent_dim = 1
G = build_generator(latent_dim)
D = build_discriminator()
D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Combined GAN
z = tf.keras.Input(shape=(latent_dim,))
img = G(z)
D.trainable = False
valid = D(img)
combined = tf.keras.Model(z, valid)
combined.compile(optimizer='adam', loss='binary_crossentropy')

# Simplified training
batch_size = 32
for epoch in range(10):
    # 1) Train D with real data
    idx = np.random.randint(0, X_real.shape[0], batch_size)
    real_samples = y_real[idx]
    D_loss_real = D.train_on_batch(real_samples, np.ones((batch_size, 1)))

    # 2) Train D with generated data
    noise = np.random.uniform(-1, 1, (batch_size, latent_dim))
    fake_samples = G.predict(noise)
    D_loss_fake = D.train_on_batch(fake_samples, np.zeros((batch_size, 1)))

    # 3) Train G (to fool D)
    noise = np.random.uniform(-1, 1, (batch_size, latent_dim))
    G_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 2 == 0:
        print(f"Epoch {epoch}: D_loss_real={D_loss_real}, D_loss_fake={D_loss_fake}, G_loss={G_loss}")
```
1. Two networks are defined (Generator and Discriminator).
2. The Discriminator is trained on real and fake samples, then the Generator is trained to fool the Discriminator.
3. This example is highly simplified and not typical for images, but illustrates the workflow.

---

## PANEL 7: Attention-Based Networks and Transformers

### 7.1. Definition
The advent of **Transformers** (in the 2017 paper "Attention is all you need") revolutionized sequential data processing and more, by using **attention** mechanisms to model long-range dependencies without recurrence.

### 7.2. Self-Attention Mechanism
Allows each "token" or sequence element to attend to every other, computing relevance weights. Thus, distant dependencies are captured more effectively than with RNNs.

### 7.3. Architecture
1. **Encoder:** Multiple layers of self-attention and feedforward networks.
2. **Decoder:** Similar to the encoder, but with "masked" attention to predict sequences step by step.
3. **Positional Encoding:** Adds information about the position of each token.

### 7.4. Applications
- **NLP:**
  - Language models (GPT, BERT, T5).
  - Automatic translation and summarization.
- **Vision Transformers (ViT):** Image classification, detection.
- **AlphaFold:** Protein folding.

### 7.5. Strengths and Limitations
- **Advantages:**
  - Great capacity to capture long-distance dependencies.
  - Parallel token processing.
- **Disadvantages:**
  - Complexity O(n^2) in sequence length.
  - Requires large memory and computing power.

### 7.7. Data Preparation and Inputs
- **Text Tokenization:** Split text into tokens and transform them into vectors (e.g., embeddings).
- **Positional Encoding:** A sinusoidal or trainable position vector is added for each token.
- **Masking:** For translation or sequences, mask future tokens or non-existent positions.
- **Batching and Padding:** As with RNNs, unify sequence lengths and apply masks where no content exists.
- **Vision Transformers:** Divide the image into patches, each patch is vectorized, and a positional embedding is added.

### 7.8. Example of Network Creation and Training in Python

**Example with the Keras API for a simple Transformer (simplified version):**
```python
import tensorflow as tf
from tensorflow.keras import layers

# Simplified example: suppose sequences of 10 tokens, vocab of 1000.
max_len = 10
vocab_size = 1000
embed_dim = 32  # embedding dimension

inputs = tf.keras.Input(shape=(max_len,))  # assuming tokenized inputs

# Embedding
x = layers.Embedding(vocab_size, embed_dim)(inputs)

# Small attention layer (Self-Attention)
attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim)(x, x)
x = x + attn_output  # Residual
x = layers.LayerNormalization()(x)

# Feedforward
ff = layers.Dense(embed_dim*2, activation='relu')(x)
ff = layers.Dense(embed_dim)(ff)
x = x + ff
x = layers.LayerNormalization()(x)

# Final output (e.g., for classification into 5 classes)
x = layers.Flatten()(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Synthetic data
import numpy as np
X_train = np.random.randint(0, vocab_size, size=(1000, max_len))
y_train = np.random.randint(0, 5, size=(1000,))

# One-hot
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=5)

model.fit(X_train, y_train_onehot, epochs=3, batch_size=32)
```
1. A model is defined with embedding, MultiHeadAttention, and a small feedforward (simplified Transformer block).
2. Synthetic data is generated (1000 sequences, each of length 10).
3. Trained for 3 epochs.

---

## PANEL 8: Other Specialized Neural Networks

### 8.1. Radial Basis Function Networks (RBF)
- **Use of radial basis functions** (often Gaussian) as activation.
- Can be interpreted as networks with a hidden feature layer.
- **Applications:** function approximation, control problems.

### 8.4. Graph Neural Networks (GNN)
- Designed for data with graph structure (nodes, edges).
- **Applications:** social network analysis, computational chemistry, recommendation systems, routing.

**Main Idea:** Each node learns a representation from its neighbors and connections iteratively.

### 8.5. Spiking Neural Networks (SNN)
- Inspired by biological brains, using "spikes" (discrete events) instead of continuous values.
- **Applications:** neuromorphic computing, low-power systems.

### 8.6. Capsule Networks
- Proposed by Geoffrey Hinton to overcome some CNN limitations in recognizing poses and compositions.
- A "capsule" groups neurons and models an entity and its spatial relationship.

### 8.7. Data Preparation and Inputs in Specialized Networks
- **RBF:** Typically normalize the input space to facilitate definition of \(c_i\) and \(\gamma\).
- **SOM:** Each input is vectorized and presented to the map; important to scale/normalize so distance is meaningful.
- **RBM/DBN:** For binary (0/1) or scaled [0,1] data, input as vectors. In the case of images, flatten to 1D.
- **GNN:** Define a graph (nodes, edges) and create initial node vectors. Data is passed to layers performing “message passing.”
- **SNN:** Convert continuous stimuli into spike sequences; requires specific preprocessing (e.g., rate-based or temporal coding).
- **Capsule Networks:** For images, typically use small patches as initial inputs to the capsules.

### 8.8. Example of Creation and Training in Python

Because these specialized architectures often require specific libraries and techniques, here’s a very basic (non-official) example of an RBM in Python (pseudo-implementation):
```python
import numpy as np

class RBM:
    def __init__(self, n_visible=6, n_hidden=3, lr=0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        # Initialize weights
        self.W = np.random.normal(0, 0.1, size=(n_visible, n_hidden))
        self.b = np.zeros(n_visible)  # visible biases
        self.c = np.zeros(n_hidden)   # hidden biases

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sample_hidden(self, v):
        # Hidden probabilities
        h_prob = self.sigmoid(np.dot(v, self.W) + self.c)
        return (np.random.rand(*h_prob.shape) < h_prob).astype(np.float32)

    def sample_visible(self, h):
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.b)
        return (np.random.rand(*v_prob.shape) < v_prob).astype(np.float32)

    def contrastive_divergence(self, v):
        h0 = self.sample_hidden(v)
        v1 = self.sample_visible(h0)
        h1 = self.sample_hidden(v1)
        # Update weights
        # v*h.T - v1*h1.T (average)
        dW = np.dot(v.reshape(-1,1), h0.reshape(1,-1)) - np.dot(v1.reshape(-1,1), h1.reshape(1,-1))
        self.W += self.lr * dW
        self.b += self.lr * (v - v1)
        self.c += self.lr * (h0 - h1)

# Example usage
rbm = RBM(n_visible=6, n_hidden=3, lr=0.1)
X_train = np.random.randint(0,2,size=(100,6))  # 100 binary samples of 6 bits

# Simple training
for epoch in range(5):
    for x in X_train:
        rbm.contrastive_divergence(x)
```
This is a minimal, unoptimized, unverified example showing the idea of Contrastive Divergence for training an RBM.

---

# Conclusion

This **extensive canvas** covers the most influential neural network architectures to date. Although many other variations exist and new ideas are constantly proposed, those described here form the fundamental pillars of **deep learning** and serve as a guide to choose the best tool depending on the problem:

1. **MLP/FNN:** if there is no clear spatial or sequential structure.
2. **CNN:** for vision tasks or structured signals.
3. **RNN/LSTM/GRU:** for sequences and temporal dependencies.
4. **Transformers:** for long sequences or complex contexts in NLP and vision.
5. **Autoencoders:** unsupervised learning, compression, generation.
6. **GANs:** data generation, data augmentation.
7. **Others (SOM, GNN, etc.):** specialized domains with unique structures.

The **combination** of different architectures and **creativity** in their use continues to propel the field of artificial intelligence to new frontiers.


