import matplotlib.pyplot as plt
from keras.api.utils import image_dataset_from_directory
from keras.api.applications import VGG16
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Dropout, Rescaling
from keras.api.optimizers import Adam
from keras.api import Input
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pickle

# Caminhos do dataset
dataset = './wave_plots'

# Parâmetros
batch_size = 32  # Ajustado para pequenos datasets
img_height, img_width = 224, 224
num_classes = 7  # Número de classes

# Data generators
train_generator = image_dataset_from_directory(
  dataset,
  labels='inferred',
  label_mode='categorical',
  class_names=None,
  subset="training",
  seed=123,
  validation_split=0.2,
  color_mode='rgb',
  batch_size=batch_size,
  image_size=(img_height, img_width),
  shuffle=True)


validation_generator = image_dataset_from_directory(
  dataset,
  labels='inferred',
  label_mode='categorical',
  class_names=None,
  color_mode='rgb',
  subset="validation",
  validation_split=0.2,
  seed=123,
  batch_size=batch_size,
  image_size=(img_height, img_width),
  shuffle=False)

normalization_layer = Rescaling(1./255)
normalized_train = train_generator.map(lambda x, y: (normalization_layer(x), y))
normalized_validator = validation_generator.map(lambda x, y: (normalization_layer(x), y))

print("Classes detectadas no treinamento:", train_generator.class_names)
print("Número de classes no treinamento:", len(train_generator.class_names))

# Carregar a base do VGG16
vgg16_base = VGG16(weights='imagenet', include_top=False, classifier_activation='softmax', input_shape=(img_height, img_width, 3))
vgg16_base.trainable = False

# Criar o modelo
model = Sequential([
     vgg16_base,
     Flatten(name='flatten'),
     Dense(256, activation='relu', name='fc1'),
     Dropout(0.1),
     Dense(128, activation='relu', name='fc2'),
     Dropout(0.1),
     Dense(num_classes, activation='softmax', name='output')
])

model.summary()
# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
epochs = 10

history = model.fit(
    normalized_train,
    validation_data=normalized_validator,
    epochs=epochs
)

model_pkl_file = "trained_classifier_image_formula.pkl"  
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)
