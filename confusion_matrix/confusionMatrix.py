import matplotlib.pyplot as plt
from keras.api.utils import image_dataset_from_directory
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
from keras.api.layers import Rescaling
import seaborn as sns
# Caminhos do dataset
dataset = '../dataset/wave_plots'

# Parâmetros
batch_size = 32  # Ajustado para pequenos datasets
img_height, img_width = 224, 224
num_classes = 7  # Número de classes

# Data generators
validation_generator = image_dataset_from_directory(
  dataset,
  labels='inferred',
  label_mode='categorical',
  class_names=None,
  color_mode='rgb',
  seed=123,
  batch_size=batch_size,
  image_size=(img_height, img_width),
  shuffle=False)

model_pkl_file = "../model/trained_classifier_image_formula.pkl" 
# load model from pickle file
with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)

class_names = validation_generator.class_names

normalization_layer = Rescaling(1./255)
normalized_validator = validation_generator.map(lambda x, y: (normalization_layer(x), y))

# Get predictions
y_pred = []
y_true = []
for images, labels in normalized_validator:
    # Predict
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

# # Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = validation_generator.class_names

# # Visualize confusion matrix
fig = plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion-matrix.png')
plt.show()

