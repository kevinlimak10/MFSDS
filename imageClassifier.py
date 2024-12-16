from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the classifier (you can use any other classifier as well)
clf_amplitude = RandomForestClassifier()
clf_frequency = RandomForestClassifier()
clf_phase = RandomForestClassifier()

# Train the classifiers
clf_amplitude.fit(X_train, y_train['amplitude'])
clf_frequency.fit(X_train, y_train['frequency'])
clf_phase.fit(X_train, y_train['phase'])

# Step 4: Model Evaluation

# Evaluate the classifiers
y_pred_amplitude = clf_amplitude.predict(X_test)
accuracy_amplitude = accuracy_score(y_test['amplitude'], y_pred_amplitude)

y_pred_frequency = clf_frequency.predict(X_test)
accuracy_frequency = accuracy_score(y_test['frequency'], y_pred_frequency)

y_pred_phase = clf_phase.predict(X_test)
accuracy_phase = accuracy_score(y_test['phase'], y_pred_phase)

print("Accuracy for Amplitude:", accuracy_amplitude)
print("Accuracy for Frequency:", accuracy_frequency)
print("Accuracy for Phase:", accuracy_phase)

# Step 5: Prediction

# Given a new graph image, apply feature extraction
# Use the trained classifiers to predict the attributes (higher, lower, medium)
# Output the predictions

# Function to load images and corresponding labels
def load_data(root_dir):
    images = []
    labels = {'amplitude': [], 'frequency': [], 'phase': []}

    for category in labels.keys():
        category_dir = os.path.join(root_dir, category)
        for label in os.listdir(category_dir):
            label_dir = os.path.join(category_dir, label)
            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                # Read image and append it to the list
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                # Append label
                labels[category].append(label)

    return np.array(images), labels


# Root directory containing subfolders for each attribute
root_dir = 'path/to/directory'

# Load images and labels
images, labels = load_data(root_dir)

# Preprocess the images if necessary (e.g., resize, normalization)

# Convert labels to pandas DataFrame for easier handling
import pandas as pd

labels_df = pd.DataFrame(labels)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels_df, test_size=0.2, random_state=42)


# Apply feature extraction techniques
# For example, you can resize images and convert them to grayscale
def preprocess_images(images):
    processed_images = []
    for img in images:
        # Resize image if necessary
        resized_img = cv2.resize(img, (new_width, new_height))  # Specify new_width and new_height
        # Convert to grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray_img)
    return np.array(processed_images)

# Preprocess training and testing images
X_train_processed = preprocess_images(X_train)
X_test_processed = preprocess_images(X_test)

# Flatten images to be used as features
X_train_features = X_train_processed.reshape(X_train_processed.shape[0], -1)
X_test_features = X_test_processed.reshape(X_test_processed.shape[0], -1)
