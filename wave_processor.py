import os
import uuid

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity

def process(buffer):
    processed_buffer = []
    all_formulas = getFormulas()
    
    for i in range(0, len(buffer)):
        print(buffer[i])
        binary_image = buffer[i]['figure_buffer']
        name_file = uuid.uuid4()
        path = "./tmp/" + str(name_file) + ".wav"
        with open(path, "wb") as f:
            binary_image.seek(0)  # Make sure to move the cursor to the beginning of the BytesIO object
            binary_data = binary_image.getvalue()  # Get the bytes from the BytesIO object
            f.write(binary_data)
        simility = []
        for y in range(0, len(all_formulas)):
            form = './formula/' + all_formulas[y]
            simility.append({'formula': all_formulas[y], 'percent': compare_images(form,path)})
        max_similarity = max(simility, key=lambda x: x['percent'])
        processed_buffer.append( { 'analised': buffer[i]['title'], 'simility': max_similarity })
    return processed_buffer

def load_and_preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def compare_images(image_path1, image_path2):
    # Load and preprocess images
    img1 = load_and_preprocess_image(image_path1, target_size=(224, 224))
    img2 = load_and_preprocess_image(image_path2, target_size=(224, 224))

    # Load pre-trained VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

    # Extract features
    features1 = model.predict(img1)
    features2 = model.predict(img2)

    # Calculate cosine similarity
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

    return similarity


def getFormulas(directory="./formula"):
    files = []
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            # Check if the filepath is a file (not a directory)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as file:
                    # Append the filename and content to the list
                    files.append(filename)
    return files