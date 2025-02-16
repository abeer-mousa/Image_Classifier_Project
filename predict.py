import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Global variable to store class names (loaded from a label map JSON)
class_names = {}


def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image.numpy(), axis=0)

    return image

def prepare_image(image_path):
    image = Image.open(image_path)
    test_image = np.asarray(image)

    processed_image = process_image(test_image)
    processed_test_image = np.squeeze(processed_image)

    # Create a subplot to display the original and processed images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(test_image)
    ax1.set_title('Original Image')

    ax2.imshow(processed_test_image)
    ax2.set_title('Processed Image')

    # Adjust layout and display the images
    plt.tight_layout()
    plt.show()

    return processed_image


def plot_top_k_classes(top_k_classes, top_k_probs):
    top_k_class_names = [class_names[str(idx)] for idx in top_k_classes]

    plt.figure(figsize=(8, 6))
    plt.barh( top_k_class_names, top_k_probs, color='skyblue')
    plt.xlabel('Probability')
    plt.title('Top K Flower Predictions')
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest probability at the top
    plt.show()


def predict(image_path, model, top_k=5):

    processed_image = prepare_image(image_path)
    predictions = model.predict(processed_image)

    # Get the top K indices and probabilities
    top_k_indices = tf.argsort(predictions, axis=-1, direction='DESCENDING')[:, :top_k]
    top_k_probs = tf.gather(predictions, top_k_indices, batch_dims=1)

    # Convert to numpy arrays
    top_k_probs = top_k_probs.numpy().flatten()
    top_k_classes = top_k_indices.numpy().flatten().astype(str)

    plot_top_k_classes(top_k_classes, top_k_probs)

    for _class, prob in zip(top_k_classes, top_k_probs):
       print(f'class name: {class_names[str(_class)]} | Probability: {prob}')


def load_label_map(label_map_path):
    global class_names
    with open(label_map_path, 'r') as f:
        class_names = json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to a JSON file mapping labels to flower names')
    

    args = parser.parse_args()


    if args.category_names:
        load_label_map(args.category_names)


    model = load_model(args.model_path)

    predict(args.image_path, model, top_k=args.top_k)

if __name__ == '__main__':
    main()
