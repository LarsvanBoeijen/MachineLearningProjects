import cv2
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 6
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python flowers.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    
    # Split data into training and testing sets
    # labels = tf.one_hot(labels, depth = NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    
    # Get a compiled neural network
    model = get_model()
    model.summary()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
    

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # # Create arrays for images and labels
    # images = np.empty((0, IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8)
    # labels = []
    
    # # Get the absolute paths to the image directories
    # image_dirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir)]
    
    # # Loop over each image in each subdirectory of data_dir
    # for label, image_dir in enumerate(image_dirs):
    #     for file in os.listdir(image_dir):
    #         # Determine the path to the image
    #         path = os.path.join(image_dir, file)
            
    #         # Convert image to numpy array
    #         image = cv2.imread(path)
            
    #         # Resize the image and add it to the images array
    #         image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    #         images = np.append(images, [image], axis=0)
            
    #         # Add the current label to labels
    #         labels.append(label)
            
    # return images, labels
    
    # Create lists for images and labels
    images = []
    labels = []
    
    # Loop over each image in each subdirectory of data_dir
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            
            # Determine the path to the image
            path = os.path.join(subdir, file)
            
            # Convert image to numpy array
            image = cv2.imread(path)
            
            # Resize the image and add it to the images list
            images.append(cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)))
            
            # Add the current subdirectory to labels
            labels.append(os.path.basename(subdir))
            
    labels = [eval(label) for label in labels]
            
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a neural network
    model = tf.keras.models.Sequential(
        [
            # Set the input shape
            tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            
            # Convolutional layer. Learn 32 filters using a 3x3 kernel
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            
            # Convolutional layer. Learn 32 filters using a 3x3 kernel
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),

            # Avg-pooling layer, using 2x2 pool size
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            
            # Convolutional layer. Learn 64 filters using a 3x3 kernel
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            
            # Convolutional layer. Learn 64 filters using a 3x3 kernel
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            
            # Avg-pooling layer, using 2x2 pool size
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            
            # Flatten the data
            tf.keras.layers.Flatten(),
            
            # Add a hidden layer with 264 units, with ReLU activation
            tf.keras.layers.Dense(1024, activation="relu"),
            
            # Add dropout
            tf.keras.layers.Dropout(0.30),
                
            # Add a hidden layer with 264 units, with ReLU activation
            tf.keras.layers.Dense(1024, activation="relu"),
            
            # Add dropout
            tf.keras.layers.Dropout(0.30),
            
            # Add output layer with NUM_CATEGORIES units, with softmax activation
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="linear"),
        ]
    )
    
    # Train neural network
    model.compile(
        optimizer="adam",
        loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()
