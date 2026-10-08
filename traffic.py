import cv2 #cv is computer vision!
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

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
    # note: i think gtsrb is basically data_dir

    # goal: read images from gtsrb and turn them into format AI can understand

    # create empty lists "images" and "labels"
    images = []
    labels = []

    # access folders & files: iterate through 43 folders (NUM_CATEGORIES - 1). each folder name is my label for images inside
    for folder in range(NUM_CATEGORIES):
        # build folder path
        folder_path = os.path.join(data_dir, str(folder))

        # get inside to now get LIST of image file names (like 0000.ppm, 0002.ppm for example)
        files_in_folder = os.listdir(folder_path)

        # second loop for each filname in that folder, build path and read image
        # filename is the individual ppm files in the subfolders
        for filename in files_in_folder:
            # Build the address for the specific image file
            file_path = os.path.join(folder_path, filename)

            # read the images: using cv2.imread to read each img file as numerical array (numpy.ndarray). thats replacing open() func.
            # below returns the numerical array i need
            img = cv2.imread(file_path)

            # use cv2.resize [does it produce an array?] -  to make every image same dimensions (IMG_WIDTH by IMG_HEIGHT) bc neural network requires fixed input shape for all
            # add to images list (and labels list?). do it at once?
            # pass img into below
            images.append(cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)))

            # add index, folder, to labels list
            labels.append(folder) # what do i put in here

    # return values - return tuple (images, labels). images is a list of all resized arrays and labels is list of corresponding integers (folder names 0 through 42)
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # see lecture code on tensorflow layers number and types used. OPTIONAL
    # GOAL: build and compile a neural network using TensorFlow that can categorize images into 43 different road sign types

    # create a convolutional neural network (CNN)
    model = tf.keras.models.Sequential([

        # add conv2d layer to extract features. specify num filters, kernel size, input_shape of img_width img_height and 3
        # Learn 34 filters using a 3x3 kernel EDIT: 128, (5,5)
        tf.keras.layers.Conv2D(
            128, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # pooling - add maxPooling2d layer to reduce dimensions of my feature maps, making model faster and more robust
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # flatten - flatten layer to turn 2d image data into 1d vector so it can get into hidden layers
        tf.keras.layers.Flatten(),

        # hidden layers - add 1+ dense layers (often w relu activation) to learn complex patterns
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),

        # dropout - add dropout layer (eg 0.5) to prevent overfitting by randomly ignoring some units during training
        tf.keras.layers.Dropout(0.4),

        # output - finally, add final dense layer with NUM-CATEGORIES units & softmax activation to produce probabilities for each sign type
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # TRAIN neural network - use model.compile to set the optimizer (like "adam"), the loss function (use "categorical_crossentropy" for this project), and the metrics (usually ["accuracy"]
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
