import tensorflow as tf
import numpy as np
import cv2
import os
import sys

from sklearn.model_selection import train_test_split


def main():

    images, labels = load_data(sys.argv[1])

    class_weights = calculate_weights(labels)

    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=0.2, stratify=np.array(labels)
    )

    x_train = x_train / 255
    x_test = x_test / 255

    model = get_model()
    model.fit(x_train, y_train, epochs=35, validation_split=0.1, class_weight=class_weights)
    model.evaluate(x_test, y_test, verbose=2)


def load_data(directory):

    images = []
    labels = []
    count = 0

    for i in os.listdir(directory):
        path = os.path.join(directory, i)
        if len(os.listdir(path)) >= 50:
            for j in os.listdir(path):

                img_path = os.path.join(path, j)

                image = cv2.imread(img_path)
                image = cv2.resize(image, (62, 47))

                images.append(image)
                labels.append(count)
            count += 1
    result = (images, labels)

    return result


def calculate_weights(classes):
    classes = np.array(classes)
    unique_classes = np.unique(classes)
    samples = len(classes)
    class_weights = {}

    for i in unique_classes:
        class_weights[i] = (samples / (len(unique_classes) * np.count_nonzero(classes == i)))

    return class_weights


def get_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(62, 47, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        tf.keras.layers.Dropout(0.3),


        tf.keras.layers.Dense(12, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy", "precision"]
                  )

    return model


main()
