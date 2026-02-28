import os
import numpy as np
import csv
import scipy.io as scio
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input, \
    concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import tifffile

# Hyperparameter setting
DATA_PATH = 'C:/xx'  # Define path
WINDOW_SIZE = 40  # sliding window size
NUM_CLASSES = 8 # number of classes
LEARNING_RATE = 1e-5
BATCH_SIZE = 128
EPOCHS = 100
MC_SAMPLES = 100  # Monte Carlo sampling
STEP_SIZE = 1  # Sliding step size
SAMPLE_RATIO = 0.2  # Training samples ratio

# Load training data and label
def load_data():
    raw_data = tifffile.imread(f'{DATA_PATH}image_train.tif')
    labels = tifffile.imread(f'{DATA_PATH}class_train.tif')
    train_samples, val_samples = [], []
    train_labels, val_labels = [], []

    for class_id in range(1, NUM_CLASSES + 1):
        locs = np.argwhere(labels == class_id)
        step_sampled = locs[::STEP_SIZE]
        sampled = step_sampled[np.random.choice(len(step_sampled), int(len(step_sampled) * SAMPLE_RATIO), replace=False)]

        for idx, (x, y) in enumerate(sampled):
            x_start = x - WINDOW_SIZE // 2
            y_start = y - WINDOW_SIZE // 2
            patch = raw_data[x_start:x_start + WINDOW_SIZE,
                    y_start:y_start + WINDOW_SIZE, :]

            if patch.size == WINDOW_SIZE ** 2 * raw_data.shape[2]:
                target = [class_id - 1]
                if idx < int(len(sampled) * 0.8):
                    train_samples.append(patch)
                    train_labels.append(target)
                else:
                    val_samples.append(patch)
                    val_labels.append(target)

    return (np.asarray(train_samples, 'float32'),
            np.asarray(val_samples, 'float32'),
            np.asarray(train_labels),
            np.asarray(val_labels),
            raw_data.shape[2])

# U-Net network architecture
def build_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    # Encoder layer
    # Block 1
    conv1 = Conv2D(32, 3, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    relu1 = tf.keras.layers.Activation('relu')(bn1)
    conv1 = Conv2D(32, 3, padding='same')(relu1)
    bn1 = BatchNormalization()(conv1)
    relu1 = tf.keras.layers.Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(relu1)
    dropout1 = Dropout(0.2)(pool1)

    # Block 2
    conv2 = Conv2D(64, 3, padding='same')(dropout1)
    bn2 = BatchNormalization()(conv2)
    relu2 = tf.keras.layers.Activation('relu')(bn2)
    conv2 = Conv2D(64, 3, padding='same')(relu2)
    bn2 = BatchNormalization()(conv2)
    relu2 = tf.keras.layers.Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(relu2)
    dropout2 = Dropout(0.2)(pool2)

    # Block 3
    conv3 = Conv2D(128, 3, padding='same')(dropout2)
    bn3 = BatchNormalization()(conv3)
    relu3 = tf.keras.layers.Activation('relu')(bn3)
    conv3 = Conv2D(128, 3, padding='same')(relu3)
    bn3 = BatchNormalization()(conv3)
    relu3 = tf.keras.layers.Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(relu3)
    dropout3 = Dropout(0.2)(pool3)

    # Bottleneck
    conv4 = Conv2D(256, 3, padding='same')(dropout3)
    bn4 = BatchNormalization()(conv4)
    relu4 = tf.keras.layers.Activation('relu')(bn4)
    conv4 = Conv2D(256, 3, padding='same')(relu4)
    bn4 = BatchNormalization()(conv4)
    relu4 = tf.keras.layers.Activation('relu')(bn4)
    dropout4 = Dropout(0.2)(relu4)

    # Decoder layer
    # Block 5
    up5 = UpSampling2D(size=(2, 2))(dropout4)
    up5 = Conv2D(128, 2, padding='same')(up5)
    bn_up5 = BatchNormalization()(up5)
    up5 = tf.keras.layers.Activation('relu')(bn_up5)
    merge5 = concatenate([relu3, up5], axis=3)

    conv5 = Conv2D(128, 3, padding='same')(merge5)
    bn5 = BatchNormalization()(conv5)
    relu5 = tf.keras.layers.Activation('relu')(bn5)
    conv5 = Conv2D(128, 3, padding='same')(relu5)
    bn5 = BatchNormalization()(conv5)
    relu5 = tf.keras.layers.Activation('relu')(bn5)

    # Block 6
    up6 = UpSampling2D(size=(2, 2))(relu5)
    up6 = Conv2D(64, 2, padding='same')(up6)
    bn_up6 = BatchNormalization()(up6)
    up6 = tf.keras.layers.Activation('relu')(bn_up6)
    merge6 = concatenate([relu2, up6], axis=3)

    conv6 = Conv2D(64, 3, padding='same')(merge6)
    bn6 = BatchNormalization()(conv6)
    relu6 = tf.keras.layers.Activation('relu')(bn6)
    conv6 = Conv2D(64, 3, padding='same')(relu6)
    bn6 = BatchNormalization()(conv6)
    relu6 = tf.keras.layers.Activation('relu')(bn6)

    # Block 7
    up7 = UpSampling2D(size=(2, 2))(relu6)
    up7 = Conv2D(32, 2, padding='same')(up7)
    bn_up7 = BatchNormalization()(up7)
    up7 = tf.keras.layers.Activation('relu')(bn_up7)
    merge7 = concatenate([relu1, up7], axis=3)

    conv7 = Conv2D(32, 3, padding='same')(merge7)
    bn7 = BatchNormalization()(conv7)
    relu7 = tf.keras.layers.Activation('relu')(bn7)
    conv7 = Conv2D(32, 3, padding='same')(relu7)
    bn7 = BatchNormalization()(conv7)
    relu7 = tf.keras.layers.Activation('relu')(bn7)

    # Output layer
    x = Conv2D(NUM_CLASSES, 1)(relu7)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Flatten()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Monte Carlo Dropout
# Load testing data
def evaluate_model_mc_dropout(model):
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    height = raw_data.shape[0] - WINDOW_SIZE + 1
    width = raw_data.shape[1] - WINDOW_SIZE + 1
# Monte Carlo sampling times
    mc_predictions = np.zeros((height, width, MC_SAMPLES), dtype=np.uint8)

    mc_model = tf.keras.models.Model(inputs=model.input, outputs=model.output)

    # Applying dropout
    @tf.function
    def mc_predict(x):
        return mc_model(x, training=True)

    for sample_idx in range(MC_SAMPLES):
        print(f"MC Dropout sampling {sample_idx + 1}/{MC_SAMPLES}")
        predictions = []

        for i in range(height):
            row_patches = [
                raw_data[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE, :]
                for j in range(width)
            ]
            batch_data = np.array(row_patches, dtype='float32')

            batch_pred = mc_predict(batch_data).numpy()
            predictions.extend(np.argmax(batch_pred, axis=1))

        pred_map = np.array(predictions).reshape((height, width))
        mc_predictions[:, :, sample_idx] = pred_map

    return mc_predictions

    # Save training loss curves
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.show()

# Main function
if __name__ == '__main__':
    train_data, val_data, train_labels, val_labels, num_channels = load_data()
    input_shape = (WINDOW_SIZE, WINDOW_SIZE, num_channels)

    # Model training
    model = build_model(input_shape)
    history = model.fit(train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)

    # Save training loss and accuracy
    with open('training_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        for epoch in range(len(history.history['loss'])):
            row = [
                epoch + 1,
                history.history['loss'][epoch],
                history.history['accuracy'][epoch],
                history.history['val_loss'][epoch],
                history.history['val_accuracy'][epoch]
            ]
            writer.writerow(row)

    # Model evaluation
    mc_pred_map = evaluate_model_mc_dropout(model)

    # Save results as .mat
    scio.savemat(f'{DATA_PATH}mc_pred_map.mat', {'mc_pred_map': mc_pred_map})
    plot_history(history)

    # Mean
    avg_pred_map = np.mean(mc_pred_map, axis=-1)
    plt.imshow(avg_pred_map, cmap='jet')
    plt.title('Average Prediction Map')
    plt.show()

    # Variance
    uncertainty_map = np.std(mc_pred_map, axis=-1)
    plt.imshow(uncertainty_map, cmap='hot')
    plt.title('Uncertainty Map')
    plt.colorbar()
    plt.show()