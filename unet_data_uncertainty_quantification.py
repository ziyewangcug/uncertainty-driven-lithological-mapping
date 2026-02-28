import os
import numpy as np
import csv
import scipy.io as scio
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tifffile
from scipy.ndimage import binary_dilation, generate_binary_structure

# Hyperparameter settings
DATA_PATH = 'C:/xx/'  # Define path
LOG_CSV_PATH = f'{DATA_PATH}training_log.csv'
WINDOW_SIZE = 40 # sliding window size
NUM_CLASSES = 8 # number of classes
LEARNING_RATE = 1e-5
BATCH_SIZE = 128
EPOCHS = 50
NOISE_LEVEL = 0.05 # Gaussian noise intensity
NUM_ITERATIONS = 100
STEP_SIZE = 1 # Sliding step size
SAMPLE_RATIO = 0.2 # Training samples ratio

# Label Generator
def generate_soft_labels(DATA_PATH, num_generations=100, boundary_width=2): # Define boundary width=2
    label_map = tifffile.imread(f'{DATA_PATH}class_train.tif')
    class_min, class_max = label_map.min(), label_map.max()

    # Detect boundaries
    boundary_mask = np.zeros_like(label_map, dtype=bool)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(label_map, dx, axis=0), dy, axis=1)
            boundary_mask |= (label_map != shifted)

    # Dilate boundaries
    se = generate_binary_structure(2, 1)
    uncertain_mask = binary_dilation(boundary_mask, structure=se, iterations=boundary_width)

    # Generate noisy soft labels
    soft_labels = []
    for _ in range(num_generations):
        soft_label = label_map.astype(float).copy()
        noise_map = 0.1 * (2 * np.random.rand(*label_map.shape) - 1)  # [-0.1, 0.1]
        soft_label[uncertain_mask] += noise_map[uncertain_mask]
        soft_label = np.clip(soft_label, class_min - 0.49, class_max + 0.49)
        soft_labels.append(soft_label)

# save soft labels as .mat
    scio.savemat(f"{DATA_PATH}soft_labels_collection.mat", {"soft_labels": soft_labels})
    print(f"Saved {num_generations} soft labels to soft_labels_collection.mat")
    return soft_labels

# Load training data and label
def load_data(soft_labels=None, gen_idx=0):
    raw_data = tifffile.imread(f'{DATA_PATH}image_train.tif')
    if soft_labels is None:
        labels = tifffile.imread(f'{DATA_PATH}class_train.tif')
    else:
        labels = soft_labels[gen_idx]

    train_samples, val_samples = [], []
    train_labels, val_labels = [], []

    for class_id in range(1, NUM_CLASSES + 1):
        locs = np.argwhere(np.round(labels) == class_id)
        step_sampled = locs[::STEP_SIZE]
        if len(step_sampled) == 0:
            continue
        sampled = step_sampled[np.random.choice(len(step_sampled),
                                                int(len(step_sampled) * SAMPLE_RATIO),
                                                replace=False)]
        for idx, (x, y) in enumerate(sampled):
            x_start = x - WINDOW_SIZE // 2
            y_start = y - WINDOW_SIZE // 2
            patch = raw_data[x_start:x_start + WINDOW_SIZE,
                             y_start:y_start + WINDOW_SIZE, :]

            if patch.size == WINDOW_SIZE ** 2 * raw_data.shape[2]:
                target = [labels[x, y] - 1]
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

# U-Net Architecture

def build_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)

    # Encoder layer
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)

    # Decoder layer
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(128, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(64, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(32, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)

    # Output layer
    x = Conv2D(NUM_CLASSES, 1, activation='softmax')(conv7)
    x = Flatten()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Noise Injection
def add_gaussian_noise(patch):
    min_val = np.min(patch)
    max_val = np.max(patch)
    data_range = max_val - min_val
    noise = NOISE_LEVEL * data_range * np.random.randn(*patch.shape)
    noisy_patch = patch + noise
    return np.clip(noisy_patch, min_val, max_val)

def evaluate_model_with_noise(model):
    raw_data = tifffile.imread(f'{DATA_PATH}image_test.tif')
    H, W, C = raw_data.shape
    H_out = H - WINDOW_SIZE + 1
    W_out = W - WINDOW_SIZE + 1

    predictions = np.zeros((H_out, W_out), dtype=np.uint8)
    for i in range(H_out):
        for j in range(W_out):
            patch = raw_data[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE, :]
            noisy_patch = add_gaussian_noise(patch)
            noisy_patch = np.expand_dims(noisy_patch, axis=0)
            pred = model.predict(noisy_patch, verbose=0)
            class_pred = np.argmax(pred, axis=1)[0]
            predictions[i, j] = class_pred
    return predictions


# Main Loop
if __name__ == '__main__':
    with open(LOG_CSV_PATH, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['iteration', 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

    all_predictions = []

    # Generate soft labels first
    soft_labels = generate_soft_labels(DATA_PATH, num_generations=NUM_ITERATIONS)

    # Iterative training
    for iter in range(NUM_ITERATIONS):
        print(f"\n=== Iteration {iter+1}/{NUM_ITERATIONS} with soft labels ===")
        train_data, val_data, train_labels, val_labels, num_channels = load_data(soft_labels, gen_idx=iter)
        input_shape = (WINDOW_SIZE, WINDOW_SIZE, num_channels)

        model = build_model(input_shape)

# Save training loss and accuracy
        class LogCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                with open(LOG_CSV_PATH, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([
                        iter + 1,
                        epoch + 1,
                        logs['loss'],
                        logs['accuracy'],
                        logs['val_loss'],
                        logs['val_accuracy']
                    ])

        model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=[LogCallback()]
        )

        noise_preds = evaluate_model_with_noise(model)
        all_predictions.append(noise_preds)

# Save results as .mat
    prediction_cube = np.stack(all_predictions, axis=-1)
    output_path = f'{DATA_PATH}uncertainty_predictions_cube.mat'
    print(f"Saving 3D prediction cube to {output_path}")
    scio.savemat(output_path, {'prediction_cube': prediction_cube})
    print("All iterations completed. Results saved.")
