import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization

base_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(base_dir, 'dataset', 'train')
test_dir = os.path.join(base_dir, 'dataset', 'test')

if not os.path.exists(train_dir):
    print(f"ERROR: Could not find dataset at: {train_dir}")
    print("Make sure you created a 'dataset' folder and put 'train' and 'test' inside it.")
    exit()

print(f"Loading data from: {train_dir}")

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    shuffle=True,
    label_mode='int'
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    shuffle=False,
    label_mode='int'
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. BUILD MODEL ---
print("Building Lightweight CNN...")
model = Sequential([
    # Convolution Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Convolution Block 2
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Convolution Block 3
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 4. TRAIN ---
print("Starting Training...")
model.fit(
    train_data,
    validation_data=test_data,
    epochs=30
)

# --- 5. SAVE ---
model.save('emotion_model.h5')
print("\nSUCCESS: Model saved as 'emotion_model.h5'")