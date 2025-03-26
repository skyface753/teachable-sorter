import tensorflow as tf
import os

# Set dataset directories
train_dir = "dataset/train"
val_dir = "dataset/valid"

# Parameters
img_size = (224, 224)
batch_size = 16
num_epochs = 50

# Load the training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Get the class names and export them to labels.txt
class_names = train_ds.class_names
print("Class names:", class_names)
with open("labels.txt", "w") as f:
    for label in class_names:
        f.write(label + "\n")

# Build a model using MobileNetV2 as a feature extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze the base

inputs = tf.keras.Input(shape=img_size + (3,))
# Preprocess input using MobileNetV2's preprocessing function
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# Final dense layer for classification
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=num_epochs)

# Save the model as a TensorFlow SavedModel
saved_model_dir = "saved_model"
model.export(saved_model_dir)

# Define a representative dataset generator for quantization calibration


def representative_data_gen():
    # Use a few batches from the training set for calibration
    for input_data, _ in train_ds.take(100):
        yield [input_data.numpy()]


# Convert the SavedModel to a fully quantized TFLite model (uint8) for the Edge TPU
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that all ops get quantized to int8 and that the input/output are uint8.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the TFLite model to disk
with open("model_edgetpu_self.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model with uint8 quantization saved as model_edgetpu_self.tflite")
