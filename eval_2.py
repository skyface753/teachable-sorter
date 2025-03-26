#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter


def load_labels(labels_path):
    """Load labels from file."""
    with open(labels_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def evaluate_model(model_path, labels_path, dataset_dir, input_size=(224, 224)):
    # Load the Edge TPU model with the Coral delegate.
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Load labels and build a mapping from class name to index.
    labels = load_labels(labels_path)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    total_images = 0
    correct_predictions = 0

    # For per-class statistics
    per_class_total = {label: 0 for label in labels}
    per_class_correct = {label: 0 for label in labels}

    # Iterate through each class subdirectory in the validation dataset.
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name not in label_to_index:
            print(
                f"Warning: Class folder '{class_name}' not found in labels.txt. Skipping.")
            continue

        gt_index = label_to_index[class_name]

        # Process each image file in the class directory.
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            try:
                # Open and resize the image.
                image = Image.open(image_path).convert("RGB")
                image = image.resize(input_size, Image.LANCZOS)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            # Convert image to a NumPy array.
            input_image = np.array(image)
            # The pycoral adapter function takes care of quantization details.
            common.set_input(interpreter, input_image)
            interpreter.invoke()

            # Get the output tensor and determine the predicted class.
            output = np.copy(common.output_tensor(interpreter, 0))
            predicted_index = int(np.argmax(output))

            total_images += 1
            per_class_total[class_name] += 1
            print(
                f"Predicted: {labels[predicted_index]}, Ground truth: {class_name}")

            if predicted_index == gt_index:
                correct_predictions += 1
                per_class_correct[class_name] += 1

    # Calculate and print overall accuracy.
    overall_accuracy = (correct_predictions / total_images) * \
        100 if total_images else 0
    print(
        f"Overall accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_images})")

    # Print per-class accuracy.
    print("\nPer-class accuracy:")
    for label in labels:
        count = per_class_total[label]
        if count > 0:
            acc = (per_class_correct[label] / count) * 100
            print(
                f"  {label}: {acc:.2f}% ({per_class_correct[label]}/{count})")
        else:
            print(f"  {label}: No samples found.")


if __name__ == "__main__":
    # Paths to the model, labels, and validation dataset directory.
    model_path = "model_edgetpu_self.tflite"
    labels_path = "labels.txt"
    dataset_dir = "dataset/valid"

    evaluate_model(model_path, labels_path, dataset_dir)
