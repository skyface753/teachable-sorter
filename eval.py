import argparse
import time
import os
import numpy as np
from PIL import Image
from pycoral.adapters import classify, common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def load_images(dataset_path, labels):
    """Load images and ensure correct class mappings from labels.txt."""
    image_paths = []
    ground_truths = []

    # Reverse the label mapping from labels.txt
    # e.g., {'5c': 0, '2e': 1}
    name_to_index = {v: k for k, v in labels.items()}

    # List directories, filtering out hidden/system files
    class_names = sorted([d for d in os.listdir(dataset_path) if not d.startswith(
        '.') and os.path.isdir(os.path.join(dataset_path, d))])

    # Build the class_mappings dictionary correctly
    class_mappings = {class_name: name_to_index[class_name]
                      for class_name in class_names if class_name in name_to_index}

    for class_name, class_id in class_mappings.items():
        class_dir = os.path.join(dataset_path, class_name)

        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, img_file))
                ground_truths.append(class_id)

    return image_paths, ground_truths


def preprocess_image(image_path, input_size, mean=128.0, std=128.0, scale=1.0, zero_point=0):
    """Preprocess an image for inference with Edge TPU model."""
    image = Image.open(image_path).convert(
        'RGB').resize(input_size, Image.LANCZOS)

    # Normalize and quantize the image
    normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)

    return normalized_input.astype(np.uint8)


def evaluate_model(model_path, dataset_path, labels_path, top_k=1, threshold=0.0):
    """Evaluates the TFLite model using the dataset and computes accuracy."""
    print("Loading model...")
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    labels = read_label_file(labels_path) if labels_path else {}

    input_size = common.input_size(interpreter)
    params = common.input_details(interpreter, 'quantization_parameters')
    scale, zero_point = params['scales'], params['zero_points']

    # Load dataset
    image_paths, ground_truths = load_images(dataset_path, labels)

    correct_predictions = 0
    total_predictions = len(image_paths)

    print(f"Evaluating {total_predictions} images...")

    print(labels)

    for img_path, true_label in zip(image_paths, ground_truths):
        input_data = preprocess_image(
            img_path, input_size, scale=scale, zero_point=zero_point)
        common.set_input(interpreter, input_data)

        start_time = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start_time

        results = classify.get_classes(interpreter, top_k, threshold)
        predicted_label = results[0].id if results else None
        predicted_score = results[0].score if results else 0.0

        if predicted_label == true_label:
            correct_predictions += 1

        print(f"Image: {os.path.basename(img_path)} | "
              f"True: {labels.get(true_label, true_label)} | "
              f"Predicted: {labels.get(predicted_label, predicted_label)} | "
              f"Score: {predicted_score:.4f} | "
              f"Time: {inference_time * 1000:.1f}ms")

    accuracy = (correct_predictions / total_predictions) * 100
    print(
        f"\nModel Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correct)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a TFLite model on a dataset using PyCoral.")
    parser.add_argument('-m', '--model', required=True,
                        help="Path to the .tflite model file")
    parser.add_argument('-d', '--dataset', required=True,
                        help="Path to the dataset directory")
    parser.add_argument('-l', '--labels', required=True,
                        help="Path to labels.txt file")
    args = parser.parse_args()

    evaluate_model(args.model, args.dataset, args.labels)
