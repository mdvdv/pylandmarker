import argparse

import numpy as np

from landmarker import Predictor, load_image, read_yaml


def run_on_single_image(image: np.ndarray, task: str, config: dict) -> dict:
    model = Predictor(task=task, config=config)
    predictions = model.predict(image)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Landmarker on a single image.")
    parser.add_argument("--image", type=str, help="Path to the input image.")
    parser.add_argument("--task", type=str, help="Detection task.")
    parser.add_argument(
        "--config", type=str, help="Path to the model configuration file."
    )
    args = parser.parse_args()

    image = load_image(args.image)
    config = read_yaml(args.config)
    predictions = run_on_single_image(image=image, task=args.task, config=config)
    print(predictions)
