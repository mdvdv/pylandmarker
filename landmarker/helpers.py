import json
import os
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import requests
import yaml
from PIL import Image

ACCEPTED_IMAGE_FORMATS = ["PIL", "cv2", "numpy"]


def read_yaml(path: str) -> dict:
    """Read a YAML file."""
    with open(path, mode="r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.loader.SafeLoader)
    return yaml_data


def save_json(data: dict, path: str) -> str:
    """Save data as a JSON file."""
    with open(path, mode="w", encoding="utf-8") as json_file:
        json.load(data, json_file)
    return path


def load_image(
    image: Any,
    return_format: str = "numpy",
) -> Any:
    """Load an image from a file path, URI, PIL Image, or NumPy array.

    Args:
        image (Any): The image to load.
        return_format (str): The format to return the image in.

    Returns:
        Any: The image in the specified format.
    """
    if return_format not in ACCEPTED_IMAGE_FORMATS:
        raise ValueError(f"return_format must be one of {ACCEPTED_IMAGE_FORMATS}.")

    if isinstance(image, Image.Image) and return_format == "PIL":
        return image
    elif isinstance(image, Image.Image) and return_format == "cv2":
        # Channels need to be reversed for cv2
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, Image.Image) and return_format == "numpy":
        return np.array(image)

    if isinstance(image, np.ndarray) and return_format == "PIL":
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, np.ndarray) and return_format == "cv2":
        return image
    elif isinstance(image, np.ndarray) and return_format == "numpy":
        return image

    if isinstance(image, str) and image.startswith("http"):
        if return_format == "PIL":
            response = requests.get(image)
            return Image.open(BytesIO(response.content))
        elif return_format == "cv2" or return_format == "numpy":
            response = requests.get(image)
            pil_image = Image.open(BytesIO(response.content))
            return np.array(pil_image)
    elif os.path.isfile(image):
        if return_format == "PIL":
            return Image.open(image)
        elif return_format == "cv2":
            # Channels need to be reversed for cv2
            return cv2.cvtColor(np.array(Image.open(image)), cv2.COLOR_RGB2BGR)
        elif return_format == "numpy":
            pil_image = Image.open(image)
            return np.array(pil_image)
    else:
        raise ValueError(f"{image} is not a valid file path or URI.")
