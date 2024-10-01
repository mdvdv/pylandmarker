# Landmarker

MediaPipe-based toolkit for landmark localization and export.

## Installation

1. Clone the repository locally.

```bash
git clone https://github.com/mdvdv/pylandmarker.git
```

2. Install basic requirements.

```bash
# With PDM
pdm install

# With PyPI
python -m pip install -r requirements.txt

# With conda
conda create --name landmarker python=3.10
conda activate landmarker
python -m pip install -r requirements.txt
```

## Usage

Run detection model on a single image.

```bash
python -m scripts.run_on_single_image --image "assets/face.jpg" --task "face" --config "configs/face_detector.yaml"
```

## Test

```bash
python -m pytest
```

## Documentation

In progress.

## Citation

```bash
@software{pylandmarker,
    title = {landmarker},
    author = {Medvedev, Anatolii},
    year = {2024},
    url = {https://github.com/mdvdv/pylandmarker.git},
    version = {1.0.0}
}
```
