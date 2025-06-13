# barcode-detection

Детектирование зон – грубые коробки (bbox detection). Ищем одномерные и двумерные коды на фотографиях: qr, ean13, ean8, upc, 1d (нераспознанные одномерки), dm, az, pdf, id (невозможно детектировать), hd (трудно детектировать).

# Постановка задачи

В данном проекте я решаю задачу детекции одномерных (например, штрих-кодов типа ean13) и двумерных (например, qr-коды) кодов на фотографиях. Моя система анализирует изображения и выдаёт координаты четырех углов для каждой фигуры, в которую помещается обнаруженный код, или сообщает об его отсутствии. 

__Input:__ картинка в формате .jpg или .png – фотография с мобильного устройства (до 5 Мб)

__Output:__ массив вида [(𝑥1, 𝑥2, 𝑥3, 𝑥4), ...] – координаты left top – right bottom углов bbox’а, в котором находится код. Возвращает пустой список, если штрихкод не найден. Есть возможность получить на выход картинку с bbox’ом.

## Метрики

Ключевые метрики - mAP (mean Average Precision), IoU (Intersection over Union), Hausdorff Distance, Precision, Recall, F1-score.

## Валидация

Для разделения выборок использую метод K-fold кросс-валидацию с 5 фолд.

## Данные 

Мой датасет: 117 фотографий 1D и 2D кодов в формате .jpg и соответствующие им файлы .json (имя-фото.jpg.json) с разметкой. Из них:
```
qr  | ean13 | 1d | dm | pdf | upc |  hd | id |

103 |   83  | 69 | 69 |  4  |  3  | 159 | 68 |
```
# Моделирование

__Бейзлайн__

Сравнение со стандартной моделью YOLO (без дообучения)

__Основная модель__

yolov5s.pt

__Внедрение__

Программный пакет состоит из нескольких модулей, отвечающих за обучение и валидацию, а также за инференс модели

# Setup
## Prerequisites
1. Python 3.9 or higher

2. Git

3. Poetry (for dependency management)

## Installation
1. Клонируйте репозиторий:

```
git clone <your-repository-url>
cd barcode-detection
```
2. Установите Poetry (если ещё не установлено):
```
curl -sSL https://install.python-poetry.org | python3 -
```

3. Установите dependencies:
```
poetry install
```

4. Активируйте virtual environment:

```
poetry shell
```

5. Установите pre-commit хуки:

```
pre-commit install
```

6. Проверьте setup:

```
pre-commit run --all-files
```

7. Data Setup
Так как датасет хранится на локальном устройстве, проверить, что есть следующая структура:

```
data/
├── train/
│   ├── image1.jpg
│   ├── image1.json
│   ├── image2.jpg
│   ├── image2.json
│   └── ...
├── val/
│   ├── image1.jpg
│   ├── image1.json
│   └── ...
└── test/
    ├── image1.jpg
    ├── image1.json
    └── ...
```

8. Инициализируйте DVC for data versioning:

```
dvc init
dvc add data/train data/val data/test
git add data.dvc .dvc/config
git commit -m "Add data with DVC"
```

9. MLflow Setup
Start MLflow server for experiment tracking:

```
mlflow server --host 127.0.0.1 --port 8080
```
The MLflow UI will be available at http://127.0.0.1:8080

## Train
1. С дефолтными параметрами:

```
python commands.py train
```

2. С кастомными параметрами
Можете переписать параметры конфигурации:

```
python commands.py train training.epochs=100 training.dataloader.batch_size=32 model.model_size=m
```

3. Training Steps
Data Loading: Загружает автоматически и валидирует данные из configured directories

## Model Initialization: Инициализирует YOLO с pretrained weights

1. Training Loop: Trains the model with PyTorch Lightning

2. Validation: Validates on validation set each epoch

3. Checkpointing: Saves best models based on validation mAP

4. Logging: Logs metrics, parameters, and artifacts to MLflow

## Training Monitoring
1. MLflow UI: Смотрите training metrics, гиперпараметры и артефакты на http://127.0.0.1:8080

2. TensorBoard logs: Automatically logged by PyTorch Lightning

3. Console output: Real-time training progress

4. Plots directory: Training plots saved to plots/ directory

## Configuration Files
Training behavior is controlled by YAML configuration files in configs/:

1. config.yaml - Main configuration

2. data/local.yaml - Data loading configuration

3. model/yolo.yaml - Model architecture configuration

4. training/default.yaml - Training hyperparameters

5. mlflow/default.yaml - MLflow logging configuration

## Production Preparation
Model Conversion to ONNX
1. Convert the trained PyTorch model to ONNX format for deployment:

```
python commands.py convert_onnx outputs/best.pt outputs/model.onnx
```
2. Model Conversion to TensorRT
3. Convert ONNX model to TensorRT for optimized GPU inference:

```
python commands.py convert_tensorrt outputs/model.onnx outputs/model.trt --precision fp16
```

## Production Artifacts
After training and conversion, the following artifacts are ready for production deployment:

1. outputs/best.pt - Best PyTorch model checkpoint

2. outputs/model.onnx - ONNX model for cross-platform deployment

3. outputs/model.trt - TensorRT engine for optimized GPU inference

4. configs/ - Configuration files needed for inference

5. plots/ - Training visualizations and metrics

## Deployment Requirements
Для production deployment вам понадобится:

1. PyTorch or ONNX Runtime (depending on model format)

2. OpenCV for image preprocessing

3. NumPy for numerical operations

4. FastAPI (for REST API server)

## Minimal inference dependencies:

```
pip install torch onnxruntime opencv-python numpy fastapi uvicorn
```

## Infer
Input Data Format
Создаёт картинки в поддерживаемых форматах (JPEG, JPG, PNG). Для формата каждая картинка должна иметь соответствуйщую разметку .json со следующей структурой:

```
json
{
  "objects": [
    {
      "data": [
        [x1, y1],
        [x2, y2], 
        [x3, y3],
        [x4, y4]
      ],
      "class": "qr"
    }
  ]
}
```

Single Image Inference

```
python commands.py infer path/to/image.jpg results/predictions.json
```

Batch Inference
```
python commands.py infer path/to/images/ results/batch_predictions/
```
Using ONNX Model
```
python commands.py infer path/to/image.jpg results/predictions.json --model_path outputs/model.onnx
```
Inference Server
Start a REST API server for real-time inference:

```
python commands.py serve --model_path outputs/best.pt --host 0.0.0.0 --port 8080
```

The server provides the following endpoints:

1. POST /predict - Upload image for detection

2. GET /health - Health check endpoint

3. GET /info - Model information

## Пример API usage:

```
curl -X POST "http://localhost:8080/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"
```
Output Format
Inference results are returned in JSON format:

```
json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class": "qr",
      "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ],
  "image_shape": [height, width],
  "processing_time": 0.123
}
```

## Model Evaluation
Оценка модели на тестовых данных:

```
python commands.py evaluate --model_path outputs/best.pt --data_split test
```

## Project Structure
```
barcode-detection/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data/                  # Data configurations
│   ├── model/                 # Model configurations
│   ├── training/              # Training configurations
│   ├── inference/             # Inference configurations
│   └── mlflow/                # MLflow configurations
├── barcode_detection/         # Main Python package
│   ├── data/                  # Data loading modules
│   │   ├── data_module.py     # PyTorch Lightning DataModule
│   │   ├── dataset.py         # Dataset class
│   │   └── transforms.py      # Data augmentation
│   ├── models/                # Model definitions
│   │   ├── yolo_lightning.py  # PyTorch Lightning YOLO wrapper
│   │   └── metrics.py         # Evaluation metrics
│   ├── training/              # Training utilities
│   │   └── trainer.py         # Training logic
│   ├── inference/             # Inference utilities
│   │   ├── predictor.py       # Model inference
│   │   └── server.py          # FastAPI server
│   └── utils/                 # Utility functions
│       ├── logging.py         # Logging setup
│       └── helpers.py         # Helper functions
├── data/                      # Data directory (not in git)
├── outputs/                   # Model outputs
├── plots/                     # Training plots
├── logs/                      # Log files
├── commands.py                # Main CLI entry point
├── convert_to_production.py   # Model conversion script
├── pyproject.toml            # Poetry dependencies
├── .pre-commit-config.yaml   # Pre-commit hooks
└── README.md                 # This file
```

## Development
Code Quality
Для проверки качества кода в проекте используются:

1. Black: Code formatting

2. isort: Import sorting

3. flake8: Linting

4. pre-commit: Git hooks for quality checks

## Запустить проверку вручную:

```
pre-commit run --all-files
```
Добавление зависимостей
Добавьте новые зависимости с Poetry:

```
poetry add package_name
```
For development dependencies:

```
poetry add --group dev package_name
```

Configuration Management
Этот проект использует Hydra для configuration management. Конфигурации являются иерархическими и могут быть переопределены:

```
python commands.py train model=yolo training=custom data=local
```

Data Versioning
Управление данными осуществляется с помощью DVC. Для отслеживания изменений в данных:

```
dvc add data/new_dataset
git add data/new_dataset.dvc
git commit -m "Add new dataset"
```

## Experiment Tracking
Все эксперименты отслеживаются с помощью MLflow:

1. Hyperparameters

2. Metrics (loss, mAP, IoU, etc.)

3. Model artifacts

4. Training plots

5. Git commit information

View experiments at http://127.0.0.1:8080

## Оптимизация производительности
1. Использование обучения со смешанной точностью для более быстрого обучения на современных графических процессорах

2. Увеличьте размер партии, если у вас достаточно памяти GPU.

3. Используйте несколько рабочих для загрузки данных

4. Рассмотрите возможность использования TensorRT для производственных выводов
