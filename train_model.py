import warnings
warnings.filterwarnings("ignore")

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import config

EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
MODEL_NAME = config.MODEL_NAME
CLASSES = config.CLASSES
MODEL_PATH = config.MODEL_PATH
TRAIN_DATASET_PATH = config.TRAIN_DATASET_PATH
VALID_DATASET_PATH = config.VALID_DATASET_PATH
MODEL = config.MODEL

train_data = object_detector.DataLoader.from_pascal_voc(
    TRAIN_DATASET_PATH,
    TRAIN_DATASET_PATH,
    CLASSES
)

val_data = object_detector.DataLoader.from_pascal_voc(
    VALID_DATASET_PATH,
    VALID_DATASET_PATH,
    CLASSES
)

spec = model_spec.get(MODEL)

model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=BATCH_SIZE,
    train_whole_model=True,
    epochs=EPOCHS,
    validation_data=val_data
)

model.evaluate(val_data)

model.export(export_dir=MODEL_PATH, tflite_filename=MODEL_NAME)

print('-'*100)
print('Training completed.')
print('See the model folder.')