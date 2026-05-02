from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

def create_generators():
    train_datagen = ImageDataGenerator(
        validation_split = val_split,
        rescale=1.0/255.0,
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        validation_split = val_split,
        rescale=1.0/255.0,
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        directory = path,
        seed = seed,
        batch_size = batch_size, 
        class_mode='binary',
        shuffle = True,
        target_size=(img_rows, img_cols),
        subset = 'training'
    )

    val_generator = val_datagen.flow_from_directory(
        directory = path,
        seed = seed,
        batch_size = batch_size, 
        class_mode='binary',
        shuffle = True,
        target_size=(img_rows, img_cols),
        subset = 'validation'
    )

    test_generator = test_datagen.flow_from_directory(
        directory=path_test,
        class_mode='binary',
        seed=seed,
        batch_size=batch_size,
        shuffle=False,
        target_size=(img_rows, img_cols)
    )

    return train_generator, val_generator, test_generator, train_datagen