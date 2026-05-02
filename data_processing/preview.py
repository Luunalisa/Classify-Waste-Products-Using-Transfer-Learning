import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pathlib import Path

def preview_augmentation(train_datagen):
    IMG_DIM = (100, 100)

    train_files = glob.glob('./o-vs-r-split/train/O/*')
    train_files = train_files[:20]
    train_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in train_files]
    train_imgs = np.array(train_imgs)
    train_labels = [Path(fn).parent.name for fn in train_files]

    img_id = 0
    O_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],
                                       batch_size=1)

    O = [next(O_generator) for i in range(0,5)]
    fig, ax = plt.subplots(1,5, figsize=(16, 6))
    print('Labels:', [item[1][0] for item in O])
    l = [ax[i].imshow(O[i][0][0]) for i in range(0,5)]