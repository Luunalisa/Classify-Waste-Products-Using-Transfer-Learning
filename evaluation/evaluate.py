
import tensorflow as tf
import numpy as np
import glob
from pathlib import Path
from sklearn import metrics

def evaluate_models():

    extract_feat_model = tf.keras.models.load_model('O_R_tlearn_vgg16.keras')
    fine_tune_model = tf.keras.models.load_model('O_R_tlearn_fine_tune_vgg16.keras')

    IMG_DIM = (150, 150)

    test_files_O = glob.glob('./o-vs-r-split/test/O/*')
    test_files_R = glob.glob('./o-vs-r-split/test/R/*')
    test_files = test_files_O[:50] + test_files_R[:50]

    test_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in test_files]
    test_imgs = np.array(test_imgs)
    test_labels = [Path(fn).parent.name for fn in test_files]

    test_imgs_scaled = test_imgs.astype('float32') / 255

    num2class_lt = lambda l: ['O' if x < 0.5 else 'R' for x in l]

    predictions_extract_feat_model = extract_feat_model.predict(test_imgs_scaled, verbose=0)
    predictions_fine_tune_model = fine_tune_model.predict(test_imgs_scaled, verbose=0)

    predictions_extract_feat_model = num2class_lt(predictions_extract_feat_model)
    predictions_fine_tune_model = num2class_lt(predictions_fine_tune_model)

    print(metrics.classification_report(test_labels, predictions_extract_feat_model))
    print(metrics.classification_report(test_labels, predictions_fine_tune_model))