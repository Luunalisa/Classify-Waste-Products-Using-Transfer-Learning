from data.dataset import prepare_dataset
from data_processing.generators import create_generators
from data_processing.preview import preview_augmentation
from models.feature_model import build_feature_model
from models.fine_tune_model import build_finetune_model
from training.callbacks import get_callbacks
from training.train_feature_model import train_model
from training.train_finetune_model import train_fine_tune_model

from evaluation.evaluate import evaluate_models
from evaluation.visualize import plot_loss, plot_accuracy
from evaluation.visualize import plot_image_with_title

def main():
    prepare_dataset()

    train_gen, val_gen, test_gen, train_datagen = create_generators()

    preview_augmentation(train_datagen)
    

    callbacks_list_ = get_callbacks('O_R_tlearn_vgg16.keras')
    model = build_feature_model()
    train_model(model, train_gen, val_gen, callbacks_list_)

    callbacks_list_ = get_callbacks('O_R_tlearn_fine_tune_vgg16.keras')
    fine_model = build_finetune_model()
    train_fine_tune_model(fine_model, train_gen, val_gen, callbacks_list_)

    evaluate_models()

    plot_loss(model)
    plot_accuracy(model)

    plot_image_with_title(
        image=test_imgs[0].astype('uint8'),
        model_name='Extract Features Model',
        actual_label=test_labels[0],
        predicted_label=predictions_extract_feat_model[0],
    )
    


if __name__ == "__main__":
    main()