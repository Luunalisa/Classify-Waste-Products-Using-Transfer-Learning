import tensorflow as tf

def train_fine_tune_model(model, train_generator, val_generator, callbacks_list_):

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        metrics=['accuracy'])
    
    history = model.fit(
        train_generator, 
        steps_per_epoch=5, 
        epochs=10,
        callbacks = callbacks_list_,   
        validation_data=val_generator, 
        validation_steps=val_generator.samples // batch_size, 
        verbose=1)


    return history