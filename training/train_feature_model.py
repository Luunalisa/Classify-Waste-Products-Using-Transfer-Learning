import tensorflow as tf

def train_model(model, train_generator, val_generator, callbacks_list_):
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=5,
        epochs=10,
        callbacks=callbacks_list_,
        validation_data=val_generator,
        validation_steps=val_generator.samples // train_generator.batch_size,
        verbose=1
    )

    return history