import tensorflow as tf

NUM_EPOCHS = 50


def compile_model(training_padded, training_labels, testing_padded, testing_labels, vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # vs precision?

    history = model.fit(training_padded, training_labels, epochs=NUM_EPOCHS,
                        validation_data=(testing_padded, testing_labels), verbose=2)

    return model